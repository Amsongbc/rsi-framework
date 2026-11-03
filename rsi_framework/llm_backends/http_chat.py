"""OpenAI-compatible chat completions endpoint chosen entirely by the profile.

base_url and model have no defaults. The key is read from `key_file` (never stored
in records). Images are sent as PNG data URLs. Every request records the model
named by the server, latency, usage as reported, and the request/response digests.
"""
import base64
import json
from pathlib import Path
import time

from rsi_framework.core.io import append_json, canonical_sha256, write_json
from rsi_framework.core.registry import register


class HttpChatBackend:
    kind = 'llm_backend'
    evidence = 'real'
    mailbox = False

    def __init__(self, *, base_url, model, key_file=None, timeout_s=300, max_tokens=None, temperature=None,
                 extra_body=None, headers=None, client=None):
        if not base_url or not model:
            raise ValueError('http_chat needs base_url and model from the profile')
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.key_file = key_file
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_body = dict(extra_body or {})
        self.headers = dict(headers or {})
        self._client = client

    def describe(self):
        return {'block': 'http_chat', 'base_url': self.base_url, 'model': self.model,
                'key_file_configured': bool(self.key_file)}

    def client(self):
        if self._client is None:
            import httpx
            self._client = httpx.Client(timeout=self.timeout_s, trust_env=True)
        return self._client

    def payload(self, request):
        messages = [dict(m) for m in request['messages']]
        if request['images']:
            content = [{'type': 'text', 'text': messages[-1]['content']}]
            for image in request['images']:
                data = base64.b64encode(Path(image['path']).read_bytes()).decode()
                content.append({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + data}})
            messages[-1]['content'] = content
        body = {'model': self.model, 'messages': messages, **self.extra_body}
        if self.max_tokens is not None:
            body['max_tokens'] = self.max_tokens
        if self.temperature is not None:
            body['temperature'] = self.temperature
        return body

    def complete(self, run, request):
        headers = {'Content-Type': 'application/json', **self.headers}
        if self.key_file:
            headers['Authorization'] = 'Bearer ' + Path(self.key_file).read_text().strip()
        body = self.payload(request)
        started = time.monotonic()
        reply = self.client().post(self.base_url + '/chat/completions', json=body, headers=headers)
        data = reply.json()
        row = {'id': request['id'], 'status': reply.status_code, 'latency_s': time.monotonic() - started,
               'request_sha256': canonical_sha256(body), 'model_reported': data.get('model'),
               'usage': data.get('usage'), 'images': [i['sha256'] for i in request['images']]}
        append_json(run.out / 'llm_calls.jsonl', row)
        if reply.status_code != 200:
            raise RuntimeError(f'LLM endpoint returned status {reply.status_code}')
        text = data['choices'][0]['message']['content'] or ''
        text = text.strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        return text

    def respond(self, run, identifier, response):
        write_json(run.out / 'responses' / (identifier + '.json'), response)


@register('llm_backend', 'http_chat', description='OpenAI-compatible endpoint from the profile')
def make(**options):
    return HttpChatBackend(**options)
