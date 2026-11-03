"""Responses transport for PyRUA's original OpenAI agent loop, with replay evidence."""
from __future__ import annotations

# Export portability change: resolve external projects via PROJECTS_ROOT.
import os as _export_os

import base64
import copy
import hashlib
import json
from pathlib import Path
import time

import httpx


class ResponsesTransport:
    def __init__(self, api, out_dir, *, client=None):
        self.api = dict(api)
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out/'api_requests').mkdir(exist_ok=True)
        (self.out/'api_images').mkdir(exist_ok=True)
        self.key = Path(api['key_file']).read_text().strip()
        if not self.key or any(c.isspace() for c in self.key):
            raise ValueError('Key file must contain one nonempty bare API key')
        self.client = client or httpx.Client(
            headers={'Authorization': 'Bearer '+self.key},
            trust_env=api.get('trust_env', True), timeout=None)
        self.history = []
        self.seen = 0
        self.requests = 0
        self.responses = 0
        self.on_visual_response = None
        self.pending_visual_input = False
        self.compactions = 0
        self.static_instructions = []

    def safe(self, value):
        if isinstance(value, str):
            return value.replace(self.key, '[REDACTED]')
        if isinstance(value, list):
            return [self.safe(v) for v in value]
        if isinstance(value, dict):
            return {k:self.safe(v) for k,v in value.items()}
        return value

    def externalize_images(self, value):
        if isinstance(value, str) and value.startswith('data:image/'):
            prefix, encoded = value.split(',',1)
            data = base64.b64decode(encoded, validate=True)
            digest = hashlib.sha256(data).hexdigest()
            extension = '.png' if 'png' in prefix else '.jpg'
            target = self.out/'api_images'/(digest+extension)
            if not target.exists():
                target.write_bytes(data)
            return {'image_ref':str(target.relative_to(self.out)), 'sha256':digest}
        if isinstance(value, list):
            return [self.externalize_images(v) for v in value]
        if isinstance(value, dict):
            return {k:self.externalize_images(v) for k,v in value.items()}
        return value

    def __call__(self, ignored_chat_url, body, timeout_s):
        if body['model'] != self.api['model']:
            raise ValueError('Model substitution is prohibited')
        messages = body['messages']
        new_images = any(isinstance(m.get('content'),list) and
                         any(p.get('type')=='image_url' for p in m['content'])
                         for m in messages[self.seen:])
        self.pending_visual_input = self.pending_visual_input or new_images
        if len(messages) < self.seen:
            raise ValueError('Conversation unexpectedly shrank')
        for message in messages[self.seen:]:
            role = message['role']
            if role == 'assistant':
                continue  # retain original Responses output, including encrypted reasoning
            if role == 'tool':
                self.history.append({'type':'function_call_output',
                                     'call_id':message['tool_call_id'],
                                     'output':message['content']})
                continue
            content = message.get('content') or ''
            if isinstance(content, str):
                content = [{'type':'input_text','text':content}]
            else:
                converted = []
                for part in content:
                    if part['type']=='text':
                        converted.append({'type':'input_text','text':part['text']})
                    elif part['type']=='image_url':
                        converted.append({'type':'input_image','image_url':part['image_url']['url']})
                    else:
                        raise ValueError('Unsupported input content part')
                content = converted
            self.history.append({'role':role,'content':content})
        self.seen = len(messages)
        if not self.requests:
            self.static_instructions = copy.deepcopy([
                item for item in self.history if item.get('role') in ('system', 'developer')])
        payload = {'model':self.api['model'], 'input':copy.deepcopy(self.history),
                   'tools':[{'type':'function',**tool['function']} for tool in body.get('tools',[])],
                   'tool_choice':body.get('tool_choice','auto'), 'parallel_tool_calls':False,
                   'max_output_tokens':self.api['max_output_tokens'],
                   'reasoning':{'effort':self.api['reasoning_effort']},
                   'include':['reasoning.encrypted_content'], 'store':False}
        if self.api.get('context_compaction', False):
            payload['context_management'] = [{'type':'compaction',
                'compact_threshold':self.api['compact_threshold']}]
        self.requests += 1
        tick = time.monotonic()
        serial = f'{self.requests:04d}'
        (self.out/'api_requests'/f'{serial}.json').write_text(json.dumps(
            self.safe(self.externalize_images(payload)),ensure_ascii=False,indent=2)+'\n')
        event = {'request':self.requests, 'time':time.time(), 'input_items':len(self.history),
                 'model_requested':self.api['model'], 'reasoning_effort':self.api['reasoning_effort']}
        try:
            response = self.client.post(self.api['base_url'].rstrip('/')+'/responses',
                                        json=payload, timeout=timeout_s)
            event['status'] = response.status_code
            try:
                data = response.json()
            except ValueError:
                data = {'error':{'type':'non_json_response','message':response.text[:4000]}}
            event['request_id'] = response.headers.get('x-request-id')
            event['response'] = self.safe(data)
        except Exception as exc:
            event.update(status=None,transport_error=type(exc).__name__)
            raise
        finally:
            event['latency_s'] = time.monotonic()-tick
            with (self.out/'responses_events.jsonl').open('a') as handle:
                handle.write(json.dumps(event,ensure_ascii=False)+'\n')
        if response.status_code != 200:
            return response.status_code,self.safe(data)
        if not str(data.get('model','')).startswith(self.api['model']):
            raise ValueError('API response model differs from the requested model')
        self.responses += 1
        self.history.extend(data.get('output',[]))
        if self.api.get('context_compaction', False):
            indices = [i for i,item in enumerate(self.history) if item.get('type')=='compaction']
            if indices and any(item.get('type')=='compaction' for item in data.get('output',[])):
                before_items = len(self.history)
                before_bytes = len(json.dumps(self.history).encode())
                # Server-side compaction permits dropping items before its latest
                # opaque item. Keep the unchanged system/developer API contract
                # explicit; never summarize or split an outstanding tool call.
                self.history = copy.deepcopy(self.static_instructions) + self.history[indices[-1]:]
                self.compactions += 1
                row = {'request':self.requests, 'time':time.time(),
                    'event':'server_compaction_applied', 'compaction_number':self.compactions,
                    'before_items':before_items, 'after_items':len(self.history),
                    'before_json_bytes':before_bytes,
                    'after_json_bytes':len(json.dumps(self.history).encode()),
                    'static_instruction_items':len(self.static_instructions),
                    'output_types':[item.get('type') for item in data.get('output',[])],
                    'documentation':'https://developers.openai.com/api/docs/guides/compaction'}
                with (self.out/'compaction_events.jsonl').open('a') as handle:
                    handle.write(json.dumps(row)+'\n')
        if self.pending_visual_input and self.on_visual_response is not None:
            self.on_visual_response()
        self.pending_visual_input=False
        text, calls = [], []
        for item in data.get('output',[]):
            if item['type']=='function_call':
                calls.append({'id':item['call_id'],'type':'function',
                              'function':{'name':item['name'],'arguments':item['arguments']}})
            elif item['type']=='message':
                text.extend(p['text'] for p in item.get('content',[]) if p['type']=='output_text')
        usage = data.get('usage') or {}
        return 200, {'choices':[{'finish_reason':'tool_calls' if calls else 'stop',
                                'message':{'role':'assistant','content':'\n'.join(text),'tool_calls':calls}}],
                     'usage':{'prompt_tokens':usage.get('input_tokens',0),
                              'completion_tokens':usage.get('output_tokens',0),
                              'prompt_tokens_details':usage.get('input_tokens_details',{}),
                              'completion_tokens_details':usage.get('output_tokens_details',{})}}

    def close(self):
        self.client.close()
