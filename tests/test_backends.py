"""LLM backends: request shape of http_chat (no key in records) and the file request/reply
channel of command_queue."""
import base64
import json
from pathlib import Path
import tempfile
import threading
import time
import types
import unittest

from rsi_framework.llm_backends.command_queue import CommandQueue
from rsi_framework.llm_backends.http_chat import HttpChatBackend


class FakeClient:
    def __init__(self, status, content):
        self.status, self.content, self.calls = status, content, []

    def post(self, url, json, headers):
        self.calls.append({'url': url, 'json': json, 'headers': headers})
        body = {'model': 'served-model-name', 'usage': {'prompt_tokens': 10, 'completion_tokens': 3},
                'choices': [{'message': {'content': self.content}}]}
        return types.SimpleNamespace(status_code=self.status, json=lambda: body)


class Backends(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name)
        (self.out / 'responses').mkdir()
        image = self.out / 'cam_head.png'
        image.write_bytes(b'\x89PNG fake')
        self.request = {'id': 'turn0001', 'messages': [{'role': 'system', 'content': 'rules'},
                                                       {'role': 'user', 'content': '{"task": "t"}'}],
                        'images': [{'camera': 'cam_head', 'path': str(image), 'sha256': 'abc'}]}
        self.run = types.SimpleNamespace(out=self.out)

    def tearDown(self):
        self.temp.cleanup()

    def test_http_chat_sends_profile_model_and_images_and_hides_the_key(self):
        key = self.out / 'key'
        key.write_text('secret-test-key\n')
        client = FakeClient(200, '```json\n{"op": "finish", "note": "done"}\n```')
        backend = HttpChatBackend(base_url='http://127.0.0.1:9/v1/', model='configured-model', key_file=str(key), client=client)
        text = backend.complete(self.run, self.request)
        self.assertEqual(json.loads(text), {'op': 'finish', 'note': 'done'})
        call = client.calls[0]
        self.assertEqual(call['url'], 'http://127.0.0.1:9/v1/chat/completions')
        self.assertEqual(call['json']['model'], 'configured-model')
        self.assertEqual(call['headers']['Authorization'], 'Bearer secret-test-key')
        content = call['json']['messages'][-1]['content']
        self.assertEqual(content[1]['image_url']['url'], 'data:image/png;base64,' + base64.b64encode(b'\x89PNG fake').decode())
        record = (self.out / 'llm_calls.jsonl').read_text()
        self.assertNotIn('secret-test-key', record)
        self.assertEqual(json.loads(record)['model_reported'], 'served-model-name')

    def test_http_chat_needs_endpoint_and_model_from_the_profile(self):
        with self.assertRaises(ValueError):
            HttpChatBackend(base_url='', model='m')
        with self.assertRaises(ValueError):
            HttpChatBackend(base_url='http://x', model='')
        backend = HttpChatBackend(base_url='http://x', model='m', client=FakeClient(500, ''))
        with self.assertRaisesRegex(RuntimeError, 'status 500'):
            backend.complete(self.run, self.request)

    def test_command_queue_request_reply(self):
        queue = CommandQueue(mailbox=False, poll_s=0.01, reply_timeout_s=10)

        def controller():
            path = self.out / 'requests/turn0001.json'
            while not path.exists():
                time.sleep(0.01)
            request = json.loads(path.read_text())
            reply = {'op': 'tool', 'tool': 'state', 'args': {}, 'note': f"saw {len(request['images'])} images"}
            tmp = self.out / 'replies/turn0001.tmp'
            tmp.write_text(json.dumps(reply))
            tmp.replace(self.out / 'replies/turn0001.json')
        thread = threading.Thread(target=controller)
        thread.start()
        text = queue.complete(self.run, self.request)
        thread.join()
        self.assertEqual(json.loads(text)['note'], 'saw 1 images')
        self.assertEqual(queue.describe()['api_requests'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
