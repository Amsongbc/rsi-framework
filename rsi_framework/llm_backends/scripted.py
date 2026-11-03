"""Fixed replies, for replaying recorded decisions and for tests.

Replies are command objects, raw strings, or (in tests) callables of
(run, request) that build a reply from the current observation. They are returned
in order; running out is an error, never a silent default.
"""
import json
from pathlib import Path

from rsi_framework.core.io import append_json, write_json
from rsi_framework.core.registry import register


class ScriptedBackend:
    kind = 'llm_backend'
    evidence = 'replay'
    mailbox = False

    def __init__(self, *, replies=None, file=None):
        if file:
            replies = json.loads(Path(file).read_text())
        self.replies = list(replies or [])
        self.used = 0

    def describe(self):
        return {'block': 'scripted', 'replies': len(self.replies), 'evidence': self.evidence}

    def complete(self, run, request):
        if self.used >= len(self.replies):
            raise RuntimeError('Scripted backend has no reply left')
        reply = self.replies[self.used]
        self.used += 1
        (run.out / 'requests').mkdir(exist_ok=True)
        write_json(run.out / 'requests' / f"{request['id']}.json", request)
        append_json(run.out / 'llm_calls.jsonl', {'id': request['id'], 'backend': 'scripted',
                                                  'images': [i['sha256'] for i in request['images']]})
        if callable(reply):
            reply = reply(run, request)
        return reply if isinstance(reply, str) else json.dumps(reply)

    def respond(self, run, identifier, response):
        write_json(run.out / 'responses' / (identifier + '.json'), response)


@register('llm_backend', 'scripted', evidence='replay', description='Fixed replies (replay / tests)')
def make(**options):
    return ScriptedBackend(**options)
