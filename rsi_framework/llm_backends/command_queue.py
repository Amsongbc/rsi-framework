"""External controller over files.

Mailbox mode (`mailbox=True`) is the original window queue: the oldest
commands/*.json by mtime is accepted (renamed to .accepted, id = file stem) and the
answer goes to responses/<id>.json; window_control.py works with it unchanged.
Request mode writes requests/<id>.json (messages + image files) and waits for
replies/<id>.json written by the controller. Like window_control.py, a controller
must write a temporary file and rename it, so a reply is never read half-written.
"""
import json
import os
import time

from rsi_framework.core.io import write_json
from rsi_framework.core.registry import register


class CommandQueue:
    kind = 'llm_backend'
    evidence = 'real'

    def __init__(self, *, mailbox=True, reply_timeout_s=1200, poll_s=0.25, controller='window_AI'):
        self.mailbox = mailbox
        self.reply_timeout_s = reply_timeout_s
        self.poll_s = poll_s
        self.controller = controller

    def describe(self):
        return {'block': 'command_queue', 'mailbox': self.mailbox, 'controller': self.controller,
                'api_requests': 0, 'model_identity': 'unverified'}

    def next_command(self, run):
        commands = sorted((run.out / 'commands').glob('*.json'), key=lambda p: p.stat().st_mtime_ns)
        if not commands:
            return None
        path = commands[0]
        identifier = path.stem
        raw_command = path.read_text()
        os.replace(path, path.with_suffix('.accepted'))
        return identifier, raw_command

    def complete(self, run, request):
        requests, replies = run.out / 'requests', run.out / 'replies'
        requests.mkdir(exist_ok=True)
        replies.mkdir(exist_ok=True)
        write_json(requests / f"{request['id']}.json", request)
        reply = replies / f"{request['id']}.json"
        deadline = time.monotonic() + self.reply_timeout_s
        while not reply.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"No controller reply for {request['id']}")
            time.sleep(self.poll_s)
        text = reply.read_text()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return text
        return text if isinstance(value, dict) else json.dumps(value)

    def respond(self, run, identifier, response):
        write_json(run.out / 'responses' / (identifier + '.json'), response)


@register('llm_backend', 'command_queue', description='External controller via files; no API calls')
def make(**options):
    return CommandQueue(**options)
