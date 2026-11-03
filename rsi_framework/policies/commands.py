"""Command loop: one command per turn, executed, answered with output and a fresh observation.

Commands are JSON objects `{"id", "op", "note", ...}` with ops:
  python (code)  observe  ack_visual (...)  tool (tool, args)  vla (language, model_actions)  finish
A profile lists the allowed ops. With the mailbox channel and ops
{python, observe, ack_visual, finish} this is window_session.Session.loop and
Session.python with the same checks, messages, records and response files.
With an LLM channel the backend writes the command; the response is fed back on
the next turn by the context builder.
"""
import contextlib
import io
import json
import time
import traceback

import numpy as np

from rsi_framework.core.io import append_json
from rsi_framework.core.registry import register
from rsi_framework.policies import context as context_builder

WINDOW_OPS = ('python', 'observe', 'ack_visual', 'finish')


class CommandLoop:
    kind = 'policy'
    evidence = 'real'

    def __init__(self, *, channel, ops=WINDOW_OPS, require_note=True, require_id=True, max_cells=None,
                 max_turns=None, idle_s=0.2, history_turns=6, images='latest', system_prompt=None):
        self.channel = channel
        self.ops = tuple(ops)
        self.require_note = require_note
        self.require_id = require_id
        self.max_cells = max_cells
        self.max_turns = max_turns
        self.idle_s = idle_s
        self.history_turns = history_turns
        self.images = images
        self.system_prompt = system_prompt
        self.history = []
        self.turns = 0

    def describe(self):
        return {'block': 'commands', 'channel': self.channel.describe(), 'ops': list(self.ops),
                'require_note': self.require_note, 'max_cells': self.max_cells, 'max_turns': self.max_turns,
                'history_turns': self.history_turns, 'images': self.images}

    @property
    def evidence_label(self):
        return getattr(self.channel, 'evidence', 'real')

    def prepare_dirs(self, run):
        for name in ('commands', 'responses', 'workspace'):
            (run.out / name).mkdir()

    def attach(self, run):
        make_builtins = run.tools.sandbox()[0]
        if self.max_cells is None:
            self.max_cells = run.protocol.max_cells
        run.namespace = {'__builtins__': make_builtins(run.out / 'workspace'), 'robo': run.robo, 'np': np}

    def next_turn(self, run):
        if getattr(self.channel, 'mailbox', False):
            return self.channel.next_command(run)
        if self.max_turns is not None and self.turns >= self.max_turns:
            run.stop_reason = 'max_turns'
            return None
        self.turns += 1
        identifier = f'turn{self.turns:04d}'
        request = context_builder.build(run, self, identifier)
        if hasattr(run.protocol, 'observation_delivered'):
            run.protocol.observation_delivered(run)
        raw = self.channel.complete(run, request)
        return identifier, raw

    def run_turn(self, run, turn):
        identifier, raw_command = turn
        response = {}
        command = None
        try:
            command = json.loads(raw_command)
            if not isinstance(command, dict):
                raise ValueError('Command must be an object')
            if not self.require_id and 'id' not in command:
                command = dict(id=identifier, **command)
            append_json(run.out / 'commands.jsonl', dict(time=time.time(), **command))
            if self.require_note and not str(command.get('note', '')).strip():
                raise ValueError('Reviewable command note is required')
            if command.get('id') != identifier:
                raise ValueError('Command ID must equal filename stem')
            op = command['op']
            if op not in self.ops:
                raise ValueError(f'Unknown operation: {op}')
            if op == 'python':
                response['output'] = self.python(run, command['code'])
            elif op == 'observe':
                response['output'] = None
            elif op == 'ack_visual':
                response['output'] = run.protocol.acknowledge(run, command)
            elif op == 'tool':
                response['output'] = run.call_tool(command['tool'], command.get('args', {}))
            elif op == 'vla':
                response['output'] = run.protocol.vla_call(run, command.get('language'), command.get('model_actions'))
            elif op == 'finish':
                run.stop_reason = 'planner_finish'
                response['output'] = {'finishing': True}
        except Exception:
            response['error'] = traceback.format_exc()
            append_json(run.out / 'command_errors.jsonl', dict(id=identifier, raw_command=raw_command, **response))
        response['observation'] = run.observe()
        self.channel.respond(run, identifier, response)
        if not getattr(self.channel, 'mailbox', False):
            row = dict(turn=self.turns, id=identifier, command=command, output=response.get('output'),
                       error=response.get('error'), observation_index=response['observation']['observation_index'])
            append_json(run.out / 'decisions.jsonl', row)
            self.history.append(row)

    def python(self, run, code):
        _, audit_source, blocked_names, finished = run.tools.sandbox()
        if run.cells >= self.max_cells:
            raise RuntimeError(f'{self.max_cells}-cell budget exhausted')
        flags = audit_source(code, workspace=run.out / 'workspace')
        blocked = blocked_names(code)
        if flags or blocked:
            raise ValueError(f'Policy audit flags={flags}, blocked={blocked}')
        run.cells += 1
        path = run.out / 'workspace' / f'cell_{run.cells:03d}.py'
        path.write_text(code)
        stdout, stderr = io.StringIO(), io.StringIO()
        error = None
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                exec(compile(code, str(path), 'exec'), run.namespace)
            except finished as exc:
                run.stop_reason = exc.reason
            except Exception:
                error = traceback.format_exc()
        result = dict(stdout=stdout.getvalue(), stderr=stderr.getvalue(), error=error)
        append_json(run.out / 'python_cells.jsonl', dict(index=run.cells, code=code, **result))
        return result


@register('policy', 'commands', description='Command loop over a mailbox or an LLM backend')
def make(**options):
    return CommandLoop(**options)
