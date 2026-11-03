"""Pure VLA policy: every turn hands the task instruction to the VLA for a full chunk.

No LLM, no tools, no Memory. This is the generic counterpart of the original pure
EWAM baseline; the original baseline itself is reproduced by the compat profile
that launches projects/ewam_comparison/session.py --mode baseline.
"""
import time
import traceback

from rsi_framework.core.io import append_json
from rsi_framework.core.registry import register


class VLAOnly:
    kind = 'policy'
    evidence = 'real'
    evidence_label = 'real'

    def __init__(self, *, model_actions=16, language=None):
        self.model_actions = model_actions
        self.language = language
        self.turns = 0

    def describe(self):
        return {'block': 'vla_only', 'model_actions': self.model_actions, 'language': self.language}

    def prepare_dirs(self, run):
        (run.out / 'responses').mkdir()

    def attach(self, run):
        run.namespace = None

    def next_turn(self, run):
        self.turns += 1
        if hasattr(run.protocol, 'observation_delivered'):
            run.protocol.observation_delivered(run)
        return f'turn{self.turns:04d}'

    def run_turn(self, run, identifier):
        row = {'time': time.time(), 'turn': self.turns, 'id': identifier,
               'command': {'op': 'vla', 'language': self.language, 'model_actions': self.model_actions}}
        try:
            row['output'] = run.protocol.vla_call(run, self.language, self.model_actions)
        except Exception:
            row['error'] = traceback.format_exc()
        row['observation_index'] = run.observe()['observation_index']
        append_json(run.out / 'decisions.jsonl', row)
        if row.get('error'):
            run.stop_reason = 'policy_error'


@register('policy', 'vla_only', description='Pure VLA: task instruction, full chunks, no LLM')
def make(**options):
    return VLAOnly(**options)
