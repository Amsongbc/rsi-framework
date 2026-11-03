"""A release binds the whole runnable combination, not a single checkpoint:
environment, policy and LLM backend, tools and protocol, VLA and its checkpoint,
Memory snapshot, and (when training happened) dataset, trainer lineage and the
evaluation that admitted it. Episodes started from a release refuse to run when the
assembled blocks differ from it. Rollback restores the previous combination.
"""
import copy
import json
from pathlib import Path
import time

from rsi_framework.core.io import append_json, canonical_sha256, write_json

COMPONENTS = ('environment', 'llm_backend', 'policy', 'tools', 'protocol', 'vla', 'memory')
VOLATILE = frozenset({'evidence', 'service_metadata_sha256', 'calls'})


def _static(value):
    if isinstance(value, dict):
        return {k: _static(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [_static(v) for v in value]
    return value


def components(assembly):
    described = assembly.describe()
    return {k: _static(described.get(k)) for k in COMPONENTS}


def make_release(profile, assembly, parent=None, extras=None):
    body = {'format': 'rsi-release/1', 'parent': parent, 'profile': copy.deepcopy(profile),
            'components': components(assembly), 'extras': extras or {}}
    body['id'] = 'r' + canonical_sha256(body)[:12]
    body['created'] = time.time()
    return body


def verify_assembly(release, assembly):
    current = components(assembly)
    return [f'{key}: release {json.dumps(release["components"].get(key), sort_keys=True)} '
            f'!= assembled {json.dumps(current.get(key), sort_keys=True)}'
            for key in COMPONENTS if release['components'].get(key) != current.get(key)]


class ReleaseStore:
    def __init__(self, root):
        self.root = Path(root)
        (self.root / 'releases').mkdir(parents=True, exist_ok=True)

    def path(self, release_id):
        return self.root / 'releases' / f'{release_id}.json'

    def save(self, release):
        path = self.path(release['id'])
        if path.exists():
            if json.loads(path.read_text())['components'] != release['components']:
                raise FileExistsError(f'Release {release["id"]} exists with other content')
            return path
        write_json(path, release)
        append_json(self.root / 'events.jsonl', {'time': time.time(), 'event': 'saved', 'release': release['id']})
        return path

    def get(self, release_id):
        return json.loads(self.path(release_id).read_text())

    def state(self):
        path = self.root / 'state.json'
        return json.loads(path.read_text()) if path.exists() else {'active': None, 'history': []}

    def active(self):
        state = self.state()
        return self.get(state['active']) if state['active'] else None

    def activate(self, release_id, *, evaluation=None, memory_revalidation=None, bootstrap=False):
        if not bootstrap:
            if not evaluation or not evaluation.get('passed'):
                raise PermissionError('A release is activated only with a passed evaluation')
            if memory_revalidation is not None and not memory_revalidation.get('passed'):
                raise PermissionError('Carried Memory failed revalidation under this release')
        self.get(release_id)
        state = self.state()
        state['history'].append(release_id)
        state['active'] = release_id
        write_json(self.root / 'state.json', state)
        append_json(self.root / 'events.jsonl', {'time': time.time(), 'event': 'activated', 'release': release_id,
                                                 'bootstrap': bootstrap,
                                                 'evaluation_sha256': canonical_sha256(evaluation) if evaluation else None})
        return state

    def rollback(self, reason):
        state = self.state()
        if len(state['history']) < 2:
            raise RuntimeError('No earlier release to roll back to')
        removed = state['history'].pop()
        state['active'] = state['history'][-1]
        write_json(self.root / 'state.json', state)
        append_json(self.root / 'events.jsonl', {'time': time.time(), 'event': 'rolled_back', 'from': removed,
                                                 'to': state['active'], 'reason': reason})
        return state

    def reject(self, release_id, reason, report=None):
        append_json(self.root / 'events.jsonl', {'time': time.time(), 'event': 'rejected', 'release': release_id,
                                                 'reason': reason, 'report': report})
