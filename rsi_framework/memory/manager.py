"""Memory modes.

- off: no cross-episode experience is read or produced.
- frozen: one immutable snapshot is read for the whole evaluation.
- learn: episodes run on the current snapshot; afterwards candidates are extracted
  from the records, checked in independent practice episodes and, if they pass a
  rule fixed beforehand, written as the next immutable snapshot. The running
  episode never sees a snapshot that was produced from it.
- legacy_doctor: the original cohorts' `doctor.frozen_memory` value, passed through
  unchanged (compat only).

Lessons keep the original field set (trigger, diagnosis, adaptation, verification,
confidence observed|hypothesis, source_episode), so original validators accept the
`experience` view. A snapshot also records what it is compatible with (task, policy,
tools, VLA); retrieval refuses a mismatch instead of silently reusing it.
"""
import copy
import json
from pathlib import Path
import time

from rsi_framework.core.contracts import ToolSpec
from rsi_framework.core.io import canonical_sha256, read_jsonl, sha256_file, write_json
from rsi_framework.core.registry import register

FORMAT = 'rsi-memory/1'
LESSON_FIELDS = ('trigger', 'diagnosis', 'adaptation', 'verification', 'confidence', 'source_episode')
MEMORY_TOOL = ToolSpec('memory', 'memory', 'Read the frozen experience snapshot for this task. Lessons are '
                       'hypotheses to re-ground in the current scene; old coordinates are not current facts.', {})


def validate_snapshot(snapshot):
    if snapshot.get('format') != FORMAT:
        raise ValueError(f'Unknown memory format: {snapshot.get("format")}')
    lessons = snapshot.get('lessons')
    if not isinstance(lessons, list):
        raise ValueError('Memory snapshot needs a lessons list')
    for lesson in lessons:
        for key in LESSON_FIELDS:
            if not isinstance(lesson.get(key), str) or not lesson[key].strip():
                raise ValueError('Malformed lesson field ' + key)
        if lesson['confidence'] not in ('observed', 'hypothesis'):
            raise ValueError('Lesson confidence must be observed or hypothesis')
    for key in ('task', 'version', 'compatibility'):
        if key not in snapshot:
            raise ValueError('Memory snapshot missing ' + key)
    return snapshot


def experience_view(snapshot):
    """The original {task, lessons, uncertainties} shape."""
    return {'task': snapshot['task'], 'lessons': [{k: l[k] for k in LESSON_FIELDS} for l in snapshot['lessons']],
            'uncertainties': list(snapshot.get('uncertainties', []))}


def compatibility_errors(snapshot, requirements):
    errors = []
    for key, value in requirements.items():
        if value is None:
            continue
        recorded = snapshot['compatibility'].get(key)
        if recorded not in (None, '*') and recorded != value:
            errors.append(f'{key}: snapshot {recorded!r} != current {value!r}')
    return errors


class MemoryOff:
    kind = 'memory'
    mode = 'off'
    evidence = 'real'

    def describe(self):
        return {'block': 'off', 'mode': 'off'}

    def attach(self, run):
        pass

    def legacy_value(self):
        return None

    def tool_specs(self):
        return []

    def call(self, run, name, args):
        raise ValueError(f'Unknown or unavailable tool: {name}')

    def context_text(self, run):
        return ''


class LegacyDoctorMemory(MemoryOff):
    """compat: the original config's doctor.frozen_memory, exactly as given."""

    def __init__(self, *, value=None):
        self.value = value
        self.mode = 'frozen' if value else 'off'

    def describe(self):
        return {'block': 'legacy_doctor', 'mode': self.mode,
                'sha256': canonical_sha256(self.value) if self.value is not None else None}

    def legacy_value(self):
        return self.value


class MemoryFrozen(MemoryOff):
    mode = 'frozen'

    def __init__(self, *, snapshot, sha256=None, retrieval='tool+context', requirements=None):
        self.path = Path(snapshot)
        self.expected_sha256 = sha256
        self.retrieval = retrieval
        self.requirements = dict(requirements or {})
        self.snapshot = None

    def load(self):
        if self.snapshot is None:
            digest = sha256_file(self.path)
            if self.expected_sha256 and digest != self.expected_sha256:
                raise ValueError(f'Memory snapshot hash {digest} != expected {self.expected_sha256}')
            self.snapshot = validate_snapshot(json.loads(self.path.read_text()))
            self.sha256 = digest
        return self.snapshot

    def describe(self):
        snapshot = self.load()
        return {'block': self.mode, 'mode': self.mode, 'version': snapshot['version'], 'sha256': self.sha256,
                'lessons': len(snapshot['lessons']), 'retrieval': self.retrieval}

    def attach(self, run):
        snapshot = self.load()
        if snapshot['task'] != run.case['task']:
            raise ValueError('Experience task mismatch')
        errors = compatibility_errors(snapshot, self.requirements)
        if errors:
            raise ValueError('Memory snapshot is not compatible with this assembly: ' + '; '.join(errors))

    def legacy_value(self):
        return experience_view(self.load())

    def tool_specs(self):
        return [MEMORY_TOOL] if 'tool' in self.retrieval else []

    def call(self, run, name, args):
        if name != 'memory' or 'tool' not in self.retrieval:
            raise ValueError(f'Unknown or unavailable tool: {name}')
        return experience_view(self.load())

    def context_text(self, run):
        if 'context' not in self.retrieval or not self.load()['lessons']:
            return ''
        rows = [f"- [{l['confidence']}] when {l['trigger']}: {l['adaptation']} (verify: {l['verification']})"
                for l in self.load()['lessons']]
        return ('Experience snapshot ' + self.load()['version'] + ' (hypotheses; re-observe and re-ground, '
                'never replay old coordinates):\n' + '\n'.join(rows))


class MemoryLearn(MemoryFrozen):
    """Reads its starting snapshot like `frozen`; produces the next version between episodes."""
    mode = 'learn'

    def __init__(self, *, snapshot, store, sha256=None, retrieval='tool+context', requirements=None,
                 max_lessons=10, promotion=None):
        super().__init__(snapshot=snapshot, sha256=sha256, retrieval=retrieval, requirements=requirements)
        self.store = Path(store)
        self.max_lessons = max_lessons
        self.promotion = dict({'min_cases': 2, 'min_success_delta': 0, 'min_score_delta': 0.0,
                               'no_lost_successes': True}, **(promotion or {}))

    def describe(self):
        return dict(super().describe(), store=str(self.store), promotion=self.promotion)


def candidates_from_episode(episode_dir):
    """Deterministic candidates from what an episode actually recorded.

    Sources: the doctor `diagnose()` notes and decision notes that carry an expected
    change. Only official outcomes and executed actions can make a lesson
    `observed`; everything else stays a `hypothesis`.
    """
    episode_dir = Path(episode_dir)
    summary = json.loads((episode_dir / 'episode_summary.json').read_text())
    if not summary.get('valid'):
        return []
    out = []
    for row in read_jsonl(episode_dir / 'diagnoses.jsonl'):
        verified = row.get('outcome', 'unverified') not in ('unverified', '') and summary.get('success')
        out.append({'trigger': row['observation'], 'diagnosis': row['subgoal'],
                    'adaptation': f"{row['decision']}: {row['expected_change']}",
                    'verification': row['outcome'], 'confidence': 'observed' if verified else 'hypothesis',
                    'source_episode': episode_dir.name,
                    'evidence': {'native_action_index': row['native_action_index'], 'official_success': summary.get('success')}})
    for row in read_jsonl(episode_dir / 'decisions.jsonl'):
        decision = row.get('command') or {}
        if not decision.get('note') or not decision.get('expected_change') or row.get('error'):
            continue
        out.append({'trigger': decision['note'], 'diagnosis': f"turn {row['turn']} {decision.get('op')}",
                    'adaptation': json.dumps({k: decision[k] for k in ('op', 'tool', 'args', 'language', 'model_actions')
                                              if k in decision}, sort_keys=True),
                    'verification': decision['expected_change'],
                    'confidence': 'observed' if summary.get('success') else 'hypothesis',
                    'source_episode': episode_dir.name,
                    'evidence': {'observation_index': row.get('observation_index'), 'official_success': summary.get('success')}})
    return out


def compare_practice(baseline, candidate, rule):
    """Paired comparison of practice summaries (same cases, same order)."""
    if len(baseline) != len(candidate):
        raise ValueError('Practice runs must be paired case by case')
    valid = [(b, c) for b, c in zip(baseline, candidate) if b.get('valid') and c.get('valid')]
    lost = sum(bool(b['success']) and not c['success'] for b, c in valid)
    won = sum(c['success'] and not bool(b['success']) for b, c in valid)
    delta_score = (sum(c.get('official_score') or 0 for _, c in valid) - sum(b.get('official_score') or 0 for b, _ in valid)) / max(1, len(valid))
    passed = (len(valid) >= rule['min_cases'] and won - lost >= rule['min_success_delta']
              and delta_score >= rule['min_score_delta'] and (lost == 0 or not rule['no_lost_successes']))
    return {'cases': len(baseline), 'valid_pairs': len(valid), 'won': won, 'lost': lost,
            'mean_score_delta': round(delta_score, 6), 'rule': rule, 'passed': bool(passed)}


def write_snapshot(store, snapshot):
    """Write an immutable snapshot and index it; returns (path, sha256)."""
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    validate_snapshot(snapshot)
    path = store / f"{snapshot['version']}.json"
    if path.exists():
        raise FileExistsError(f'Memory version {snapshot["version"]} already exists')
    write_json(path, snapshot)
    digest = sha256_file(path)
    index = json.loads((store / 'index.json').read_text()) if (store / 'index.json').exists() else []
    index.append({'version': snapshot['version'], 'sha256': digest, 'parent': snapshot.get('parent'),
                  'lessons': len(snapshot['lessons']), 'created': time.time()})
    write_json(store / 'index.json', index)
    return path, digest


def next_snapshot(parent, lessons, validation, max_lessons):
    snapshot = copy.deepcopy(parent)
    number = int(parent['version'].lstrip('m')) + 1
    snapshot.update(version=f'm{number:04d}', parent=parent['version'],
                    lessons=(parent['lessons'] + lessons)[-max_lessons:],
                    validation=validation)
    return snapshot


def empty_snapshot(task, compatibility, version='m0000'):
    return {'format': FORMAT, 'version': version, 'task': task, 'parent': None,
            'compatibility': dict(compatibility), 'lessons': [], 'uncertainties': []}


@register('memory', 'off', description='No cross-episode experience')
def make_off(**options):
    return MemoryOff()


@register('memory', 'legacy_doctor', description='Original doctor.frozen_memory passed through (compat)')
def make_legacy(**options):
    return LegacyDoctorMemory(**options)


@register('memory', 'frozen', description='One immutable snapshot for the whole evaluation')
def make_frozen(**options):
    return MemoryFrozen(**options)


@register('memory', 'learn', description='Snapshot + candidates, independent practice, next version')
def make_learn(**options):
    return MemoryLearn(**options)
