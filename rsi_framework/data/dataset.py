"""Executed actions -> samples `(pre-action observation, language) -> native action`.

RecordedEnv writes one row per native action AFTER executing it (frame k holds
action k and the state after it; frame 0 is the initial state). A sample therefore
pairs action k with the observation of frame k-1, never with the post-action state.

Lineage per sample: episode, frame, owner (vla / primitive), the call that produced
it (VLA prediction or primitive ledger record), the language actually sent to the
VLA (empty for tools), the task instruction, the episode's official result and the
evidence label of the run. Only executed actions exist in the record, so discarded
proposal tails never become samples. Actions of a failed call, or of a primitive
record that raised, are rejected with a reason. Segments break at a change of
owner, call or language.

The dataset version is frozen as samples.jsonl + manifest.json with hashes. Model
specific conversion (EWAM's stride-3 model actions, normalisation) belongs to the
trainer plugin, which reads the native-level samples.
"""
import json
from pathlib import Path

from rsi_framework.core.io import canonical_sha256, read_jsonl, sha256_file, write_json
from rsi_framework.core.registry import register

SELECTIONS = ('success_only', 'valid_all', 'all_executed')


def _vla_calls(episode):
    """Executed VLA prefixes as (first_frame, count, call_id, language, error)."""
    out = []
    plans = {row['prediction_id']: row for row in read_jsonl(episode / 'ewam_plans.jsonl') if 'prediction_id' in row}
    for row in read_jsonl(episode / 'ewam_calls.jsonl'):
        if 'prediction_id' in row and row['prediction_id'] in plans:       # window doctor protocol
            plan = plans[row['prediction_id']]
            out.append((plan['native_action_index'] + 1, row['native_actions'], row['prediction_id'],
                        plan.get('instruction_override') or plan['result'].get('instruction'), row.get('error')))
    hybrid_plans = [r for r in read_jsonl(episode / 'ewam_plans.jsonl') if 'start_native_action' in r]
    hybrid_calls = [r for r in read_jsonl(episode / 'ewam_calls.jsonl') if 'reports' in r]
    for plan, call in zip(hybrid_plans, hybrid_calls):                    # EWAM-first hybrid robot
        executed = sum(report['executed_native_actions'] for report in call['reports'])
        out.append((plan['start_native_action'] + 1, executed, f"ewam_call_{plan['call']}",
                    plan.get('instruction_override') or plan['result'].get('instruction'), call.get('error')))
    for row in read_jsonl(episode / 'vla_calls.jsonl'):                    # framework vla_prefix protocol
        out.append((row['start_native_action'] + 1, row['native_actions'], row['call_id'],
                    row['language'] if row['language'] is not None else row.get('instruction'), row.get('error')))
    return out


def _primitive_calls(episode, frames):
    """Primitive ledger records mapped onto primitive-owned frames in order."""
    records = [r for r in read_jsonl(episode / 'primitive_calls.jsonl') if r.get('stateful')]
    primitive_frames = [f for f in frames if f['owner'] == 'primitive']
    out, cursor = {}, 0
    for index, record in enumerate(records):
        for frame in primitive_frames[cursor:cursor + record['env_steps']]:
            out[frame['frame_index']] = (f'primitive_{index:04d}_{record["name"]}', record.get('ok'), record.get('error'))
        cursor += record['env_steps']
    return out


def episode_samples(episode_dir, evidence=None):
    episode = Path(episode_dir)
    summary = json.loads((episode / 'episode_summary.json').read_text())
    frames = read_jsonl(episode / 'native_actions.jsonl')
    by_frame = {}
    for first, count, call_id, language, error in _vla_calls(episode):
        for frame_index in range(first, first + count):
            by_frame[frame_index] = ('vla', call_id, language or '', error)
    primitive = _primitive_calls(episode, frames)
    videos = {p.stem: f'native_video/{p.name}' for p in sorted((episode / 'native_video').glob('*.mp4'))}
    samples, rejected = [], []
    segment, last_key = 0, None
    for previous, frame in zip(frames, frames[1:]):
        index = frame['frame_index']
        if frame['owner'] in ('ewam', 'vla'):
            owner, call_id, language, error = by_frame.get(index, ('vla', None, '', 'no matching VLA call record'))
        else:
            call_id, ok, error = primitive.get(index, (None, None, 'no matching primitive record'))
            owner, language = 'primitive', ''
            error = error or (None if ok else 'primitive call not ok')
        key = (owner, call_id, language)
        if key != last_key:
            segment += 1
            last_key = key
        row = {'episode': episode.name, 'frame_index': index, 'pre_frame_index': previous['frame_index'],
               'owner': owner, 'call_id': call_id, 'language': language, 'task_instruction': frame['instruction'],
               'segment': f'{episode.name}#{segment}', 'pre_state': previous['robot_state'], 'action': frame['action'],
               'post_state': frame['robot_state'], 'video': videos,
               'episode_valid': summary.get('valid'), 'episode_success': summary.get('success'),
               'episode_score': summary.get('official_score'),
               'evidence': evidence or summary.get('evidence', 'unlabelled')}
        if error:
            rejected.append(dict(episode=episode.name, frame_index=index, owner=owner, reason=str(error).splitlines()[-1]))
        else:
            samples.append(row)
    return samples, rejected


def build(episode_dirs, out_dir, *, selection='success_only', owners=('vla', 'primitive'), version=None,
          evidence=None):
    if selection not in SELECTIONS:
        raise ValueError(f'selection must be one of {SELECTIONS}')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    kept, rejected, episodes = [], [], []
    for episode in sorted(Path(e) for e in episode_dirs):
        samples, bad = episode_samples(episode, evidence)
        summary = json.loads((episode / 'episode_summary.json').read_text())
        include = (selection == 'all_executed' or (selection == 'valid_all' and summary.get('valid'))
                   or (selection == 'success_only' and summary.get('valid') and summary.get('success')))
        episodes.append({'episode': episode.name, 'path': str(episode), 'valid': summary.get('valid'),
                         'success': summary.get('success'), 'score': summary.get('official_score'),
                         'included': bool(include), 'native_actions': sorted(
                             (k, v) for k, v in (summary.get('native_actions_by_owner') or {}).items())})
        rejected += bad
        if include:
            kept += [s for s in samples if s['owner'] in owners]
            rejected += [dict(episode=episode.name, frame_index=s['frame_index'], owner=s['owner'],
                              reason='owner not selected') for s in samples if s['owner'] not in owners]
        else:
            rejected.append(dict(episode=episode.name, frame_index=None, owner=None,
                                 reason=f'episode excluded by selection {selection}'))
    with (out / 'samples.jsonl').open('w') as handle:
        for sample in kept:
            handle.write(json.dumps(sample, ensure_ascii=False) + '\n')
    with (out / 'rejected.jsonl').open('w') as handle:
        for row in rejected:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest = {'format': 'rsi-dataset/1', 'version': version or 'd' + canonical_sha256(episodes)[:12],
                'selection': selection, 'owners': list(owners), 'alignment': 'action k with pre-action frame k-1',
                'episodes': episodes, 'samples': len(kept), 'rejected': len(rejected),
                'samples_sha256': sha256_file(out / 'samples.jsonl'),
                'segments': len({s['segment'] for s in kept}),
                'by_owner': {o: sum(s['owner'] == o for s in kept) for o in owners}}
    write_json(out / 'manifest.json', manifest)
    manifest['manifest_sha256'] = sha256_file(out / 'manifest.json')
    return manifest


class NativeSampleDataset:
    kind = 'dataset'
    evidence = 'real'

    def __init__(self, *, selection='success_only', owners=('vla', 'primitive')):
        self.selection, self.owners = selection, tuple(owners)

    def describe(self):
        return {'block': 'native_samples', 'selection': self.selection, 'owners': list(self.owners)}

    def build(self, episode_dirs, out_dir, version=None):
        return build(episode_dirs, out_dir, selection=self.selection, owners=self.owners, version=version)


@register('dataset', 'native_samples', description='Executed native actions aligned to pre-action frames')
def make(**options):
    return NativeSampleDataset(**options)
