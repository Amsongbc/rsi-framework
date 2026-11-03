"""Recorder blocks: provenance, run configuration, observation snapshots, episode summary.

`snapshot` is shared by every recorder and writes the original window observation
format (one PNG per robot camera with SHA-256, observation.json, latest.json); the protocol adds
its own fields. The generic recorder adds a framework manifest and a summary that
names the assembled blocks and the evidence label of the run.
"""
import hashlib
import time

import numpy as np

from rsi_framework import REPO_ROOT, __version__
from rsi_framework.compat import baseline
from rsi_framework.core.io import plain, sha256_file, write_json
from rsi_framework.core.registry import register

def snapshot(run):
    from PIL import Image
    obs = run.env.get_obs()
    folder = run.out / 'observations' / f'{run.observations:04d}'
    folder.mkdir(parents=True)
    paths, hashes = {}, {}
    for camera in run.robot.cameras:
        path = folder / (camera + '.png')
        Image.fromarray(np.asarray(obs['vision'][camera]['color'])[..., :3].astype(np.uint8)).save(path)
        paths[camera] = str(path)
        hashes[camera] = hashlib.sha256(path.read_bytes()).hexdigest()
    status = run.env.get_status()
    value = dict(observation_index=run.observations,
                 instruction=str(obs.get('instruction') or obs.get('instructions') or ''),
                 robot_state=plain(run.robo.state()), step=int(status.get('step', 0)),
                 issued_actions=run.env.issued_actions, native_action_index=run.env.issued_actions,
                 step_limit=run.env.initial_limit,
                 native_budget_remaining=max(0, run.env.initial_limit - max(run.env.issued_actions, int(status.get('step', 0)))),
                 images=paths, image_sha256=hashes, python_cells=run.cells,
                 **run.protocol.observation_fields(run))
    run.observations += 1
    write_json(folder / 'observation.json', value)
    write_json(run.out / 'latest.json', value)
    return value


def framework_sources():
    package = REPO_ROOT / 'rsi_framework'
    return sorted(p for p in package.rglob('*') if p.is_file() and p.suffix in ('.py', '.json')
                  and '__pycache__' not in p.parts)


def official_outcome(run, verdict):
    return run.env_block.outcome(run, verdict)


class GenericRecorder:
    kind = 'recorder'
    evidence = 'real'

    def describe(self):
        return {'block': 'generic'}

    def prepare(self, run):
        write_json(run.out / 'framework_manifest.json', {
            'framework_version': __version__, 'profile': run.assembly.profile,
            'original_fingerprint': baseline.load_manifest()['fingerprint'],
            'framework_sources': [{'path': str(p.relative_to(REPO_ROOT)), 'sha256': sha256_file(p)}
                                  for p in framework_sources()]})

    def run_config(self, run):
        write_json(run.out / 'run_config.json', {
            'profile': run.assembly.profile.get('name'), 'mode': run.assembly.profile.get('mode'),
            'case': run.case, 'blocks': run.assembly.describe(), 'evidence': run.assembly.evidence(),
            'native_limit': run.env.initial_limit, 'wall_budget_s': run.protocol.wall_budget_s})

    def snapshot(self, run):
        return snapshot(run)

    def summarize(self, run, verdict, policy_end):
        raw, details, valid = official_outcome(run, verdict)
        evidence = run.assembly.evidence()
        summary = dict(profile=run.assembly.profile.get('name'), mode=run.assembly.profile.get('mode'),
                       task=run.case['task'], eval_seed=run.case['eval_seed'], layout=run.case['layout'],
                       valid=valid, success=bool(details[0]['success']) if valid else None,
                       official_score=details[0].get('score') if valid else None,
                       outcome_class='valid_task_success' if valid and details[0]['success'] else
                       'valid_task_failure' if valid else 'technical_interruption',
                       client_verdict=verdict, client_results=raw, stop_reason=run.stop_reason, errors=run.errors,
                       evidence=evidence['overall'], block_evidence=evidence['blocks'],
                       robot=run.robot.name, memory=run.memory.describe(),
                       vla=run.vla.describe() if run.vla else None,
                       python_cells=run.cells, turns=getattr(run.policy, 'turns', None),
                       vla_calls=run.context.get('vla_calls', run.context.get('ewam_calls')),
                       native_actions_issued=getattr(getattr(run, 'env', None), 'issued_actions', 0),
                       native_actions_by_owner=getattr(getattr(run, 'env', None), 'actions_by_owner', {}),
                       first_native_action_owner=getattr(getattr(run, 'env', None), 'first_native_owner', None),
                       startup_s=None if run.ready_at is None else run.ready_at - run.started,
                       control_wall_s=None if run.ready_at is None else policy_end - run.ready_at,
                       total_wall_s=time.monotonic() - run.started)
        write_json(run.out / 'episode_summary.json', summary)
        return summary


@register('recorder', 'generic', description='Framework manifest, window-format snapshots, generic summary')
def make(**options):
    return GenericRecorder()
