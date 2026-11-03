"""EWAM served by projects/ewam_comparison/ewam_service.py (or a compatible service).

The client is the original EwamClient. Every call is recorded in
ewam_service_events.jsonl exactly like window_session's RecordedEwam; failures are
also appended to the run's errors. Reset uses the original seed rule.
"""
import json
from pathlib import Path
import time
import traceback

from rsi_framework.compat import paths
from rsi_framework.core.io import append_json, sha256_file
from rsi_framework.core.registry import register

# EWAM's own contract (checked against the RobotSpec at assembly); other VLAs declare theirs.
ACTION_CONTRACT = {'cameras': ['cam_head', 'cam_left_wrist', 'cam_right_wrist'],
                   'state_keys': ['left_arm_joint_state', 'left_ee_joint_state', 'right_arm_joint_state', 'right_ee_joint_state'],
                   'action_keys': {'left_arm_joint_state': 6, 'left_ee_joint_state': 1,
                                   'right_arm_joint_state': 6, 'right_ee_joint_state': 1},
                   'stride': 3, 'model_actions_per_chunk': 16, 'native_actions_per_chunk': 48,
                   'action_type': 'absolute joint targets'}


def episode_seed(eval_seed, layout):
    return 100000 + eval_seed * 1000 + layout


class RecordedClient:
    """Same recording semantics as window_session's RecordedEwam."""

    def __init__(self, service_root, run):
        paths.ensure_importable()
        from ewam_service import EwamClient
        self.client = EwamClient(service_root)
        self.run = run

    def call(self, op, **kwargs):
        started = time.monotonic()
        error = None
        try:
            return self.client.call(op, **kwargs)
        except Exception:
            error = traceback.format_exc()
            self.run.errors.append(error)
            raise
        finally:
            append_json(self.run.out / 'ewam_service_events.jsonl', dict(
                time=time.time(), op=op, session_id=kwargs.get('session_id'),
                wall_s=time.monotonic() - started, error=error))


class EwamService:
    kind = 'vla'
    contract = ACTION_CONTRACT

    def __init__(self, *, service_root, checkpoint_ref=None, checkpoint_path=None):
        self.service_root = str(service_root)
        self.checkpoint_ref = checkpoint_ref
        self.checkpoint_path = checkpoint_path
        self.client = None
        self.evidence = 'real'

    def metadata(self):
        path = Path(self.service_root) / 'model.json'
        return json.loads(path.read_text()) if path.exists() else None

    def describe(self):
        meta = self.metadata() or {}
        path = Path(self.service_root) / 'model.json'
        return {'block': 'ewam', 'service_root': self.service_root, 'contract': ACTION_CONTRACT,
                'checkpoint_ref': self.checkpoint_ref or meta.get('checkpoint_sha256'),
                'checkpoint_path': self.checkpoint_path,
                'service_metadata_sha256': sha256_file(path) if path.exists() else None,
                'evidence': self.evidence}

    def attach(self, run):
        if (self.metadata() or {}).get('test_double'):
            self.evidence = 'software_synthetic'
        self.client = RecordedClient(self.service_root, run)
        return self.client

    def reset(self, run):
        self.client.call('reset', session_id=run.context['session_id'],
                         seed=episode_seed(run.case['eval_seed'], run.case['layout']))
        run.context['lease_acquired'] = True

    def release(self, run):
        self.client.call('release', session_id=run.context['session_id'])

    def infer(self, run, obs, language):
        from ewam_service import public_observation
        return self.client.call('infer', session_id=run.context['session_id'], obs=public_observation(obs, language))


@register('vla', 'ewam', aliases=('ewam_service',), description='EWAM over the original Unix-socket service')
def make(**options):
    return EwamService(**options)
