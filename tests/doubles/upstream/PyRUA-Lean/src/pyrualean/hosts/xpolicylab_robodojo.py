"""RoboDojo host with a deterministic synthetic bimanual task (TEST DOUBLE).

Same entry names as the upstream host (Cell, lay_out, boot, make_backend). The
task hides a joint-space goal per (eval_seed, layout); the official result scores
how close both arms end to it. Observations are public RGB, joint state, gripper
and link6 pose only.
"""
import dataclasses
import json
from pathlib import Path
import types

import numpy as np

from pyrualean.hosts import robodojo_kinematics as kin

ARMS = ('left', 'right')
CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')
STEP_LIMITS = {'classify_objects': 150, 'stack_blocks': 120, 'insert_key': 120,
               'push_T': 120, 'cover_blocks': 120, 'build_tower': 120}
IMAGE_SHAPE = (24, 32, 3)


@dataclasses.dataclass
class Cell:
    suite: str
    task: str
    seed: int = 0
    cuda_device: int = 0


def lay_out(cell, out, **kwargs):
    bench = Path(out) / 'bench'
    (bench / 'scripts').mkdir(parents=True, exist_ok=True)
    (bench / 'upstream_assets').mkdir(exist_ok=True)
    assets = bench / 'Assets'
    if not assets.is_symlink() and not assets.exists():
        assets.symlink_to(bench / 'upstream_assets', target_is_directory=True)
    (bench / 'scripts/eval_policy.sh').write_text('#!/bin/sh\nKIT_ARGS=""\necho evaluate\n')
    return bench


class Bridge:
    def __init__(self, env):
        self.env = env

    def verdict(self, timeout_s=60):
        success, score = self.env.result()
        return {'success': success, 'score': score}


class SyntheticRoboDojoEnv:
    def __init__(self, cell, bench, eval_seed):
        self.cell, self.bench, self.eval_seed = cell, Path(bench), int(eval_seed)
        self.n = 0
        self.limit = STEP_LIMITS.get(cell.task, 120)
        rng = np.random.default_rng(1000 * self.eval_seed + int(cell.seed))
        self.goal = {arm: rng.uniform(-0.4, 0.4, 6).round(3) for arm in ARMS}
        self.joints = {arm: np.zeros(6) for arm in ARMS}
        self.grip = {arm: 1.0 for arm in ARMS}
        self.concluded = False
        self.bridge = Bridge(self)
        self.instruction = f'Synthetic {cell.task}: bring both arms to the hidden pose.'

    def _image(self, index):
        base = np.concatenate([self.joints['left'], self.joints['right'], [self.grip['left'], self.grip['right']]])
        image = np.zeros(IMAGE_SHAPE, np.uint8)
        image[..., 0] = np.uint8((self.n * 7 + index * 40) % 256)
        image[..., 1] = np.uint8(int(abs(base).sum() * 50) % 256)
        image[2:6, 2:2 + len(base), 2] = np.uint8(np.clip((base + 1) * 120, 0, 255))
        return image

    def get_obs(self):
        state = {}
        for arm in ARMS:
            pose = kin.fk(arm, self.joints[arm])
            state[arm + '_arm_joint_state'] = self.joints[arm].copy()
            state[arm + '_ee_joint_state'] = np.array([self.grip[arm]])
            state[arm + '_ee_pose'] = np.r_[pose[:3, 3], kin.quat_wxyz(pose[:3, :3])]
        return {'vision': {camera: {'color': self._image(i)} for i, camera in enumerate(CAMERAS)},
                'state': state, 'instruction': self.instruction}

    def get_status(self):
        return {'step': self.n, 'step_limit': self.limit}

    def step(self, action):
        if self.n >= self.limit:
            raise RuntimeError('Synthetic environment step beyond its limit')
        for arm in ARMS:
            self.joints[arm] = np.asarray(action[arm + '_arm_joint_state'], float).reshape(6).copy()
            self.grip[arm] = float(np.asarray(action[arm + '_ee_joint_state'], float).reshape(-1)[0])
        self.n += 1
        return self.get_obs(), 0.0, self.episode_over(), {}

    def episode_over(self):
        return self.concluded or self.n >= self.limit

    def result(self):
        error = float(np.mean([np.abs(self.joints[arm] - self.goal[arm]).mean() for arm in ARMS]))
        score = round(max(0.0, 1.0 - error / 0.4), 4)
        return bool(score >= 0.9), score

    def conclude(self):
        if self.concluded:
            return
        self.concluded = True
        success, score = self.result()
        path = self.bench / 'eval_result' / self.cell.task / '_result.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'eval_time': 1, 'details': {str(self.cell.seed): {
            'layout_id': int(self.cell.seed), 'success': success, 'score': score}}}, indent=2) + '\n')


class StateCache:
    def __init__(self):
        self.steps = []

    def record(self, obs, status):
        self.steps.append(int(status.get('step', 0)))


class Toolkit:
    def __init__(self, env):
        self._primitives = types.SimpleNamespace(env=env, sam3_client=None, _last_obs=None)
        self.state = StateCache()

    def episode_over(self):
        return self._primitives.env.episode_over()

    def conclude(self):
        self._primitives.env.conclude()


class Daemon:
    def __init__(self, name):
        self.name, self.running = name, True

    def alive(self):
        return self.running

    def stop(self):
        self.running = False


class Backend:
    def __init__(self, toolkit):
        self.toolkit = toolkit


def boot(cell, out, *, env_only=True, depth=False, source_root=None, eval_seed=0, boot_timeout_s=900):
    bench = lay_out(cell, out)
    env = SyntheticRoboDojoEnv(cell, bench, eval_seed)
    return Toolkit(env), [Daemon('evaluation_client')]


def make_backend(toolkit):
    return Backend(toolkit)
