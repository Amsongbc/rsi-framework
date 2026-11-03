"""A different robot, environment and VLA plugged in without touching rsi_framework.

Everything a new setup needs is here: a RobotSpec (one 3-joint arm, one camera), a
native environment with six methods plus a 4-line adapter, and a VLA that implements
predict(). Profiles load it with "plugins": ["tests.plugins.toy"].
"""
import hashlib

import numpy as np

from rsi_framework.core.registry import register
from rsi_framework.environments.base import NativeEnvAdapter
from rsi_framework.robots.spec import RobotSpec
from rsi_framework.vla_backends.base import InProcessVLA

TOY_ARM = RobotSpec(name='toy_arm', cameras=('front',), state_keys=('arm', 'grip'),
                    action_keys={'arm': 3, 'grip': 1}, description='One 3-joint arm and a gripper')


class ToyReachNative:
    """Bring the arm to a hidden pose; scored at the end like an official evaluator."""

    def __init__(self, case):
        rng = np.random.default_rng(1000 * case['eval_seed'] + case['layout'])
        self.goal = rng.uniform(-0.5, 0.5, 3)
        self.arm, self.grip, self.n, self.limit, self.concluded = np.zeros(3), 1.0, 0, 60, False

    def get_obs(self):
        image = np.zeros((12, 16, 3), np.uint8)
        image[..., 0] = np.uint8(self.n * 9 % 256)
        image[2:5, 2:5, 1] = np.uint8(np.clip((self.arm[0] + 1) * 120, 0, 255))
        return {'vision': {'front': {'color': image}}, 'state': {'arm': self.arm.copy(), 'grip': np.array([self.grip])},
                'instruction': 'Toy: bring the arm to the hidden pose'}

    def get_status(self):
        return {'step': self.n, 'step_limit': self.limit}

    def step(self, action):
        self.arm = np.asarray(action['arm'], float).reshape(3).copy()
        self.grip = float(np.asarray(action['grip']).reshape(-1)[0])
        self.n += 1
        return self.get_obs(), 0.0, self.episode_over(), {}

    def episode_over(self):
        return self.concluded or self.n >= self.limit

    def conclude(self):
        self.concluded = True

    def result(self):
        score = round(max(0.0, 1.0 - float(np.abs(self.arm - self.goal).mean()) / 0.5), 4)
        return {'success': score >= 0.9, 'score': score}


@register('robot', 'toy_arm', evidence='software_synthetic')
def make_robot(**options):
    return TOY_ARM


@register('environment', 'toy_reach', evidence='software_synthetic', description='Example: toy reaching task')
class ToyReach(NativeEnvAdapter):
    robot = 'toy_arm'
    name = 'toy_reach'
    evidence = 'software_synthetic'
    defaults = {'task': 'toy_reach', 'eval_seed': 0, 'layout': 3}

    def make_native(self, run):
        return ToyReachNative(run.case)


@register('vla', 'toy_vla', evidence='software_synthetic', description='Example: language-hashed reaching policy')
class ToyVLA(InProcessVLA):
    evidence = 'software_synthetic'
    contract = {'cameras': ['front'], 'state_keys': ['arm', 'grip'], 'action_keys': {'arm': 3, 'grip': 1}, 'stride': 2}

    def predict(self, images, state, language):
        target = np.frombuffer(hashlib.sha256(language.encode()).digest()[:3], np.uint8) / 255.0 - 0.5
        arm = np.asarray(state['arm'], float)
        actions = []
        for _ in range(12):
            arm = arm + (target - arm) * 0.2 + self.rng.normal(0, 1e-3, 3)
            actions.append({'arm': arm.copy(), 'grip': np.asarray(state['grip'], float).copy()})
        return actions
