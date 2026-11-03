"""RobotSpec and its consistency checks."""
import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class RobotSpec:
    """obs = {'vision': {camera: {'color': HxWx3 uint8}}, 'state': {key: array}, 'instruction': str};
    native action = {key: array of action_keys[key] values}."""
    name: str
    cameras: tuple
    state_keys: tuple
    action_keys: dict
    description: str = ''
    kind = 'robot'
    evidence = 'real'

    def describe(self):
        return {'block': self.name, 'cameras': list(self.cameras), 'state_keys': list(self.state_keys),
                'action_keys': dict(self.action_keys)}

    def observation_problems(self, obs):
        problems = [f'missing camera {c}' for c in self.cameras if c not in obs.get('vision', {})]
        problems += [f'missing state {k}' for k in self.state_keys if k not in obs.get('state', {})]
        return problems

    def action_problems(self, action):
        problems = [f'missing action key {k}' for k in self.action_keys if k not in action]
        problems += [f'{k} has {np.asarray(action[k]).size} values, expected {n}'
                     for k, n in self.action_keys.items() if k in action and np.asarray(action[k]).size != n]
        return problems
