"""InProcessVLA: wrap a Python-callable policy as a VLA block.

Subclass, set `contract`, implement predict(images, state, language) returning a list
of native action dicts keyed by contract['action_keys']. The base class selects the
contract's cameras and state keys from the observation, seeds `self.rng` per episode
(same rule as EWAM) and checks every returned action against the contract.
"""
import numpy as np


class InProcessVLA:
    kind = 'vla'
    evidence = 'real'
    contract = {}

    def __init__(self, **options):
        self.options = options
        self.rng = None

    def describe(self):
        return {'block': type(self).__name__, 'contract': self.contract, 'options': self.options,
                'checkpoint_ref': self.options.get('checkpoint_ref'), 'evidence': self.evidence}

    def predict(self, images, state, language):
        raise NotImplementedError

    def attach(self, run):
        return None

    def reset(self, run):
        self.rng = np.random.default_rng(100000 + run.case['eval_seed'] * 1000 + run.case['layout'])
        run.context['lease_acquired'] = True

    def release(self, run):
        pass

    def infer(self, run, obs, language):
        language = language or str(obs.get('instruction') or '')
        images = {c: np.asarray(obs['vision'][c]['color']).copy() for c in self.contract['cameras']}
        state = {k: np.asarray(obs['state'][k]).copy() for k in self.contract['state_keys']}
        actions = self.predict(images, state, language)
        for action in actions:
            missing = [k for k in self.contract['action_keys'] if k not in action]
            if missing:
                raise ValueError(f'{type(self).__name__} action lacks {missing}')
        return {'actions': actions, 'instruction': language}
