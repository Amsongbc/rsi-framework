"""Offline state-machine checks, not claimed as benchmark results."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from doctor_robot import robot_class, kin, RobodojoOfficialColourRobot, ToolError
from recording import RecordedEnv
from learning import gate, validate_memory


class Environment:
    def __init__(self):
        self.n = 0
        self.action = {f'{arm}_{part}_joint_state': [0.]*6 if part == 'arm' else [1.]
                       for arm in ('left', 'right') for part in ('arm', 'ee')}

    def get_status(self):
        return {'step': self.n, 'step_limit': 10}

    def get_obs(self):
        state = dict(self.action)
        for arm in ('left', 'right'):
            t = kin.fk(arm, state[f'{arm}_arm_joint_state'])
            state[f'{arm}_ee_pose'] = list(t[:3, 3]) + list(kin.quat_wxyz(t[:3, :3]))
        return {'vision': {c: {'color': np.full((32,32,3), self.n, np.uint8)}
                           for c in ('cam_head', 'cam_left_wrist', 'cam_right_wrist')},
                'state': state, 'instruction': 'Move the object'}

    def step(self, action):
        assert self.n < 10
        self.n += 1; self.action = action
        return self.get_obs(), None, self.episode_over(), {}

    def episode_over(self):
        return self.n >= 10


class Policy:
    def call(self, op, **kwargs):
        return {'inference_s': 0, 'instruction': 'Move the object',
                'actions': [{f'{arm}_{part}_joint_state': [0.]*6 if part == 'arm' else [1.]
                             for arm in ('left','right') for part in ('arm','ee')} for _ in range(48)]}


def blocked(fn):
    try: fn()
    except ToolError: return
    raise AssertionError('Expected a protocol rejection')


def state_machine(path):
    env = RecordedEnv(Environment(), path)
    context = {'out': path, 'env': env, 'ewam': Policy(), 'session_id': 'offline',
               'doctor_phase': 'hybrid', 'ewam_calls': 0, 'ewam_attempts': 0,
               'toolkit': SimpleNamespace(state=SimpleNamespace(record=lambda *a:None))}
    robo = robot_class(context)(SimpleNamespace())
    packet = robo.ewam_propose()
    assert packet['physical_steps'] == 0 and env.issued_actions == 0
    assert packet['robot_only_fk']['fk_position_check_passed']
    blocked(lambda: robo.ewam_execute(packet['prediction_id'], 'progress', 'RGB', 2))
    blocked(lambda: robo._stateful('move', {}, lambda _:None))
    blocked(lambda: robo.ewam_propose())
    context['proposal']['delivered_to_model'] = True
    robo.ewam_execute(packet['prediction_id'], 'progress', 'RGB', 2)
    assert env.issued_actions == 6 and context['proposal'] is None
    blocked(lambda: robo.ewam_propose())
    blocked(lambda: robo.ewam_execute(packet['prediction_id'], 'progress', 'RGB', 2))
    context['awaiting_ewam_observation'] = False
    packet2 = robo.ewam_propose(); context['proposal']['delivered_to_model'] = True
    # A primitive invalidates the old plan even if its motion implementation errors.
    with patch.object(RobodojoOfficialColourRobot, '_stateful', return_value='primitive-ok'):
        assert robo._stateful('move', {}, lambda _:None) == 'primitive-ok'
    blocked(lambda: robo.ewam_execute(packet2['prediction_id'], 'progress', 'RGB', 2))
    env.step(env.env.action)  # actual new state for next proposal
    packet3 = robo.ewam_propose(); context['proposal']['delivered_to_model'] = True
    robo.ewam_execute(packet3['prediction_id'], 'progress', 'RGB', 16)
    assert env.issued_actions == 10 and env.actions_by_owner == {'primitive': 1, 'ewam': 9}
    env.close_recording()
    assert len((path/'native_actions.jsonl').read_text().splitlines()) == 11
    context['doctor_phase'] = 'direct'
    blocked(lambda: robo.ewam_propose())


def main():
    with TemporaryDirectory() as name: state_machine(Path(name))
    from pyrualean.prompt import api_reference
    reference = api_reference(robot_class({}), primitives='no-vla')
    assert 'ewam_propose' in reference and 'ewam_execute' in reference and 'move_to' in reference
    assert gate([.3,.3,.3]) == 'hybrid'
    assert gate([0,.15,.2]) == 'direct'
    for scores in ([1,1], [1,1,float('nan')], [1,1,1.1]):
        try: gate(scores)
        except ValueError: pass
        else: raise AssertionError('Invalid official scores accepted')
    try: validate_memory({'task':'wrong','lessons':[]}, 'test')
    except ValueError: pass
    else: raise AssertionError('Cross-task memory accepted')
    output = {'passed': True, 'checks': ['proposal_has_zero_physical_steps',
        'preview_requires_next_model_turn', 'no_duplicate_same_state_inference',
        'suffix_invalid_after_motion', 'postexecution_visual_gate',
        'shared_native_budget_and_owner_recording', 'direct_branch_disables_ewam',
        'gate_rejects_invalid_outcomes', 'task_memory_isolation'], 'benchmark_episode':False}
    Path(__file__).with_name('offline_checks.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output))


if __name__ == '__main__': main()
