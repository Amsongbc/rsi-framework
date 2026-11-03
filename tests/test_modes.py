"""New modes on the shared runner: LLM policy (no VLA), Agent + frozen VLA, pure VLA,
plus the combination checks, Memory compatibility, release verification and the
dataset alignment. Offline: upstream test double, fake EWAM, scripted LLM replies.
"""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from rsi_framework.core import assembly as asm
from rsi_framework.core.io import read_jsonl, sha256_file, write_json
from rsi_framework.data import dataset
from rsi_framework.memory import manager as memory
from tests import support

CASE = {'task': 'classify_objects', 'eval_seed': 0, 'layout': 10}


def lesson(text):
    return {'trigger': text, 'diagnosis': 'synthetic', 'adaptation': 'reobserve, then continue the VLA segment',
            'verification': 'arm visibly closer', 'confidence': 'hypothesis', 'source_episode': 'seed'}


class Modes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = Path(tempfile.mkdtemp(prefix='rsif-modes-')).resolve()
        cls.projects = support.make_workspace(cls.temp)
        cls.service_dir = support.short_dir('rsif-m-')
        cls.ewam = support.FakeEwam(cls.service_dir / 'svc')
        cls.env = support.environment(cls.projects)
        cls.snapshot = cls.temp / 'memory_m0000.json'
        snap = memory.empty_snapshot('classify_objects', {'environment': 'pyrua_robodojo'})
        snap['lessons'] = [lesson('the arm stalls before the goal')]
        write_json(cls.snapshot, snap)

    @classmethod
    def tearDownClass(cls):
        cls.ewam.stop()
        shutil.rmtree(cls.service_dir, ignore_errors=True)
        shutil.rmtree(cls.temp, ignore_errors=True)

    def base(self, mode, replies=None, **overrides):
        profile = {'name': f'test_{mode}', 'mode': mode, 'allow_stand_ins': True,
                   'environment': {'block': 'pyrua_robodojo', 'perception_gpu': 0, 'options': {'simulator_gpu': 0}},
                   'tools': {'block': 'pyrua'}, 'memory': {'block': 'off'}, 'learning': {'enabled': False}}
        if replies is not None:
            profile['llm_backend'] = {'block': 'scripted', 'options': {'replies': replies}}
        profile.update(overrides)
        return profile

    def run_profile(self, profile, name):
        path = self.temp / 'profiles' / f'{name}.json'
        path.parent.mkdir(exist_ok=True)
        write_json(path, profile)
        out = self.temp / 'runs' / name
        argv = [support.PYTHON, '-B', '-m', 'rsi_framework', 'run', '--profile', str(path), '--task', CASE['task'],
                '--eval-seed', str(CASE['eval_seed']), '--layout', str(CASE['layout']), '--out', str(out)]
        proc = subprocess.run(argv, env=self.env, capture_output=True, text=True, timeout=300)
        summary = out / 'episode_summary.json'
        self.assertTrue(summary.exists(), proc.stdout[-3000:] + proc.stderr[-3000:])
        return out, json.loads(summary.read_text())

    def test_llm_policy_runs_without_vla(self):
        replies = [{'op': 'tool', 'tool': 'state', 'args': {}, 'note': 'read state'},
                   {'op': 'tool', 'tool': 'move_to', 'args': {'arm': 'left', 'joints': [0.2, 0, 0, 0, 0, 0]},
                    'note': 'move left arm', 'expected_change': 'left joint 0 at 0.2'},
                   {'op': 'vla', 'language': 'reach', 'note': 'not offered here'},
                   {'op': 'tool', 'tool': 'memory', 'args': {}, 'note': 'memory is off'},
                   {'op': 'tool', 'tool': 'set_gripper', 'args': {'arm': 'left', 'opening': 0.3}, 'note': 'close a bit'},
                   {'op': 'finish', 'note': 'stop'}]
        profile = self.base('llm_policy', replies,
                            protocol={'block': 'tools_only', 'options': {'observe_after_motion': True}},
                            policy={'block': 'commands', 'options': {'ops': ['tool', 'observe', 'finish'], 'require_id': False}})
        out, summary = self.run_profile(profile, 'llm_policy')
        self.assertTrue(summary['valid'])
        self.assertEqual(summary['stop_reason'], 'planner_finish')
        self.assertEqual(summary['native_actions_by_owner'], {'primitive': 9, 'ewam': 0})
        self.assertIsNone(summary['vla'])
        self.assertFalse((out / 'ewam_service_events.jsonl').exists(), 'LLM policy must not touch a VLA service')
        self.assertEqual(summary['evidence'], 'software_synthetic')
        self.assertEqual(summary['block_evidence']['llm_backend'], 'replay')
        rows = read_jsonl(out / 'decisions.jsonl')
        self.assertIn('Unknown operation: vla', rows[2]['error'])
        self.assertIn('Unknown or unavailable tool: memory', rows[3]['error'])
        request = json.loads((out / 'requests/turn0002.json').read_text())
        self.assertEqual(len(request['images']), 3)
        user = json.loads(request['messages'][1]['content'])
        self.assertEqual(set(user), {'task', 'turn', 'observation', 'protocol', 'tools', 'recent'})
        self.assertEqual(set(user['observation']['robot_state']),
                         {f'{a}_{k}' for a in ('left', 'right') for k in ('arm_joint_state', 'ee_joint_state', 'ee_pose')})

    def test_agent_vla_frozen_memory(self):
        replies = [{'op': 'tool', 'tool': 'memory', 'args': {}, 'note': 'read experience'},
                   {'op': 'vla', 'language': 'reach the goal', 'model_actions': 8, 'note': 'VLA first',
                    'expected_change': 'arms move'},
                   {'op': 'tool', 'tool': 'move_to', 'args': {'arm': 'right', 'joints': [0, 0.1, 0, 0, 0, 0]},
                    'note': 'correct right arm', 'expected_change': 'right joint 1 at 0.1'},
                   {'op': 'vla', 'language': None, 'model_actions': 16, 'note': 'continue', 'expected_change': 'closer'},
                   {'op': 'finish', 'note': 'done'}]
        profile = self.base('agent_vla', replies,
                            vla={'block': 'ewam_service', 'options': {'service_root': str(self.service_dir / 'svc')}},
                            protocol={'block': 'vla_prefix', 'options': {}},
                            policy={'block': 'commands', 'options': {'ops': ['tool', 'vla', 'observe', 'finish'], 'require_id': False}},
                            memory={'block': 'frozen', 'options': {'snapshot': str(self.snapshot)}})
        out, summary = self.run_profile(profile, 'agent_vla')
        self.assertTrue(summary['valid'])
        self.assertEqual(summary['native_actions_by_owner'], {'primitive': 6, 'ewam': 72})
        calls = read_jsonl(out / 'vla_calls.jsonl')
        self.assertEqual([c['native_actions'] for c in calls], [24, 48])
        self.assertEqual([c['discarded_native_actions'] for c in calls], [24, 0])
        self.assertEqual(summary['memory']['sha256'], sha256_file(self.snapshot))
        self.assertEqual(json.loads((out / 'responses/turn0001.json').read_text())['output']['lessons'][0]['trigger'],
                         'the arm stalls before the goal')
        prompt = json.loads((out / 'requests/turn0002.json').read_text())['messages'][1]['content']
        self.assertIn('the arm stalls before the goal', prompt)
        self.assertEqual(summary['vla']['contract']['stride'], 3)
        self.assertNotIn('trainer', json.loads((out / 'run_config.json').read_text())['blocks'])

        # Dataset: each sample pairs action k with the pre-action frame k-1.
        manifest = dataset.build([out], self.temp / 'data_agent', selection='all_executed')
        samples = read_jsonl(self.temp / 'data_agent/samples.jsonl')
        frames = read_jsonl(out / 'native_actions.jsonl')
        self.assertEqual(manifest['by_owner'], {'vla': 72, 'primitive': 6})
        for sample in samples[:5] + samples[-5:]:
            self.assertEqual(sample['pre_state'], frames[sample['frame_index'] - 1]['robot_state'])
            self.assertEqual(sample['action'], frames[sample['frame_index']]['action'])
        self.assertEqual({s['language'] for s in samples if s['owner'] == 'vla'}, {'reach the goal', frames[0]['instruction']})
        self.assertEqual(manifest['segments'], 3)

    def test_pure_vla_until_budget(self):
        profile = self.base('pure_vla', None,
                            vla={'block': 'ewam_service', 'options': {'service_root': str(self.service_dir / 'svc')}},
                            tools={'block': 'pyrua', 'options': {'perception': 'none'}},
                            protocol={'block': 'vla_prefix', 'options': {}},
                            policy={'block': 'vla_only', 'options': {'model_actions': 16}})
        out, summary = self.run_profile(profile, 'pure_vla')
        self.assertEqual(summary['native_actions_by_owner'], {'primitive': 0, 'ewam': 150})
        self.assertEqual(len(read_jsonl(out / 'decisions.jsonl')), 4)

    def test_memory_task_mismatch_is_refused(self):
        other = self.temp / 'memory_other_task.json'
        write_json(other, memory.empty_snapshot('stack_blocks', {}))
        profile = self.base('agent_vla', [{'op': 'finish', 'note': 'x'}],
                            vla={'block': 'ewam_service', 'options': {'service_root': str(self.service_dir / 'svc')}},
                            protocol={'block': 'vla_prefix'}, policy={'block': 'commands', 'options': {
                                'ops': ['tool', 'vla', 'observe', 'finish'], 'require_id': False}},
                            memory={'block': 'frozen', 'options': {'snapshot': str(other)}})
        _, summary = self.run_profile(profile, 'memory_mismatch')
        self.assertEqual(summary['stop_reason'], 'infrastructure_error')
        self.assertIn('Experience task mismatch', summary['errors'][-1])

    def test_combinations_are_checked(self):
        good = self.base('llm_policy', [], protocol={'block': 'tools_only'},
                         policy={'block': 'commands', 'options': {'ops': ['tool', 'finish']}})
        asm.validate(good)
        cases = {
            'llm_policy with a VLA': dict(good, vla={'block': 'ewam_service', 'options': {'service_root': 'x'}}),
            'llm_policy offering vla': dict(good, policy={'block': 'commands', 'options': {'ops': ['vla']}}),
            'agent_vla without VLA': dict(good, mode='agent_vla', protocol={'block': 'vla_prefix'}),
            'frozen profile naming a trainer': dict(good, learning={'enabled': False, 'trainer': {'block': 'external_command'}}),
            'learning without trainer': dict(good, learning={'enabled': True, 'dataset': {}, 'evaluation': {}, 'release': {}}),
            'window protocol on an automatic backend': dict(good, mode='agent_vla', protocol={'block': 'window_full'},
                                                            vla={'block': 'ewam_service', 'options': {'service_root': 'x'}}),
            'ack_visual outside window': dict(good, policy={'block': 'commands', 'options': {'ops': ['ack_visual']}}),
            'frozen memory without snapshot': dict(good, memory={'block': 'frozen', 'options': {}}),
        }
        for label, profile in cases.items():
            with self.subTest(label):
                with self.assertRaises(asm.AssemblyError):
                    asm.validate(profile)
        strict = dict(copy.deepcopy(good), allow_stand_ins=False)
        assembled = asm.assemble(strict)
        with self.assertRaises(asm.AssemblyError):
            assembled.require_evidence()

    def test_release_refuses_changed_memory(self):
        from rsi_framework.releases.manifest import make_release, verify_assembly
        snap = self.temp / 'memory_release.json'
        shutil.copyfile(self.snapshot, snap)
        profile = self.base('agent_vla', [], vla={'block': 'ewam_service', 'options': {'service_root': str(self.service_dir / 'svc')}},
                            protocol={'block': 'vla_prefix'}, policy={'block': 'commands', 'options': {'ops': ['vla', 'finish']}},
                            memory={'block': 'frozen', 'options': {'snapshot': str(snap)}})
        release = make_release(profile, asm.assemble(profile))
        self.assertEqual(verify_assembly(release, asm.assemble(profile)), [])
        changed = json.loads(snap.read_text())
        changed['lessons'].append(lesson('edited after release'))
        write_json(snap, changed)
        problems = verify_assembly(release, asm.assemble(profile))
        self.assertTrue(problems and problems[0].startswith('memory:'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
