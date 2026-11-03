"""Swapping environment, robot and VLA through plugins only (tests/plugins/toy.py).

The same framework code that runs RoboDojo + EWAM runs a one-arm toy robot, its
environment and an in-process VLA: LLM-written Python (CaP), structured Agent + VLA,
pure VLA, data building, and contract errors when pieces do not fit.
"""
import json
from pathlib import Path
import tempfile
import unittest

from rsi_framework.core import assembly as asm
from rsi_framework.core.episode import run_episode
from rsi_framework.core.io import read_jsonl
from rsi_framework.data import dataset

CASE = {'task': 'toy_reach', 'eval_seed': 0, 'layout': 3}


def profile(mode, **blocks):
    base = {'name': f'toy_{mode}', 'mode': mode, 'allow_stand_ins': True, 'plugins': ['tests.plugins.toy'],
            'environment': {'block': 'toy_reach'}, 'tools': {'block': 'joint'}, 'memory': {'block': 'off'}}
    base.update(blocks)
    return base


def scripted(*replies):
    return {'block': 'scripted', 'options': {'replies': list(replies)}}


class Plugins(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def run_case(self, name, prof):
        return self.root / name, run_episode(prof, CASE, self.root / name)

    def test_code_as_policy_on_a_new_robot(self):
        cells = ["print(robo.state()['arm'])",
                 "robo.move('arm', [0.2, -0.1, 0.3], steps=5)\nrobo.move('grip', [0.0], steps=2)",
                 "import os"]
        prof = profile('llm_policy',
                       llm_backend=scripted(*[{'op': 'python', 'code': c, 'note': 'cap'} for c in cells],
                                            {'op': 'finish', 'note': 'done'}),
                       protocol={'block': 'tools_only'},
                       policy={'block': 'commands', 'options': {'ops': ['python', 'observe', 'finish'], 'require_id': False}})
        out, summary = self.run_case('cap', prof)
        self.assertTrue(summary['valid'], summary['errors'])
        self.assertEqual(summary['robot'], 'toy_arm')
        self.assertEqual(summary['native_actions_by_owner'], {'primitive': 7, 'vla': 0})
        self.assertEqual(sorted(p.name for p in (out / 'observations/0000').glob('*.png')), ['front.png'])
        decisions = read_jsonl(out / 'decisions.jsonl')
        self.assertIn('Policy audit', decisions[2]['error'])
        self.assertAlmostEqual(read_jsonl(out / 'native_actions.jsonl')[5]['robot_state']['arm'][2], 0.3)

    def test_agent_with_a_different_vla(self):
        prof = profile('agent_vla', vla={'block': 'toy_vla'},
                       llm_backend=scripted({'op': 'vla', 'language': 'reach left', 'model_actions': 4, 'note': 'v'},
                                            {'op': 'tool', 'tool': 'move', 'args': {'key': 'grip', 'target': [0.5]},
                                             'note': 't'},
                                            {'op': 'vla', 'language': None, 'model_actions': 6, 'note': 'v'},
                                            {'op': 'finish', 'note': 'done'}),
                       protocol={'block': 'vla_prefix', 'options': {'max_model_actions': 6}},
                       policy={'block': 'commands', 'options': {'ops': ['tool', 'vla', 'observe', 'finish'], 'require_id': False}})
        out, summary = self.run_case('agent', prof)
        self.assertTrue(summary['valid'], summary['errors'])
        self.assertEqual(summary['native_actions_by_owner'], {'primitive': 6, 'vla': 20})
        self.assertEqual([c['native_actions'] for c in read_jsonl(out / 'vla_calls.jsonl')], [8, 12])
        manifest = dataset.build([out], self.root / 'data', selection='all_executed')
        self.assertEqual(manifest['by_owner'], {'vla': 20, 'primitive': 6})
        sample = read_jsonl(self.root / 'data/samples.jsonl')[0]
        self.assertEqual(set(sample['action']), {'arm', 'grip'})
        self.assertEqual(sample['video'], {'front': 'native_video/front.mp4'})
        self.assertEqual(sample['language'], 'reach left')

    def test_pure_vla_on_a_new_environment(self):
        prof = profile('pure_vla', vla={'block': 'toy_vla'}, protocol={'block': 'vla_prefix'},
                       policy={'block': 'vla_only', 'options': {'model_actions': 6}})
        _, summary = self.run_case('pure', prof)
        self.assertEqual(summary['native_actions_issued'], 60)
        self.assertTrue(summary['valid'])

    def test_minimal_command_line(self):
        import os, subprocess, sys
        from rsi_framework import REPO_ROOT
        env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), PYTHONDONTWRITEBYTECODE='1')
        argv = [sys.executable, '-B', '-m', 'rsi_framework', 'run', '--env', 'toy_reach', '--vla', 'toy_vla',
                '--controller', 'none', '--plugin', 'tests.plugins.toy', '--allow-stand-ins']
        proc = subprocess.run(argv, cwd=self.root, env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        summaries = list((self.root / 'runs').glob('*/episode_summary.json'))
        self.assertEqual(len(summaries), 1)
        summary = json.loads(summaries[0].read_text())
        self.assertEqual((summary['task'], summary['eval_seed'], summary['layout']), ('toy_reach', 0, 3))
        self.assertEqual(summary['mode'], 'pure_vla')

    def test_mismatches_are_reported_before_running(self):
        cases = {
            'vla for another robot': (profile('agent_vla', vla={'block': 'ewam_service', 'options': {'service_root': '/x'}},
                                              llm_backend=scripted(), protocol={'block': 'vla_prefix'},
                                              policy={'block': 'commands', 'options': {'ops': ['vla']}}), 'cameras'),
            'tools for another environment': (profile('llm_policy', tools={'block': 'pyrua'}, llm_backend=scripted(),
                                                      protocol={'block': 'tools_only'},
                                                      policy={'block': 'commands', 'options': {'ops': ['tool']}}), 'work with'),
            'robot the environment does not publish': (profile('llm_policy', robot={'block': 'arx_x5_bimanual'},
                                                               llm_backend=scripted(), protocol={'block': 'tools_only'},
                                                               policy={'block': 'commands', 'options': {'ops': ['tool']}}),
                                                       'publishes robot toy_arm'),
        }
        for label, (prof, message) in cases.items():
            with self.subTest(label):
                with self.assertRaisesRegex(asm.AssemblyError, message):
                    asm.assemble(prof)


if __name__ == '__main__':
    unittest.main(verbosity=2)
