"""Improvement round P0 -> D1 -> Memory practice -> data -> train -> P1 -> revalidation ->
evaluation -> release (or not) -> D2, and the frozen round that must train nothing.

Every block here is a stand-in (test double environment, fake EWAM, synthetic
trainer, scripted LLM); the test checks the mechanics and the gates, not learning.
"""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from rsi_framework.core.io import read_jsonl, write_json
from rsi_framework.memory import manager as memory
from rsi_framework.releases.manifest import ReleaseStore
from tests import support

REPLIES = [
    {'op': 'tool', 'tool': 'memory', 'args': {}, 'note': 'read experience'},
    {'op': 'vla', 'language': 'reach the target pose', 'model_actions': 16, 'note': 'arms far from target',
     'expected_change': 'both arms move toward the target'},
    {'op': 'tool', 'tool': 'move_to', 'args': {'arm': 'left', 'joints': [0.1, 0, 0, 0, 0, 0]},
     'note': 'left arm overshoots after the VLA segment', 'expected_change': 'left joint 0 settles at 0.1'},
    {'op': 'vla', 'language': None, 'model_actions': 8, 'note': 'continue with the task instruction',
     'expected_change': 'closer to the target'},
    {'op': 'finish', 'note': 'budget for this check used'},
]


def cases(*layouts):
    return [{'task': 'classify_objects', 'eval_seed': 0, 'layout': layout} for layout in layouts]


class Round(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = Path(tempfile.mkdtemp(prefix='rsif-round-')).resolve()
        cls.projects = support.make_workspace(cls.temp)
        cls.service_dir = support.short_dir('rsif-r-')
        cls.ewam = support.FakeEwam(cls.service_dir / 'svc')
        cls.env = support.environment(cls.projects)

    @classmethod
    def tearDownClass(cls):
        cls.ewam.stop()
        shutil.rmtree(cls.service_dir, ignore_errors=True)
        shutil.rmtree(cls.temp, ignore_errors=True)

    def profile(self, name, learn=True):
        snapshot = self.temp / f'{name}_m0000.json'
        write_json(snapshot, memory.empty_snapshot('classify_objects', {'environment': 'pyrua_robodojo'}))
        profile = {'name': name, 'mode': 'agent_vla', 'allow_stand_ins': True,
                   'environment': {'block': 'pyrua_robodojo', 'perception_gpu': 0, 'options': {'simulator_gpu': 0}},
                   'vla': {'block': 'ewam_service', 'options': {'service_root': str(self.service_dir / 'svc')}},
                   'llm_backend': {'block': 'scripted', 'options': {'replies': REPLIES}},
                   'tools': {'block': 'pyrua'}, 'protocol': {'block': 'vla_prefix'},
                   'policy': {'block': 'commands', 'options': {'ops': ['tool', 'vla', 'observe', 'finish'], 'require_id': False}}}
        if learn:
            profile['memory'] = {'block': 'learn', 'options': {
                'snapshot': str(snapshot), 'store': str(self.temp / f'{name}_memory_store'),
                'promotion': {'min_cases': 1, 'min_success_delta': 0, 'min_score_delta': -1.0, 'no_lost_successes': True}}}
            profile['learning'] = {
                'enabled': True,
                'dataset': {'block': 'native_samples', 'options': {'selection': 'valid_all'}},
                'trainer': {'block': 'synthetic_target'},
                'evaluation': {'rule': None},
                'release': {'serve': {'command': ['{python}', '-B', str(support.REPO / 'tests/doubles/fake_ewam_service.py'),
                                                  '--root', '{root}', '--checkpoint', '{checkpoint}']}}}
        else:
            profile['memory'] = {'block': 'frozen', 'options': {'snapshot': str(snapshot)}}
            profile['learning'] = {'enabled': False}
        path = self.temp / f'{name}.json'
        write_json(path, profile)
        return path

    def run_round(self, name, profile, rule):
        plan = {'profile': str(profile), 'store': str(self.temp / f'{name}_releases'),
                'service_dir': str(self.service_dir), 'evaluation_rule': rule,
                'cases': {'collect': cases(10, 11), 'practice': cases(12), 'evaluate': cases(13, 14)}}
        plan_path = self.temp / f'{name}_plan.json'
        write_json(plan_path, plan)
        out = self.temp / f'{name}_round'
        proc = subprocess.run([support.PYTHON, '-B', '-m', 'rsi_framework', 'round', '--plan', str(plan_path), '--out', str(out)],
                              env=self.env, capture_output=True, text=True, timeout=900)
        self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-3000:])
        return json.loads((out / 'round_report.json').read_text()), ReleaseStore(plan['store']), out

    def test_full_round_releases_on_a_passed_gate_and_rolls_back(self):
        report, store, out = self.run_round('pass', self.profile('pass'),
                                            {'min_cases': 2, 'min_success_delta': 0, 'min_score_delta': -1.0, 'no_lost_successes': True})
        self.assertEqual(report['evidence'], ['software_synthetic'])
        self.assertTrue(all(e['valid'] for e in report['D1']))
        self.assertGreater(report['memory']['candidates'], 0)
        self.assertTrue(report['memory']['promoted'])
        self.assertEqual(report['memory']['version'], 'm0001')
        self.assertTrue(report['vla']['lineage']['parameters_changed'])
        self.assertGreater(report['vla']['dataset']['samples'], 0)
        self.assertEqual(set(report['vla']['dataset']['by_owner']), {'vla', 'primitive'})
        self.assertTrue(report['memory_revalidation']['passed'])
        self.assertTrue(report['evaluation']['passed'])
        self.assertTrue(report['decision']['released'])
        self.assertEqual(report['active_after'], report['decision']['release'])
        d2_config = json.loads(next((out / 'episodes/D2').glob('*/run_config.json')).read_text())
        self.assertEqual(d2_config['blocks']['vla']['service_root'], report['vla']['candidate_service'])
        self.assertEqual(d2_config['blocks']['memory']['version'], 'm0001')
        self.assertEqual(len(store.state()['history']), 2)
        store.rollback('check rollback')
        self.assertEqual(store.state()['active'], report['P0'])
        events = [e['event'] for e in read_jsonl(Path(store.root) / 'events.jsonl')]
        self.assertEqual(events[-1], 'rolled_back')

    def test_failed_gate_keeps_the_old_release(self):
        report, store, _ = self.run_round('fail', self.profile('fail'),
                                          {'min_cases': 2, 'min_success_delta': 0, 'min_score_delta': 1.0, 'no_lost_successes': True})
        self.assertFalse(report['evaluation']['passed'])
        self.assertFalse(report['decision']['released'])
        self.assertEqual(report['active_after'], report['P0'])
        self.assertEqual(store.state()['history'], [report['P0']])
        self.assertIn('rejected', [e['event'] for e in read_jsonl(Path(store.root) / 'events.jsonl')])

    def test_frozen_round_trains_nothing(self):
        report, store, out = self.run_round('frozen', self.profile('frozen', learn=False), None)
        self.assertEqual(report['vla'], {'enabled': False, 'trainer': 'none', 'training_calls': 0})
        self.assertNotIn('memory', report)
        self.assertEqual(report['decision'], {'released': False, 'reason': 'no candidate change'})
        self.assertFalse((out / 'training').exists())
        self.assertFalse((out / 'dataset').exists())
        self.assertEqual(len(report['D2']), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
