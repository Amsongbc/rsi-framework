"""Original window_session.py vs the block-built compat template on identical inputs.

Each case runs the unchanged original script and `python -m rsi_framework.compat.window_v1`
against the same offline workspace (upstream test double), the same EWAM service
process, the same config and the same scripted controller commands, then compares
every output file with rsi_framework.evaluation.equivalence.compare.

This is deterministic software evidence against a test double; it does not replace
a RoboDojo end-to-end comparison.
"""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from rsi_framework.compat import baseline
from rsi_framework.evaluation import equivalence
from tests import support

TASK, SEED, LAYOUT = 'classify_objects', 0, 10

FULL = [
    ('python', "print(robo.state()['left_arm_joint_state'])"),
    ('python', "robo.move_to('left', [0.1] * 6)"),
    ('python', "p = robo.ewam_propose()\nprint(p['native_horizon'], p['robot_only_fk']['fk_position_check_passed'])"),
    ('python', "robo.ewam_execute(p['prediction_id'], 'reach', 'arms visible', 4)"),
    ('ack',),
    ('python', "r = robo.ewam_execute(p['prediction_id'], 'reach toward goal', 'arms visible', 4)\nprint(r['native_actions'], r['discarded_native_actions'])"),
    ('python', "robo.move_to('left', [0.2] * 6)"),
    ('ack', {'observation_index': 0}),
    ('ack',),
    ('python', "print(robo.locate('red block'))\nrobo.diagnose('reach', 'arms moved toward the goal', 'continue_policy', 'arm nearer', 'unverified')\nprint(robo.doctor_memory()['phase'])"),
    ('python', "robo.move_to('left', [0.05, 0.1, 0.0, 0.0, 0.1, 0.0], steps=6)"),
    ('python', "p2 = robo.ewam_propose()"),
    ('ack',),
    ('python', "robo.ewam_execute(p2['prediction_id'], 'continue', 'arms visible', 16)"),
    ('python', "open('x')"),
    ('raw', {'op': 'python', 'code': 'print(1)', 'note': ''}),
    ('raw', {'op': 'teleport', 'note': 'unknown op'}),
    ('raw', {'op': 'python', 'code': 'print(2)', 'id': 'wrong'}),
    ('observe',),
    ('finish',),
]

SPARSE = [
    ('python', "robo.ewam_propose()"),
    ('python', "e = robo.ewam_step('reach', 'arms visible', model_actions=3)\nprint(e['native_actions'], e['review_mode'])"),
    ('python', "robo.ewam_step('again', 'arms visible')"),
    ('ack',),
    ('python', "robo.set_gripper('left', 0.2)"),
    ('python', "robo.ewam_step('continue', 'gripper closed', instruction='close in on the block', model_actions=16)"),
    ('ack',),
    ('python', "robo.home()"),
    ('finish',),
]

BUDGET = [('python', "robo.ewam_step('go', 'visible', model_actions=16)"), ('ack',)] * 5


class WindowEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = Path(tempfile.mkdtemp(prefix='rsif-eq-')).resolve()
        cls.projects = support.make_workspace(cls.temp)
        report = baseline.check(projects=cls.projects)
        assert report['passed'], report
        cls.service_root = support.short_dir('rsif-')
        cls.ewam = support.FakeEwam(cls.service_root / 'svc')
        cls.config = support.window_config(cls.projects, cls.service_root / 'svc', cls.temp / 'window.json')
        cls.env = support.environment(cls.projects)

    @classmethod
    def tearDownClass(cls):
        cls.ewam.stop()
        shutil.rmtree(cls.service_root, ignore_errors=True)
        shutil.rmtree(cls.temp, ignore_errors=True)

    def run_pair(self, name, variant, script, config=None, readable=None):
        config = config or self.config
        outs = {}
        for label in ('original', 'framework'):
            out = self.temp / 'runs' / name / label / 'case'
            out.parent.mkdir(parents=True)
            args = ['--config', str(config), '--task', TASK, '--eval-seed', str(SEED), '--layout', str(LAYOUT),
                    '--out', str(out), '--variant', variant, '--lane', 'A']
            if label == 'original':
                argv = [support.PYTHON, '-B', str(self.projects / 'ewam_window_parallel24/window_session.py'), *args]
            elif readable:
                # Task, seed, layout and lane come from configs/envs/robodojo.json (same values as above).
                argv = [support.PYTHON, '-B', '-m', 'rsi_framework', 'run', '--env', 'robodojo', '--vla', 'ewam',
                        '--protocol', f'window_{variant}', '--implementation', readable, '--config', str(config),
                        '--out', str(out), '--allow-stand-ins']
            else:
                argv = [support.PYTHON, '-B', '-m', 'rsi_framework.compat.window_v1', *args, '--allow-stand-ins']
            support.run_session(argv, self.env, out, script, out.parent / 'session.log')
            outs[label] = out
        report = equivalence.compare(outs['original'], outs['framework'])
        details = {key: equivalence.explain(outs['original'], outs['framework'], key) for key in report['differences'][:2]}
        self.assertTrue(report['equal'], json.dumps({'report': report, 'first_differences': details}, default=str)[:6000])
        return outs, report

    def summary(self, out):
        return json.loads((out / 'episode_summary.json').read_text())

    def response(self, out, index):
        return json.loads((out / 'responses' / f'c{index:03d}.json').read_text())

    def cell_error(self, out, index):
        return (self.response(out, index)['output'] or {}).get('error') or ''

    def test_full_protocol_script(self):
        outs, report = self.run_pair('full', 'full', FULL)
        out = outs['original']
        summary = self.summary(out)
        self.assertTrue(summary['valid'])
        self.assertEqual(summary['stop_reason'], 'planner_finish')
        self.assertEqual(summary['native_actions_by_owner'], {'primitive': 6, 'ewam': 60})
        self.assertGreater(report['files_compared'], 100)
        # The script really exercised each protocol branch (same in both runs by equality).
        self.assertIn('EWAM segment before primitive motion', self.cell_error(out, 1))
        self.assertIn('48 True', self.response(out, 2)['output']['stdout'])
        self.assertIn('First show RGB', self.cell_error(out, 3))
        self.assertEqual(self.response(out, 5)['output']['stdout'], '12 36\n')
        self.assertIn('Show fresh camera RGB and end the cell', self.cell_error(out, 6))
        self.assertIn('exact latest observation_index', self.response(out, 7)['error'])
        self.assertIn('Policy audit', self.response(out, 14)['error'])
        self.assertIn('Reviewable command note is required', self.response(out, 15)['error'])
        self.assertIn('Unknown operation: teleport', self.response(out, 16)['error'])
        self.assertIn('Command ID must equal filename stem', self.response(out, 17)['error'])

    def test_readable_flags_run_the_original_and_the_block_built_version(self):
        outs, report = self.run_pair('readable_original', 'sparse', SPARSE, readable='original')
        self.assertEqual(report['framework_extra'], 0, 'implementation original must be the original script itself')
        outs, report = self.run_pair('readable_modular', 'sparse', SPARSE, readable='modular')
        self.assertGreater(report['framework_extra'], 0)
        self.assertTrue(self.summary(outs['framework'])['valid'])

    def test_one_command_starts_ewam_runs_the_original_and_stops_it(self):
        from rsi_framework.compat import services
        root = support.short_dir('rsif-auto-') / 'svc'
        try:
            config = json.loads(self.config.read_text())
            config['lanes']['A']['service_root'] = str(root)
            cfg = self.temp / 'auto.json'
            cfg.write_text(json.dumps(config))
            profile = self.temp / 'auto_profile.json'
            profile.write_text(json.dumps({'env': 'robodojo', 'vla': 'ewam', 'vla_service': {'command': [
                support.PYTHON, '-B', str(support.REPO / 'tests/doubles/fake_ewam_service.py'), '--root', '{root}']}}))
            out = self.temp / 'runs/auto/case'
            out.parent.mkdir(parents=True)
            argv = [support.PYTHON, '-B', '-m', 'rsi_framework', 'run', '--profile', str(profile), '--config', str(cfg),
                    '--out', str(out)]
            code = support.run_session(argv, self.env, out, [('observe',), ('finish',)], out.parent / 'session.log')
            log = (out.parent / 'session.log').read_text()
            self.assertEqual(code, 0, log[-2000:])
            self.assertIn('starting EWAM service', log)
            self.assertTrue(self.summary(out)['valid'])
            self.assertFalse((out / 'implementation_snapshot/rsi_framework').exists(), 'must be the original script')
            self.assertTrue((root.parent / 'svc.service.log').exists())
            self.assertFalse(services.alive(root), 'a service started for the run is stopped afterwards')
        finally:
            shutil.rmtree(root.parent, ignore_errors=True)

    def test_comparison_detects_a_changed_action(self):
        outs, _ = self.run_pair('perturb', 'sparse', SPARSE[:3] + [('finish',)])
        self.assertTrue(equivalence.compare(outs['original'], outs['framework'])['equal'])
        changed = outs['framework']
        rows = [json.loads(line) for line in (changed / 'native_actions.jsonl').read_text().splitlines()]
        rows[2]['action']['left_arm_joint_state'][0] += 1e-9
        (changed / 'native_actions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        report = equivalence.compare(outs['original'], changed)
        self.assertEqual(report['differences'], ['native_actions.jsonl'])
        self.assertFalse(report['equal'])

    def test_sparse_protocol_script(self):
        outs, _ = self.run_pair('sparse', 'sparse', SPARSE)
        summary = self.summary(outs['original'])
        self.assertTrue(summary['valid'])
        self.assertEqual(summary['ewam_attempts'], 2)

    def test_native_budget_exhaustion(self):
        outs, _ = self.run_pair('budget', 'sparse', BUDGET)
        summary = self.summary(outs['original'])
        self.assertEqual(summary['native_actions_issued'], 150)
        self.assertEqual(summary['native_actions_by_owner'], {'primitive': 0, 'ewam': 150})

    def test_boot_failure_without_service(self):
        dead = support.short_dir('rsif-dead-')
        try:
            (dead / 'model.json').write_text((self.service_root / 'svc/model.json').read_text())
            config = support.window_config(self.projects, dead, self.temp / 'dead.json')
            outs, _ = self.run_pair('dead', 'full', [], config=config)
            summary = self.summary(outs['original'])
            self.assertEqual(summary['stop_reason'], 'infrastructure_error')
            self.assertEqual(summary['outcome_class'], 'technical_interruption')
        finally:
            shutil.rmtree(dead, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
