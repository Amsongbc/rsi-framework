"""Core pieces: original files unchanged, record helpers byte-identical to the original
ones, profile loading, compat argv, registry, and the original offline suites."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from rsi_framework import PROJECTS_DIR, REPO_ROOT
from rsi_framework.compat import baseline, launch, original_suites, profiles
from rsi_framework.core import compose, config, io
from rsi_framework.core.registry import REGISTRY, Registry, load_builtin


def original_window_session():
    sys.path.insert(0, str(PROJECTS_DIR / 'ewam_comparison'))
    spec = importlib.util.spec_from_file_location('original_window_session', PROJECTS_DIR / 'ewam_window_parallel24/window_session.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Core(unittest.TestCase):
    def test_original_export_is_unchanged(self):
        report = baseline.check()
        self.assertTrue(report['passed'], report)
        self.assertEqual(report['fingerprint'], 'b5d3deb06050575e0243f96387ab85c7f2994db30cdf5f0ba7ec7120f4a05ac1')

    def test_baseline_detects_a_changed_original_file(self):
        with tempfile.TemporaryDirectory() as name:
            copy = Path(name) / 'projects'
            shutil.copytree(PROJECTS_DIR, copy, ignore=shutil.ignore_patterns('__pycache__'))
            self.assertTrue(baseline.check(projects=copy)['passed'])
            path = copy / 'ewam_window_parallel24/doctor_robot.py'
            path.write_text(path.read_text() + '\n')
            report = baseline.check(projects=copy)
            self.assertEqual(report['changed'], ['projects/ewam_window_parallel24/doctor_robot.py'])

    def test_record_helpers_match_the_original_bytes(self):
        original = original_window_session()
        value = {'a': np.float32(1.5), 'b': [np.arange(3)], 'c': '中文', 'd': {'e': np.int64(2)}}
        with tempfile.TemporaryDirectory() as name:
            a, b = Path(name) / 'a.json', Path(name) / 'b.json'
            original.write_json(a, value)
            io.write_json(b, value)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            original.append_json(a.with_suffix('.jsonl'), value)
            io.append_json(b.with_suffix('.jsonl'), value)
            self.assertEqual(a.with_suffix('.jsonl').read_bytes(), b.with_suffix('.jsonl').read_bytes())

    def test_profile_render_and_extends(self):
        with tempfile.TemporaryDirectory() as name:
            base, child = Path(name) / 'base.json', Path(name) / 'child.json'
            base.write_text(json.dumps({'a': {'x': '${RSI_TEST_X}', 'y': 1}, 'b': [1]}))
            child.write_text(json.dumps({'extends': 'base.json', 'a': {'y': 2}}))
            data = config.load_profile(child, environ={'RSI_TEST_X': 'value'})
            self.assertEqual(data['a'], {'x': 'value', 'y': 2})
            with self.assertRaisesRegex(ValueError, 'Set environment variable RSI_TEST_X'):
                config.load_profile(child, environ={})

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cfg = Path(self.temp.name) / 'window.local.json'
        self.cfg.write_text('{}')
        self.llm = mock.patch.dict(os.environ, {'LLM_BASE_URL': 'http://llm', 'LLM_MODEL': 'm', 'LLM_KEY_FILE': '/k'})
        self.llm.start()

    def tearDown(self):
        self.llm.stop()
        self.temp.cleanup()

    def test_env_and_vla_alone_run_the_original_with_env_defaults(self):
        window = compose.resolve({'env': 'robodojo', 'vla': 'ewam', 'original': {'config': str(self.cfg)}})
        self.assertEqual(window['preset'], 'robodojo_ewam_window_full')
        self.assertEqual(window['case_defaults'], {'task': 'classify_objects', 'eval_seed': 0, 'layout': 10})
        argv = profiles.argv(window, window['case_defaults'], '/runs/new-case')
        self.assertEqual(argv, ['--config', str(self.cfg), '--task', 'classify_objects', '--eval-seed', '0', '--layout', '10',
                                '--out', '/runs/new-case', '--variant', 'full', '--lane', 'A'])
        command = launch.command('ewam_window_parallel24', 'window_session', argv, python='/venv/bin/python')
        self.assertEqual(command[:2], ['/venv/bin/python', str(PROJECTS_DIR / 'ewam_window_parallel24/window_session.py')])
        default = compose.resolve({'env': 'robodojo', 'vla': 'ewam'})   # configs/*.local.json: rendered at launch
        self.assertEqual(default['services']['render']['target'],
                         str(REPO_ROOT / 'configs/ewam_window_parallel24.local.json'))
        base = compose.resolve({'env': 'robodojo', 'vla': 'ewam', 'controller': 'none'})
        self.assertEqual(profiles.argv(base, {'task': 'push_T', 'eval_seed': 0, 'layout': 0}, '/o'),
                         ['--mode', 'baseline', '--task', 'push_T', '--layout', '0', '--eval-seed', '0', '--out', '/o',
                          '--sim-gpu', '2', '--perception-gpu', '1', '--service', str(PROJECTS_DIR / 'ewam_comparison/service')])

    def test_defaults_pick_protocol_decision_and_options(self):
        agent = compose.resolve({'env': 'robodojo', 'vla': 'ewam', 'controller': 'api'})
        self.assertEqual((agent['mode'], agent['protocol']['block'], agent['policy']['options']['ops']),
                         ('agent_vla', 'vla_prefix', ['tool', 'vla', 'observe', 'finish']))
        self.assertEqual(agent['llm_backend']['options']['base_url'], 'http://llm')
        self.assertEqual(agent['vla']['options']['service_root'], str(PROJECTS_DIR / 'ewam_comparison/service'))
        self.assertEqual((agent['environment']['options'], agent['environment']['perception_gpu']), ({'simulator_gpu': 2}, 1))
        cap = compose.resolve({'env': 'robodojo', 'vla': 'none', 'controller': 'api'})
        self.assertEqual((cap['mode'], cap['vla'], cap['protocol']['block'], cap['policy']['options']['ops']),
                         ('llm_policy', None, 'tools_only', ['python', 'observe', 'finish']))
        window_cap = compose.resolve({'env': 'robodojo', 'vla': 'none'})
        self.assertEqual((window_cap['llm_backend']['block'], window_cap['policy']['options']['require_id'],
                          window_cap['protocol']['options']), ('command_queue', True, {'observe_after_motion': False}))
        generic = compose.resolve({'env': 'robodojo', 'vla': 'ewam', 'implementation': 'generic'})
        self.assertEqual((generic['mode'], generic['policy']['options']['ops']),
                         ('agent_vla', ['python', 'observe', 'ack_visual', 'finish']))
        api = compose.resolve({'env': 'robodojo', 'vla': 'ewam', 'controller': 'api_responses', 'protocol': 'doctor',
                               'study': 'efficiency48', 'original': {'config': str(self.cfg)}})
        self.assertEqual(api['compat']['cohort'], 'ewam_doctor_efficiency48')
        for bad, message in [({'env': 'robodojo', 'vla': 'ewam', 'memory': 'frozen', 'original': {'config': str(self.cfg)}},
                              'Memory from the original config'),
                             ({'env': 'robodojo', 'vla': 'ewam', 'protocol': 'doctor'}, 'only as original cohorts'),
                             ({'env': 'robodojo', 'vla': 'none', 'implementation': 'modular'}, 'only for original combinations')]:
            with self.subTest(message):
                with self.assertRaisesRegex(ValueError, message):
                    compose.resolve(bad)

    def test_every_example_profile_resolves(self):
        from rsi_framework.core.assembly import validate
        env = {k: '/x' for k in ('MEMORY_SNAPSHOT', 'MEMORY_STORE', 'TRAINER_PYTHON', 'TRAINER_SCRIPT', 'SERVE_PYTHON', 'SERVE_SCRIPT')}
        for path in sorted((REPO_ROOT / 'configs/profiles').glob('*.json')):
            with self.subTest(path.name):
                profile = config.load_profile(path, environ=env)
                profile['original'] = {'config': str(self.cfg)}
                profile = compose.resolve(profile)
                if profile['mode'] == 'compat':
                    profiles.argv(profile, profile['case_defaults'], '/o')
                else:
                    validate(profile)

    def test_missing_local_config_is_rendered_like_the_original_tool(self):
        from rsi_framework.compat import services
        import subprocess
        template = REPO_ROOT / 'configs/ewam_window_parallel24.example.json'
        ours, theirs = Path(self.temp.name) / 'a.local.json', Path(self.temp.name) / 'b.local.json'
        with mock.patch.dict(os.environ, {'PROJECTS_ROOT': '/work/projects'}):
            services.render_config(template, ours)
            subprocess.run([sys.executable, str(REPO_ROOT / 'tools/render_config.py'), str(template), str(theirs)],
                           check=True, env=dict(os.environ))
            with self.assertRaises(FileExistsError):
                services.render_config(template, ours)
        self.assertEqual(ours.read_bytes(), theirs.read_bytes())
        self.assertEqual(oct(ours.stat().st_mode & 0o777), '0o600')
        missing = str(Path(self.temp.name) / 'window.local.json.missing')
        local = str(Path(self.temp.name) / 'new.local.json')
        resolved = compose.resolve({'env': 'robodojo', 'vla': 'ewam', 'original': {'config': local}})
        self.assertEqual(resolved['services']['render'], {'template': str(template), 'target': local})
        self.assertEqual(resolved['services']['ewam']['gpu'], 0)
        with self.assertRaisesRegex(ValueError, 'does not exist'):
            compose.resolve({'env': 'robodojo', 'vla': 'ewam', 'original': {'config': missing}})

    def test_old_block_names_still_work(self):
        load_builtin()
        self.assertEqual(REGISTRY.canonical('environment', 'pyrua_robodojo'), 'robodojo')
        self.assertEqual(REGISTRY.canonical('vla', 'ewam_service'), 'ewam')
        self.assertEqual(REGISTRY.canonical('robot', 'arx_x5_bimanual'), 'arx_x5')
        self.assertNotIn('pyrua_robodojo', REGISTRY.names('environment'))

    def test_registry(self):
        load_builtin()
        for kind in ('environment', 'vla', 'llm_backend', 'policy', 'tools', 'protocol', 'memory', 'recorder',
                     'dataset', 'trainer', 'evaluator'):
            self.assertTrue(REGISTRY.names(kind), kind)
        registry = Registry()
        registry.register('trainer', 'x')(lambda: 1)
        with self.assertRaises(ValueError):
            registry.register('trainer', 'x')(lambda: 2)
        with self.assertRaises(KeyError):
            registry.create('trainer', 'missing')

    def test_original_offline_suites_pass_against_the_double(self):
        report = original_suites.run()
        failed = [r for r in report['results'] if r['status'] == 'failed']
        self.assertFalse(failed, json.dumps(failed, indent=1)[:3000])
        self.assertEqual(sum(r['status'] == 'passed' for r in report['results']), 6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
