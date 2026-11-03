"""Run the original offline tests unchanged, in a temporary copy of projects/.

Several original tests write result files next to themselves, so they never run in
the repository. `upstream` is the directory holding PyRUA-Lean (and RoboDojo);
by default the test double. Tests that need assets this machine lacks are
reported as skipped with the reason, never as passed.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from rsi_framework import PROJECTS_DIR, REPO_ROOT

DOUBLE = REPO_ROOT / 'tests/doubles/upstream'

SUITES = [
    ('ewam_window_parallel24', ['window_protocol_tests.py'], ()),
    ('ewam_comparison', ['-m', 'unittest', 'test_episode_contract'], ()),
    ('ewam_doctor_integration', ['test_doctor.py'], ()),
    ('ewam_doctor_validation72', ['test_doctor.py'], ()),
    ('ewam_doctor_efficiency48', ['test_doctor.py'], ()),
    ('ewam_doctor_efficiency48', ['test_efficiency.py'], ('experiment_config.json',)),
    ('ewam_doctor_efficiency48', ['test_plan.py'], ('experiment_config.json', 'layouts')),
]


def _render_config(cohort, projects, key_file):
    template = REPO_ROOT / 'configs' / f'{cohort}.example.json'
    text = template.read_text().replace('${PROJECTS_ROOT}', str(projects)).replace('${API_KEY_FILE}', str(key_file))
    (projects / cohort / 'experiment_config.json').write_text(text)
    return json.loads(text)


def run(upstream=DOUBLE, python=sys.executable, keep=False):
    base = Path(tempfile.mkdtemp(prefix='rsif-suites-')).resolve()
    projects = base / 'projects'
    shutil.copytree(PROJECTS_DIR, projects, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('PyRUA-Lean', 'RoboDojo'):
        if (Path(upstream) / name).exists():
            shutil.copytree(Path(upstream) / name, projects / name, ignore=shutil.ignore_patterns('__pycache__'))
    key_file = base / 'offline_fake_key'
    key_file.write_text('offline-fake-key')
    results = []
    try:
        for cohort, argv, needs in SUITES:
            cwd = projects / cohort
            row = {'cohort': cohort, 'test': ' '.join(argv), 'upstream': str(upstream)}
            if 'experiment_config.json' in needs:
                config = _render_config(cohort, projects, key_file)
                if 'layouts' in needs:
                    layout_root = projects / 'RoboDojo-resources/Assets/Eval_Layout/RoboDojo/arx_x5'
                    missing = [f'{s}/classify_objects_{l}.json' for l in config['layouts'] for s in config['eval_seeds']
                               if not (layout_root / str(s) / f'classify_objects_{l}.json').is_file()]
                    if missing:
                        results.append(dict(row, status='skipped', reason=f'needs RoboDojo-resources layouts ({len(missing)} missing)'))
                        continue
            env = dict(os.environ, PROJECTS_ROOT=str(projects), PYTHONDONTWRITEBYTECODE='1',
                       PYTHONPATH=os.pathsep.join([str(projects / 'PyRUA-Lean/src'), str(projects / 'ewam_comparison'), str(cwd)]))
            proc = subprocess.run([python, '-B', *argv], cwd=cwd, env=env, capture_output=True, text=True, timeout=600)
            results.append(dict(row, status='passed' if proc.returncode == 0 else 'failed', returncode=proc.returncode,
                                tail=(proc.stdout + proc.stderr)[-1500:]))
    finally:
        if not keep:
            shutil.rmtree(base, ignore_errors=True)
    return {'workspace': str(base) if keep else None, 'results': results,
            'passed': all(r['status'] != 'failed' for r in results),
            'evidence': 'software_synthetic' if Path(upstream).resolve() == DOUBLE.resolve() else 'upstream_as_given'}
