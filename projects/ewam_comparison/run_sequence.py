#!/usr/bin/env python3
"""Run a fixed sequence; hybrid episodes await commands from control.py."""

# Export portability change: resolve external projects via PROJECTS_ROOT.
import os as _export_os
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--wait-for', type=Path)
p.add_argument('--cases', nargs='+', required=True, help='mode:task entries')
a = p.parse_args()
root = Path(__file__).resolve().parent
if a.wait_for:
    while not a.wait_for.exists():
        time.sleep(2)
env = dict(os.environ)
for key in list(env):
    if key.lower().endswith('_proxy') or key.startswith('CONDA'):
        env.pop(key)
env.update(NO_PROXY='*', no_proxy='*', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
for case in a.cases:
    mode, task = case.split(':')
    tag = f'{mode}_{task}_s0_l0'
    out = root / 'runs' / tag
    if out.exists():
        raise RuntimeError(f'Refusing to overwrite or select existing run: {out}')
    print(f'Starting {tag}', flush=True)
    cmd = [str(root/'.venv-ewam/bin/python'), '-u', str(root/'session.py'),
           '--mode', mode, '--task', task, '--eval-seed', '0', '--layout', '0',
           '--sim-gpu', '2', '--perception-gpu', '1', '--out', str(out)]
    with (root/(tag+'.log')).open('w') as log:
        result = subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f'{tag} exit={result.returncode}', flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    row = json.loads((out/'result.json').read_text())
    print(json.dumps({key: row[key] for key in ['task','mode','valid','success','client_score','native_actions_issued']}, ensure_ascii=False), flush=True)
