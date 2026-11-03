"""What an original run needs before it starts, done automatically.

- The private config: if configs/<cohort>.local.json is missing, it is rendered from
  configs/<cohort>.example.json exactly like tools/render_config.py (same ${VAR} rule,
  new file, mode 0600). PROJECTS_ROOT defaults to projects/ of this repository.
- The EWAM service: the root the original entry will use (window: the config's lane
  service_root; comparison: --service) is checked with a health call. If nothing
  answers, the unchanged projects/ewam_comparison/ewam_service.py is started there and
  stopped after the run. A service that was already running is left alone.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from rsi_framework import PROJECTS_DIR, REPO_ROOT
from rsi_framework.compat import paths
from rsi_framework.core.config import render


def render_config(template, target):
    environ = dict(os.environ)
    environ.setdefault('PROJECTS_ROOT', str(PROJECTS_DIR))
    data = render(json.loads(Path(template).read_text()), environ)
    fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as handle:
        handle.write(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    print(f'rendered {target} from {template}', flush=True)


def service_root(profile):
    compat = profile['compat']
    args = compat['args']
    if compat['cohort'] == 'ewam_window_parallel24':
        config = json.loads(Path(args['config']).read_text())
        lane = config.get('lanes', {}).get(args['lane'], {})
        return lane.get('service_root', config.get('service_root'))
    if compat['cohort'] == 'ewam_comparison':
        return args.get('service', str(paths.original_projects() / 'ewam_comparison/service'))
    return None     # API cohorts: service roots are per study in their configs; started by their own orchestration


def alive(root):
    root = Path(root)
    if not (root / 'ewam.sock').exists():
        return False
    paths.ensure_importable('ewam_comparison')
    from ewam_service import EwamClient
    try:
        return EwamClient(root).call('health').get('status') == 'ready'
    except Exception:
        return False


def start_ewam(root, spec, python=None):
    """Start the EWAM service at `root` unless one answers there; returns the process or None."""
    if alive(root):
        print(f'EWAM service already running at {root}', flush=True)
        return None
    python = python or os.environ.get('EWAM_PYTHON') or sys.executable
    root = Path(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    values = dict(python=python, root=str(root), gpu=spec.get('gpu', 0), repo=str(REPO_ROOT),
                  projects=str(paths.original_projects()))
    argv = [str(part).format(**values) for part in spec['command']]
    log = root.parent / f'{root.name}.service.log'
    print(f'starting EWAM service at {root} (log: {log})', flush=True)
    process = subprocess.Popen(argv, stdout=log.open('w'), stderr=subprocess.STDOUT)
    started, last = time.monotonic(), time.monotonic()
    while not alive(root):
        if process.poll() is not None:
            raise RuntimeError(f'EWAM service exited with code {process.returncode}; see {log}')
        if time.monotonic() - started > spec.get('start_timeout_s', 1800):
            process.terminate()
            raise RuntimeError(f'EWAM service not ready after {spec.get("start_timeout_s", 1800)} s; see {log}')
        if time.monotonic() - last > 30:
            print(f'  waiting for EWAM ({int(time.monotonic() - started)} s)', flush=True)
            last = time.monotonic()
        time.sleep(0.5)
    print('EWAM service ready', flush=True)
    return process


def prepare(profile, start=True):
    """Render the config if needed and start EWAM if needed; returns processes to stop afterwards."""
    services = profile.get('services') or {}
    if services.get('render'):
        render_config(services['render']['template'], services['render']['target'])
    processes = []
    root = service_root(profile)
    if start and root and services.get('ewam'):
        process = start_ewam(root, services['ewam'])
        if process is not None:
            processes.append(process)
    return processes


def stop(processes):
    for process in processes:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
