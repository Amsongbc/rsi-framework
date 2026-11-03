"""Shared test harness: an offline workspace, the fake EWAM service and a scripted
window controller that talks to a running session through its file queue.

The workspace is a byte-identical copy of projects/ plus the upstream test double
(PyRUA-Lean, RoboDojo EWAM placeholder) and an empty runtime asset directory, so the
original scripts and the framework import the same files from the same place.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
DOUBLE = REPO / 'tests/doubles/upstream'
ORIGINAL = REPO / 'pyrua-ewam-agent'
PYTHON = sys.executable


def make_workspace(base):
    projects = Path(base) / 'ws/projects'
    shutil.copytree(ORIGINAL / 'projects', projects, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(DOUBLE / 'PyRUA-Lean', projects / 'PyRUA-Lean', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(DOUBLE / 'RoboDojo', projects / 'RoboDojo', ignore=shutil.ignore_patterns('__pycache__'))
    (projects / 'ewam_comparison/runtime_assets/Assets').mkdir(parents=True)
    return projects


def short_dir(prefix):
    """AF_UNIX socket paths are limited to 104 bytes on macOS, so service roots stay short."""
    return Path(tempfile.mkdtemp(prefix=prefix, dir='/tmp' if os.path.isdir('/tmp') else None)).resolve()


class FakeEwam:
    def __init__(self, root, checkpoint=None):
        self.root = Path(root)
        argv = [PYTHON, '-B', str(REPO / 'tests/doubles/fake_ewam_service.py'), '--root', str(self.root)]
        if checkpoint:
            argv += ['--checkpoint', str(checkpoint)]
        self.process = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 30
        while not (self.root / 'ewam.sock').exists():
            if self.process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('fake EWAM service did not start')
            time.sleep(0.05)

    def stop(self):
        self.process.terminate()
        self.process.wait(timeout=10)


def environment(projects, extra=None):
    env = dict(os.environ)
    for key in list(env):
        if key.lower().endswith('_proxy'):
            env.pop(key)
    env.update(PROJECTS_ROOT=str(projects), PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE='1')
    env.pop('RSI_ORIGINAL_PROJECTS', None)
    env.update(extra or {})
    return env


def window_config(projects, service_root, path):
    """The original example template rendered for the offline workspace; all lanes use one service."""
    text = (ORIGINAL / 'configs/ewam_window_parallel24.example.json').read_text()
    config = json.loads(text.replace('${PROJECTS_ROOT}', str(projects)))
    for lane in config['lanes'].values():
        lane['service_root'] = str(service_root)
    Path(path).write_text(json.dumps(config, indent=2) + '\n')
    return Path(path)


class Controller:
    """Plays the window AI: one command at a time, waiting for each response."""

    def __init__(self, out, process, timeout=120):
        self.out, self.process, self.timeout, self.n = Path(out), process, timeout, 0

    def wait_ready(self):
        deadline = time.monotonic() + self.timeout
        while not (self.out / 'latest.json').exists():
            if self.process.poll() is not None or time.monotonic() > deadline:
                return False
            time.sleep(0.05)
        return True

    def latest(self):
        return json.loads((self.out / 'latest.json').read_text())

    def send(self, command):
        identifier = f'c{self.n:03d}'
        self.n += 1
        command = dict(command)
        command.setdefault('id', identifier)
        command.setdefault('note', 'scripted controller step')
        inbox = self.out / 'commands'
        tmp = inbox / (identifier + '.tmp')
        tmp.write_text(json.dumps(command) + '\n')
        os.replace(tmp, inbox / (identifier + '.json'))
        response = self.out / 'responses' / (identifier + '.json')
        deadline = time.monotonic() + self.timeout
        while not response.exists():
            if self.process.poll() is not None or time.monotonic() > deadline:
                return None
            time.sleep(0.02)
        return json.loads(response.read_text())

    def ack(self, **override):
        latest = self.latest()
        command = dict(op='ack_visual', observation_index=latest['observation_index'],
                       native_action_index=latest['native_action_index'],
                       image_sha256={'cam_head': latest['image_sha256']['cam_head']},
                       viewed_by='window_AI', evidence='Synthetic frame: both arms and the table are visible')
        command.update(override)
        return self.send(command)

    def play(self, script):
        if not self.wait_ready():
            return
        for step in script:
            kind = step[0]
            if kind == 'python':
                result = self.send(dict(op='python', code=step[1]))
            elif kind == 'ack':
                result = self.ack(**(step[1] if len(step) > 1 else {}))
            elif kind == 'raw':
                result = self.send(step[1])
            else:
                result = self.send(dict(op=kind))
            if result is None:
                return


def run_session(argv, env, out, script, log):
    with open(log, 'w') as handle:
        process = subprocess.Popen(argv, env=env, stdout=handle, stderr=subprocess.STDOUT)
        try:
            Controller(out, process).play(script)
            process.wait(timeout=180)
        finally:
            if process.poll() is None:
                process.kill()
    return process.returncode
