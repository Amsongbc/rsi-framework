"""Per-plugin defaults: configs/envs/<env>.json, configs/vlas/<vla>.json, configs/controllers/<name>.json.

An environment file holds the default task / eval_seed / layout, the block options
and the arguments the original entries need (`original`). A plugin without a file
may define a class attribute `defaults` with the same keys. ${REPO_ROOT},
${ORIGINAL_ROOT} and ${PROJECTS_ROOT} are filled automatically; other ${VARS} come from the environment
and are only required when that plugin is selected.
"""
import json
import os

from rsi_framework import ORIGINAL_ROOT, PROJECTS_DIR, REPO_ROOT
from rsi_framework.core.config import render
from rsi_framework.core.registry import REGISTRY

FOLDERS = {'environment': 'envs', 'vla': 'vlas', 'controller': 'controllers'}


def load(kind, name):
    path = REPO_ROOT / 'configs' / FOLDERS[kind] / f'{name}.json'
    if path.is_file():
        data = json.loads(path.read_text())
    elif kind in ('environment', 'vla'):
        try:
            data = dict(getattr(REGISTRY.entry(kind, name).factory, 'defaults', {}) or {})
        except KeyError:
            data = {}
    else:
        data = {}
    data.pop('description', None)
    environ = dict(os.environ)
    environ.setdefault('REPO_ROOT', str(REPO_ROOT))
    environ.setdefault('ORIGINAL_ROOT', str(ORIGINAL_ROOT))
    environ.setdefault('PROJECTS_ROOT', str(PROJECTS_DIR))
    return render(data, environ)
