"""Profile loading.

A profile is JSON. `${NAME}` placeholders are filled from the environment with the
same rule as tools/render_config.py (unset or empty is an error). `extends` names
another profile relative to this one; dictionaries merge recursively, everything
else is replaced.
"""
import copy
import json
import os
from pathlib import Path
import re

PLACEHOLDER = re.compile(r'\$\{([A-Z_][A-Z0-9_]*)\}')


def render(value, environ=None):
    environ = os.environ if environ is None else environ
    if isinstance(value, dict):
        return {k: render(v, environ) for k, v in value.items()}
    if isinstance(value, list):
        return [render(v, environ) for v in value]
    if isinstance(value, str):
        def replace(match):
            key = match.group(1)
            if not environ.get(key):
                raise ValueError('Set environment variable ' + key)
            return environ[key]
        return PLACEHOLDER.sub(replace, value)
    return value


def merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_raw(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    parent = data.pop('extends', None)
    if parent:
        data = merge(load_raw(path.parent / parent), data)
    return data


def load_profile(path, environ=None, overrides=None):
    data = load_raw(path)
    if overrides:
        data = merge(data, overrides)
    data = render(data, environ)
    data.setdefault('name', Path(path).stem)
    return data
