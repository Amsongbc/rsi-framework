"""Compare two run directories produced from identical inputs (original vs framework).

Allowed differences are fixed here, not tuned after a failure:
- wall-clock / latency / pid fields (by key name) are masked, their presence kept;
- the run directory prefix is replaced by <OUT>;
- uuid4 hex identifiers are renamed by order of first appearance (one-to-one);
- traceback frames inside the runner (window_session Session/RecordedEwam methods
  vs rsi_framework files) collapse to one <RUNNER> marker; frames in shared original
  code, exception types and messages stay as they are;
- the framework run may add implementation_snapshot/rsi_framework/** and the matching
  trailing rows of source_manifest.json;
- .npz archives are compared by array content (zip headers carry wall-clock time).
Everything else, including PNG and MP4 bytes, must be equal.
"""
import json
from pathlib import Path
import re

import numpy as np

VOLATILE = frozenset({'time', 'issued_at', 'updated_at', 'wall_s', 'elapsed_s', 'inference_s', 'startup_s',
                      'control_wall_s', 'total_wall_s', 'latency_s', 'pid', 'created'})
RUNNER_FUNCS = frozenset({'loop', 'python', 'run', 'boot', 'close', 'snapshot', 'call', 'main', 'layout',
                          'source_snapshot'})
FRAME = re.compile(r'^  File "(?P<path>[^"]+)", line (?P<line>\d+), in (?P<func>.+)$')
UUID = re.compile(r'\b[0-9a-f]{32}\b')
FRAMEWORK_EXTRA = 'implementation_snapshot/rsi_framework/'


def is_runner_frame(path, func):
    if '/rsi_framework/' in path:
        return True
    name = Path(path).name
    return name in ('window_session.py', 'session.py') and func in RUNNER_FUNCS


def normalize_traceback(text):
    if 'Traceback (most recent call last):' not in text:
        return text
    lines, out, skip_source, in_runner = text.split('\n'), [], False, False
    for line in lines:
        match = FRAME.match(line)
        if match:
            skip_source = False
            if is_runner_frame(match['path'], match['func']):
                if not in_runner:
                    out.append('  <RUNNER>')
                in_runner, skip_source = True, True
                continue
            in_runner = False
            out.append(line)
            continue
        if skip_source and line.startswith('    '):
            skip_source = False
            continue
        if line.strip() and set(line.strip()) <= set('^~'):
            continue    # caret markers depend on the source line shown above
        skip_source = False
        out.append(line)
    return '\n'.join(out)


class Normalizer:
    def __init__(self, out_dir):
        self.out = str(Path(out_dir).resolve())
        self.uuids = {}

    def text(self, value):
        value = value.replace(self.out, '<OUT>')
        value = normalize_traceback(value)
        return UUID.sub(lambda m: self.uuids.setdefault(m.group(0), f'<uuid{len(self.uuids) + 1}>'), value)

    def value(self, value, key=None):
        if key in VOLATILE and value is not None:
            return '<volatile>'
        if isinstance(value, dict):
            return {k: self.value(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [self.value(v) for v in value]
        if isinstance(value, str):
            return self.text(value)
        return value


def _files(root):
    root = Path(root)
    return {str(p.relative_to(root)): p for p in sorted(root.rglob('*')) if p.is_file() or p.is_symlink()}


def _load(path, normalizer):
    if path.is_symlink():
        return ('symlink', normalizer.text(str(path.readlink())))
    if path.suffix == '.json' or path.suffix == '.accepted':
        return ('json', normalizer.value(json.loads(path.read_text())))
    if path.suffix == '.jsonl':
        return ('jsonl', [normalizer.value(json.loads(line)) for line in path.read_text().splitlines()])
    if path.suffix == '.npz':
        with np.load(path) as data:
            return ('npz', {k: data[k].tolist() for k in data.files})
    if path.suffix in ('.py', '.sh', '.md', '.txt') and not str(path).startswith('implementation_snapshot'):
        return ('text', normalizer.text(path.read_text()))
    return ('bytes', path.read_bytes())


def compare(reference_dir, candidate_dir):
    """Return a report; `equal` is True only if every file matches under the fixed rules."""
    ref, cand = _files(reference_dir), _files(candidate_dir)
    extra = sorted(k for k in cand if k not in ref)
    allowed_extra = [k for k in extra if k.startswith(FRAMEWORK_EXTRA)]
    unexpected_extra = [k for k in extra if not k.startswith(FRAMEWORK_EXTRA)]
    missing = sorted(k for k in ref if k not in cand)
    rn, cn = Normalizer(reference_dir), Normalizer(candidate_dir)
    differences = []
    for key in sorted(k for k in ref if k in cand):
        a, b = _load(ref[key], rn), _load(cand[key], cn)
        if key == 'source_manifest.json':
            rows = b[1][:len(a[1])]
            framework_rows = b[1][len(a[1]):]
            if rows != a[1] or not all(r['snapshot'].startswith(FRAMEWORK_EXTRA) for r in framework_rows):
                differences.append(key)
            continue
        if a != b:
            differences.append(key)
    return {'reference': str(reference_dir), 'candidate': str(candidate_dir), 'files_compared': len(ref),
            'missing': missing, 'unexpected_extra': unexpected_extra, 'framework_extra': len(allowed_extra),
            'differences': differences, 'equal': not missing and not unexpected_extra and not differences}


def explain(reference_dir, candidate_dir, key):
    """Both normalized forms of one file, for debugging a reported difference."""
    rn, cn = Normalizer(reference_dir), Normalizer(candidate_dir)
    for path in sorted(Path(reference_dir).rglob('*')):
        if path.is_file():
            _load(path, rn)
    for path in sorted(Path(candidate_dir).rglob('*')):
        if path.is_file():
            _load(path, cn)
    return _load(Path(reference_dir) / key, rn), _load(Path(candidate_dir) / key, cn)
