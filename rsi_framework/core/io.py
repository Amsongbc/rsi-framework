"""JSON record helpers with the exact byte format of the original window_session helpers.

Compat runs must write files identical to the original ones, so these keep the same
indentation, ensure_ascii=False and trailing newline (checked in tests against the
original functions).
"""
import dataclasses
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def plain(value):
    if dataclasses.is_dataclass(value):
        return plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(plain(value), indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def append_json(path, value):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(plain(value), ensure_ascii=False) + '\n')


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value):
    """Digest of a JSON value with sorted keys and compact separators."""
    return sha256_bytes(json.dumps(plain(value), sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode())
