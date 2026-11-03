"""S0: the original export must stay byte-identical.

original_manifest.json records every file of the downloaded export with its SHA-256
and the directory fingerprint quoted in the workflow document. Any equivalence
claim is tied to this fingerprint; a changed original file invalidates it. Compat
runs also check the projects/ files of the workspace they actually import.
"""
import json
from pathlib import Path

from rsi_framework import ORIGINAL_ROOT
from rsi_framework.core.io import canonical_sha256, sha256_file

MANIFEST = Path(__file__).with_name('original_manifest.json')


def load_manifest():
    return json.loads(MANIFEST.read_text())


def check(root=ORIGINAL_ROOT, projects=None):
    """Compare the original files with the manifest.

    `root` is the full original export (pyrua-ewam-agent/). `projects`, if given, is a workspace
    projects/ directory whose cohort files are checked instead of root/projects.
    """
    manifest = load_manifest()
    changed, missing = [], []
    for relative, expected in manifest['files'].items():
        if projects is not None:
            if not relative.startswith('projects/'):
                continue
            path = Path(projects) / relative[len('projects/'):]
        else:
            path = Path(root) / relative
        if not path.is_file():
            missing.append(relative)
        elif sha256_file(path) != expected:
            changed.append(relative)
    fingerprint = canonical_sha256(manifest['files'])
    return {'fingerprint': manifest['fingerprint'], 'fingerprint_recomputed': fingerprint,
            'scope': 'projects' if projects is not None else 'export', 'checked_root': str(projects or root),
            'changed': changed, 'missing': missing,
            'passed': not changed and not missing and fingerprint == manifest['fingerprint']}


def require_unchanged(root=ORIGINAL_ROOT, projects=None):
    report = check(root, projects)
    if not report['passed']:
        raise RuntimeError('Original export changed; equivalence evidence no longer applies: '
                           + json.dumps({k: report[k] for k in ('checked_root', 'changed', 'missing')}))
    return report
