#!/usr/bin/env python3
"""Compile sources in memory and scan export bytes without revealing secrets."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--key-file',type=Path);a=p.parse_args()
    # Optional real key is used for equality comparison only, never emitted.
    key=a.key_file.read_bytes().strip() if a.key_file else None
    files=[p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts and '__pycache__' not in p.parts]
    issues=[];compiled=0
    for path in files:
        raw=path.read_bytes();relative=str(path.relative_to(ROOT))
        if key and key in raw:issues.append(dict(path=relative,kind='actual_key_match'))
        for label,pattern in [('credential_url',rb'https?://[^\s/\x22\x27:]+:[^\s/\x22\x27@]+@'),
                              ('private_key',rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
                              ('likely_live_openai_key',rb'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{40,}')]:
            if re.search(pattern,raw):issues.append(dict(path=relative,kind=label))
        if path.suffix=='.py':compile(raw,str(path),'exec');compiled+=1
    manifest=json.loads((ROOT/'provenance/source_manifest.json').read_text())
    for row in manifest:
        for field,sha in [('original_object','original_sha256'),('export','export_sha256')]:
            if hashlib.sha256((ROOT/row[field]).read_bytes()).hexdigest()!=row[sha]:issues.append(dict(path=row[field],kind='manifest_hash_mismatch'))
    report=dict(files_scanned=len(files),python_files_compiled=compiled,total_bytes=sum(p.stat().st_size for p in files),
        actual_key_comparison=bool(key),issues=issues,passed=not issues,
        limits='Static patterns and optional known-key equality are evidence, not proof against all possible unknown secrets. No runs/logs/images/API records were included by allowlist.')
    (ROOT/'SECURITY_SCAN.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return 0 if not issues else 1

if __name__=='__main__':raise SystemExit(main())
