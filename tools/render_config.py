#!/usr/bin/env python3
"""Expand explicit ${VARIABLE} placeholders into a private local JSON config."""
import json
import os
from pathlib import Path
import re
import sys

def render(value):
    if isinstance(value,dict):return {k:render(v) for k,v in value.items()}
    if isinstance(value,list):return [render(v) for v in value]
    if isinstance(value,str):
        def replace(match):
            key=match.group(1)
            if not os.environ.get(key):raise ValueError('Set environment variable '+key)
            return os.environ[key]
        return re.sub(r'\$\{([A-Z_]+)\}',replace,value)
    return value

if __name__=='__main__':
    source,target=map(Path,sys.argv[1:])
    if target.exists():raise FileExistsError('Choose a fresh private destination')
    data=render(json.loads(source.read_text()))
    fd=os.open(target,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as f:f.write(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
