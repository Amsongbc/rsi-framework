#!/usr/bin/env python3
"""Submit explicit window-AI commands and inspect their recorded responses."""

# Export portability change: resolve external projects via PROJECTS_ROOT.
import os as _export_os
import argparse
import json
import os
from pathlib import Path
import time
import uuid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--note',default='')
    p.add_argument('--async',dest='asynchronous',action='store_true')
    p.add_argument('--timeout-s',type=float,default=1200)
    sub=p.add_subparsers(dest='op',required=True)
    sub.add_parser('observe'); sub.add_parser('finish')
    code=sub.add_parser('python'); code.add_argument('--file',type=Path,required=True)
    ack=sub.add_parser('ack_visual')
    ack.add_argument('--observation-index',type=int,required=True)
    ack.add_argument('--native-action-index',type=int,required=True)
    ack.add_argument('--camera-sha256',action='append',required=True,metavar='CAMERA=SHA256')
    ack.add_argument('--evidence',required=True)
    poll=sub.add_parser('poll'); poll.add_argument('--id',required=True)
    args=p.parse_args()
    if args.op=='poll': identifier=args.id
    else:
        if not args.note.strip(): p.error('--note is required for a reviewable command record')
        identifier=uuid.uuid4().hex
        command=dict(id=identifier,op=args.op,issued_at=time.time(),note=args.note)
        if args.op=='python': command['code']=args.file.read_text()
        if args.op=='ack_visual':
            images={}
            for item in args.camera_sha256:
                camera,sep,digest=item.partition('=')
                if not sep or len(digest)!=64: p.error('Expected CAMERA=64-character-SHA256')
                images[camera]=digest
            command.update(observation_index=args.observation_index,native_action_index=args.native_action_index,
                image_sha256=images,viewed_by='window_AI',evidence=args.evidence)
        inbox=args.session/'commands'
        if not inbox.is_dir(): p.error('Session commands directory does not exist')
        if (args.session/'result.json').exists(): p.error('Session has already ended')
        temporary=inbox/(identifier+'.tmp'); temporary.write_text(json.dumps(command,ensure_ascii=False)+'\n')
        os.replace(temporary,inbox/(identifier+'.json'))
    response=args.session/'responses'/(identifier+'.json')
    if args.asynchronous:
        print(json.dumps(dict(id=identifier,response=str(response),pending=not response.exists()))); return
    deadline=time.monotonic()+args.timeout_s
    while not response.exists():
        if (args.session/'result.json').exists():
            print((args.session/'result.json').read_text()); return
        if time.monotonic()>=deadline:
            print(json.dumps(dict(id=identifier,response=str(response),pending=True))); return
        time.sleep(.25)
    print(response.read_text())


if __name__=='__main__': main()
