#!/usr/bin/env python3
"""Run24 counterbalanced waves,12 four-arm case blocks,48 episodes."""

# Export portability change: resolve external projects via PROJECTS_ROOT.
import os as _export_os
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from study_plan import ROOT,build,digest

RUNTIME=ROOT.parent/'ewam_comparison'
PYTHON=RUNTIME/'.venv-ewam/bin/python'
sys.path.insert(0,str(RUNTIME))
from ewam_service import EwamClient


def save(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temporary.replace(path)


def completed():
    count=0
    for p in (ROOT/'runs').glob('*/episode_summary.json'):
        try:count+=bool(json.loads(p.read_text()).get('valid',False))
        except json.JSONDecodeError:pass
    return count


def check_frozen(config,declared):
    if digest(ROOT/'experiment_config.json')!=declared['config_sha256']:raise RuntimeError('Frozen configuration changed')
    for name,value in declared['control_files'].items():
        if digest(ROOT/name)!=value:raise RuntimeError('Frozen controller changed: '+name)
    for task,values in declared['frozen_experience'].items():
        for name,key in [('frozen_experience.json','experience_sha256'),('gate.json','gate_sha256')]:
            if digest(ROOT/'harness'/task/name)!=values[key]:raise RuntimeError('Frozen experience/gate changed')


def ready(config):
    for _ in range(600):
        all_ready=True
        for lane in config['lanes'].values():
            try: all_ready=EwamClient(lane['service_root']).call('health').get('status')=='ready' and all_ready
            except (OSError,EOFError,ConnectionError):all_ready=False
        if all_ready:return
        time.sleep(2)
    raise RuntimeError('Independent EWAM services did not become ready within1200 seconds')


def admit(config):
    requirements={}
    for lane in config['lanes'].values():
        for role,mb in [('simulator',16*1024),('sam3',10*1024)]:
            gpu=lane['gpus'][role];requirements[gpu]=requirements.get(gpu,0)+mb
    while True:
        enough=True
        for gpu,needed in requirements.items():
            p=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            if int(p.stdout.strip())<needed:enough=False
        if enough:return
        save(ROOT/'study_status.json',{'status':'running','stage':'waiting_gpu_memory','completed_episodes':completed(),
            'planned_episodes':48,'required_free_mib':requirements,'updated_at':time.time()})
        time.sleep(20)


def prepare(config,case,retry_invalid):
    tag=f"{case['task']}_{case['arm']}_s{case['seed']}_l{case['layout']}"
    attempts=sorted((ROOT/'runs').glob(tag+'_attempt*'))
    valid=[]
    for out in attempts:
        path=out/'episode_summary.json'
        if not path.exists():raise RuntimeError('Unfinished attempt requires inspection: '+str(out))
        summary=json.loads(path.read_text())
        if summary['valid']:valid.append((out,summary))
    if len(valid)>1:raise RuntimeError('Duplicate valid results; outcome selection prohibited')
    if valid:return {'case':case,'out':valid[0][0],'done':True,'summary':valid[0][1],'skipped_existing_valid':True}
    if attempts and not retry_invalid:raise RuntimeError('Invalid attempt retained; inspect before --retry-invalid: '+tag)
    number=1+max([int(p.name.rsplit('_attempt',1)[1]) for p in attempts] or [0])
    out=ROOT/'runs'/f'{tag}_attempt{number}'
    cfg=json.loads(json.dumps(config));lane=config['lanes'][case['lane']]
    cfg.update(gpus=lane['gpus'],service_root=lane['service_root'],study_case=case)
    gate=json.loads((ROOT/'harness'/case['task']/'gate.json').read_text())
    enabled=True
    cfg['efficiency'].update(config['arms'][case['arm']])
    memory=json.loads((ROOT/'harness'/case['task']/'frozen_experience.json').read_text()) if enabled and gate['branch']=='hybrid' else None
    cfg['doctor'].update(enabled=enabled,phase=gate['branch'] if enabled else None,frozen_memory=memory)
    cfg['execution_strategy']='factorial_'+case['arm']+'_frozen_experience'
    path=ROOT/'case_configs'/f'{out.name}.json';save(path,cfg)
    return {'case':case,'out':out,'config_path':path,'done':False}


def launch(item):
    case=item['case'];out=item['out'];environment=dict(os.environ)
    for key in list(environment):
        if key.startswith('CONDA'):environment.pop(key)
    log=(ROOT/'logs'/f'{out.name}.log').open('w')
    process=subprocess.Popen([str(PYTHON),'-u',str(ROOT/'run_episode.py'),'--config',str(item['config_path']),
        '--task',case['task'],'--eval-seed',str(case['seed']),'--layout',str(case['layout']),'--out',str(out)],
        cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    item.update(process=process,log=log,started_monotonic=time.monotonic(),started_at=time.time())
    print('Started '+out.name+' lane '+case['lane'],flush=True)


def stop(items):
    active=[x for x in items if x.get('process') and x['process'].poll() is None]
    for x in active:x['process'].send_signal(signal.SIGINT)
    deadline=time.monotonic()+120
    for x in active:
        try:x['process'].wait(timeout=max(1,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:x['process'].terminate();x['process'].wait(timeout=30)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--retry-invalid',action='store_true');args=parser.parse_args()
    config=json.loads((ROOT/'experiment_config.json').read_text())
    for name in ['runs','logs','case_configs']:(ROOT/name).mkdir(exist_ok=True)
    declared=build(config);path=ROOT/'study_declaration.json'
    if path.exists():
        if json.loads(path.read_text())!=declared:raise RuntimeError('Predeclared study changed')
    else:save(path,declared)
    active=[]
    try:
        save(ROOT/'study_status.json',{'status':'running','stage':'waiting_policy_services','planned_episodes':48,'updated_at':time.time()})
        ready(config);check_frozen(config,declared)
        for pair in declared['pairs']:
            check_frozen(config,declared)
            cases=[c for c in declared['tests'] if c['pair_id']==pair['pair_id']]
            if pair['pair_id']%2==0:cases.reverse()  # balance within-wave launch order
            items=[prepare(config,c,args.retry_invalid) for c in cases];active=items
            if not all(x['done'] for x in items):admit(config)
            for item in items:
                if not item['done']:launch(item)
            last_save=0
            while not all(x['done'] for x in items):
                for item in items:
                    if item['done']:continue
                    proc=item['process'];code=proc.poll()
                    if code is None:
                        if time.monotonic()-item['started_monotonic']>config['budget_s']+1200:
                            stop([item]);code=proc.returncode
                        else:continue
                    item['log'].close();item.update(done=True,returncode=code,finished_at=time.time())
                    p=item['out']/'episode_summary.json'
                    summary=json.loads(p.read_text()) if p.exists() else None
                    item['summary']=summary
                    item['infrastructure_error']=bool(code or not summary or not summary.get('valid'))
                    print(json.dumps({'case':item['case'],'returncode':code,'valid':(summary or {}).get('valid'),
                        'success':(summary or {}).get('success'),'score':(summary or {}).get('official_score')},ensure_ascii=False),flush=True)
                if time.monotonic()-last_save>5:
                    save(ROOT/'study_status.json',{'status':'running','stage':'paired_test','pair':pair,
                        'completed_episodes':completed(),'planned_episodes':48,
                        'workers':[{'case':x['case'],'out':str(x['out']),'done':x['done'],
                                    'pid':x['process'].pid if x.get('process') else None} for x in items],
                        'updated_at':time.time()});last_save=time.monotonic()
                time.sleep(1)
            if any(x.get('infrastructure_error') for x in items):
                raise RuntimeError('Pair contains infrastructure-invalid attempt; preserve both partners and inspect before resume')
            subprocess.run([str(PYTHON),str(ROOT/'analyze_efficiency.py')],cwd=ROOT,check=True)
        save(ROOT/'study_status.json',{'status':'completed','completed_episodes':completed(),'planned_episodes':48,
            'completed_pairs':24,'updated_at':time.time()})
        subprocess.run([str(PYTHON),str(ROOT/'analyze_efficiency.py')],cwd=ROOT,check=True)
    except BaseException as exc:
        stop(active)
        save(ROOT/'study_failure.json',{'error':type(exc).__name__+': '+str(exc),'updated_at':time.time(),
            'completed_valid':completed(),'action':'Inspect infrastructure; do not retry valid outcomes'})
        raise
    finally:
        for x in active:
            if x.get('log') and not x['log'].closed:x['log'].close()


if __name__=='__main__':main()
