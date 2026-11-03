#!/usr/bin/env python3
"""Nine independent warmups and twelve counterbalanced paired evaluations."""

# Export portability change: resolve external projects via PROJECTS_ROOT.
import os as _export_os
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from learning import summarize_episode, freeze_harness, validate_memory

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT.parent/'ewam_comparison/.venv-ewam/bin/python'
sys.path.insert(0, str(ROOT.parent/'ewam_comparison'))


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n'); tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def declaration(config):
    warmups = [{'task':t, 'seed':s, 'layout':1, 'arm':'warmup'}
               for t in config['tasks'] for s in [0,1,2]]
    tests = [{'task':t, 'seed':s, 'layout':0, 'arm':arm}
             for t in config['tasks'] for s in [0,1]
             for arm in (['baseline','doctor'] if s == 0 else ['doctor','baseline'])]
    return {'warmups':warmups, 'tests':tests, 'planned_episodes':len(warmups)+len(tests),
            'test_seeds_frozen':True, 'warmup_test_layouts_disjoint':True,
            'valid_failures_never_retried':True, 'warmup_and_learning_costs_included':True,
            'statistical_scope':'Exploratory bundled-mechanism study; 2 test episodes/task/arm, not a full benchmark or component ablation',
            'control_files':{n:sha(ROOT/n) for n in ['run_episode.py','hybrid_robot.py','doctor_robot.py','api_transport.py','recording.py','learning.py']}}


def admit(config):
    for role, gib in [('simulator',16),('sam3',10)]:
        while True:
            p = subprocess.run(['nvidia-smi','-i',str(config['gpus'][role]),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            if int(p.stdout.strip()) >= gib*1024: break
            print('Waiting for free GPU memory: '+role, flush=True);time.sleep(20)


def episode(config, case, memory=None, branch=None):
    for name, expected in json.loads((ROOT/'study_declaration.json').read_text())['control_files'].items():
        if sha(ROOT/name) != expected: raise RuntimeError('Frozen controller changed: '+name)
    tag = f"{case['task']}_{case['arm']}_s{case['seed']}_l{case['layout']}"
    out = ROOT/'runs'/tag
    summary = out/'episode_summary.json'
    if summary.exists():
        row = json.loads(summary.read_text())
        if row['valid']: return out, row
        raise RuntimeError('Infrastructure-invalid attempt retained; inspect before retry: '+tag)
    if out.exists(): raise RuntimeError('Incomplete attempt retained; inspect: '+tag)
    config = json.loads(json.dumps(config))
    arm = case['arm'];doctor = config['doctor']
    doctor.update(enabled=arm!='baseline', phase='warmup' if arm=='warmup' else branch,
                  frozen_memory=memory if arm!='baseline' else None)
    config['study_case'] = case
    if arm == 'baseline':
        config['execution_strategy'] = 'ewam_first_then_observe_then_primitive_correction'
    else:
        config['execution_strategy'] = 'proposal_preview_prefix_observe_with_frozen_warmup_experience'
    path = ROOT/'case_configs'/f'{tag}.json';path.parent.mkdir(exist_ok=True);save(path, config)
    admit(config)
    completed = sum(json.loads(p.read_text())['valid'] for p in (ROOT/'runs').glob('*/episode_summary.json'))
    save(ROOT/'study_status.json', {'status':'running','stage':arm,'case':case,'out':str(out),
         'completed_episodes':completed,'planned_episodes':21,'updated_at':time.time()})
    environment = dict(os.environ)
    for key in list(environment):
        if key.startswith('CONDA'):environment.pop(key)
    with (ROOT/'logs'/f'{tag}.log').open('w') as log:
        process = subprocess.Popen([str(PYTHON),'-u',str(ROOT/'run_episode.py'),
            '--config',str(path),'--task',case['task'],'--eval-seed',str(case['seed']),
            '--layout',str(case['layout']),'--out',str(out)],cwd=ROOT,env=environment,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try: code = process.wait(timeout=config['budget_s']+1200)
        except BaseException:
            process.send_signal(signal.SIGINT)
            try: process.wait(timeout=120)
            except subprocess.TimeoutExpired: process.terminate();process.wait(timeout=30)
            raise
    if code or not summary.exists():raise RuntimeError('Episode infrastructure error: '+tag)
    row = json.loads(summary.read_text())
    if not row['valid']:raise RuntimeError('Invalid native evaluation: '+tag)
    print(json.dumps({'case':case,'success':row['success'],'score':row['official_score'],
                      'native_actions':row['native_actions_issued'],'ewam_inferences':row['ewam_calls']}),flush=True)
    return out, row


def main():
    config = json.loads((ROOT/'experiment_config.json').read_text())
    (ROOT/'logs').mkdir(exist_ok=True)
    declared = declaration(config);path = ROOT/'study_declaration.json'
    if path.exists():
        if json.loads(path.read_text()) != declared:raise RuntimeError('Study declaration mismatch')
    else:save(path,declared)
    frozen = {}
    try:
        from ewam_service import EwamClient
        save(ROOT/'study_status.json',{'status':'running','stage':'waiting_ewam_service','updated_at':time.time()})
        for _ in range(450):
            try:
                if EwamClient(config['service_root']).call('health').get('status') == 'ready':break
            except (OSError,EOFError,ConnectionError):pass
            time.sleep(2)
        else:raise RuntimeError('EWAM service startup exceeded 900 seconds')
        for task in config['tasks']:
            memory=None;scores=[]
            for case in [c for c in declared['warmups'] if c['task']==task]:
                out,row = episode(config,case,memory,branch='warmup');scores.append(row['official_score'])
                stage = ROOT/'learning'/out.name
                if (stage/'experience.json').exists():
                    memory=json.loads((stage/'experience.json').read_text());validate_memory(memory,task)
                else:
                    if stage.exists():raise RuntimeError('Incomplete learning stage retained: '+str(stage))
                    save(ROOT/'study_status.json',{'status':'running','stage':'warmup_summary','task':task,
                        'source_episode':str(out),'updated_at':time.time()})
                    memory=summarize_episode(config['api'],out,memory,stage)
            stage = ROOT/'harness'/task
            if (stage/'gate.json').exists():record=json.loads((stage/'gate.json').read_text())
            else:
                if stage.exists():raise RuntimeError('Incomplete harness stage retained')
                record=freeze_harness(config['api'],task,memory,scores,stage)
            frozen[task]=record
            print('Frozen '+task+': '+json.dumps(record),flush=True)
        for case in declared['tests']:
            record=frozen[case['task']];path=ROOT/'harness'/case['task']/'frozen_experience.json'
            if sha(path)!=record['sha256']:raise RuntimeError('Frozen test memory was changed')
            memory=json.loads(path.read_text()) if record['branch']=='hybrid' else None
            episode(config,case,memory,branch=record['branch'])
            subprocess.run([str(PYTHON),str(ROOT/'analyze_study.py')],cwd=ROOT,check=True)
        save(ROOT/'study_status.json',{'status':'completed','completed_episodes':21,
             'warmups':9,'test_episodes':12,'updated_at':time.time()})
        subprocess.run([str(PYTHON),str(ROOT/'analyze_study.py')],cwd=ROOT,check=True)
    except BaseException as exc:
        save(ROOT/'study_failure.json',{'error':type(exc).__name__+': '+str(exc),
             'time':time.time(),'action':'Preserve all attempts; inspect infrastructure before resuming'})
        raise


if __name__ == '__main__': main()
