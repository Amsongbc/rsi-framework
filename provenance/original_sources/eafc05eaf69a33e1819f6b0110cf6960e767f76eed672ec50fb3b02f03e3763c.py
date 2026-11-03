#!/usr/bin/env python3
"""One API-planned hybrid episode using the colleague's unchanged PyRUA loop."""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parent
PROJECTS = ROOT.parent
RUNTIME = PROJECTS/'ewam_comparison'


def configure_environment(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpus']['sam3'])
    os.environ['PATH'] = str(RUNTIME/'.venv-sim/bin')+os.pathsep+os.environ['PATH']
    os.environ['PYTHONPATH'] = str(PROJECTS/'PyRUA-Lean/src')
    os.environ.update(OMNI_KIT_ACCEPT_EULA='YES', ROBODOJO_MAX_BASH_RETRIES='1',
        SAM3_CHECKPOINT_PATH=str(PROJECTS/'RoboDojo-resources/backbones/sam3/sam3.pt'),
        HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',
        TRITON_CACHE_DIR=str(RUNTIME/'.triton-cache'),
        TORCH_EXTENSIONS_DIR=str(RUNTIME/'.torch-extensions'),
        VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',
        XDG_RUNTIME_DIR=str(RUNTIME/'xdg'),TMPDIR=str(RUNTIME/'runtime_tmp'),
        XDG_CACHE_HOME=str(RUNTIME/'.cache'),CUDA_CACHE_PATH=str(RUNTIME/'.cuda-cache'),
        NO_PROXY='localhost,127.0.0.1,::1',no_proxy='localhost,127.0.0.1,::1')
    # Retain the existing HTTPS proxy for the official API; all assets are local/offline.
    for key in ['ROBODOJO_ORACLE_STATE','ROBODOJO_IK_FEEDBACK','ROBODOJO_TASK_STEP_LIM',
                'ROBODOJO_STEP_LIM','ROBODOJO_FIXED_INSTRUCTION','ROBODOJO_ORACLE_INSTRUCTION']:
        os.environ.pop(key,None)
    sys.path[:0] = [str(PROJECTS/'PyRUA-Lean/src'),str(RUNTIME)]


def source_snapshot(out, config):
    sources = list(ROOT.glob('*.py'))+[RUNTIME/'session.py',RUNTIME/'ewam_service.py']
    sources += list((PROJECTS/'RoboDojo/XPolicyLab/policy/EWAM').rglob('*.py'))
    library = PROJECTS/'PyRUA-Lean/src/pyrualean'
    sources += list(library.rglob('*.py'))+list((library/'knowledge').glob('*.md'))
    snapshot = out/'implementation_snapshot'
    snapshot.mkdir()
    rows = []
    for path in sources:
        rel = path.relative_to(PROJECTS)
        dest = snapshot/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
        rows.append({'path':str(path),'snapshot':str(dest.relative_to(out)),
                     'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (out/'source_manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'experiment_config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n')
    for filename in ['runtime_versions.json','runtime_environment.json','analysis_artifact_schema.json']:
        if (ROOT/filename).exists():shutil.copyfile(ROOT/filename,out/filename)
    service_metadata=Path(config['service_root'])/'model.json'
    if service_metadata.exists():shutil.copyfile(service_metadata,out/'ewam_service_metadata.json')


def monitor_gpus(stop,out,gpus):
    with (out/'gpu_metrics.jsonl').open('a') as handle:
        while not stop.is_set():
            result = subprocess.run(['nvidia-smi','-i',','.join(map(str,sorted(set(gpus.values())))),
                '--query-gpu=index,memory.used,memory.free,utilization.gpu',
                '--format=csv,noheader,nounits'],capture_output=True,text=True)
            handle.write(json.dumps({'time':time.time(),'scope':'shared device totals, including other jobs',
                                     'csv':result.stdout,'returncode':result.returncode})+'\n')
            handle.flush()
            stop.wait(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'experiment_config.json')
    parser.add_argument('--task',required=True)
    parser.add_argument('--eval-seed',type=int,required=True)
    parser.add_argument('--layout',type=int,default=0)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text())
    if args.task not in config['tasks'] or args.eval_seed not in config['eval_seeds'] or args.layout not in config['layouts']:
        parser.error('Case is outside the predeclared configuration')
    if config['api']['model']!='gpt-6.1-sol' or config['api']['reasoning_effort']!='high':
        parser.error('This fixed experiment requires gpt-6.1-sol / high')
    out = args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    source_snapshot(out,config)
    configure_environment(config)
    from api_transport import ResponsesTransport
    from recording import RecordedEnv, RecordedSam3, append_json
    from hybrid_robot import robot_class as baseline_robot_class
    from doctor_robot import robot_class as doctor_robot_class, protocol as doctor_protocol
    from ewam_service import EwamClient
    from session import write_json
    from pyrualean.hosts import xpolicylab_robodojo as base
    from pyrualean.hosts import xpolicylab_robodojo_rgb as host
    from pyrualean.hosts.sam3_local import LocalSam3
    import pyrualean.robodojo_xpl_rgb as robot_module
    from pyrualean.play import play_episode
    import pyrualean.play as play_module

    doctor = config.get('doctor', {})
    context = {'doctor_phase':doctor.get('phase'), 'doctor_memory':doctor.get('frozen_memory'),
               'out':out,'session_id':out.name,'ewam':EwamClient(config['service_root']),
               'ewam_calls':0,'ewam_attempts':0,'awaiting_ewam_observation':False}
    context['efficiency'] = dict(config.get('efficiency',{}))
    transport_api = dict(config['api'],
        context_compaction=context['efficiency'].get('context_compaction',False),
        compact_threshold=context['efficiency'].get('compact_threshold',32768))
    transport = ResponsesTransport(transport_api,out)
    def visual_response():
        pending=context.get('awaiting_ewam_observation',False)
        context['awaiting_ewam_observation']=False
        if context.get('proposal'):
            context['proposal']['delivered_to_model']=True
            append_json(out/'doctor_events.jsonl',{'time':time.time(),
                'event':'proposal_and_rgb_delivered_to_model',
                'prediction_id':context['proposal']['prediction_id'],
                'native_action_index':context['proposal']['native_action_index']})
        append_json(out/'protocol_events.jsonl',{'time':time.time(),
            'event':'model_received_fresh_visual_response','after_ewam':pending,
            'native_actions':context.get('env').issued_actions if context.get('env') else None})
    transport.on_visual_response=visual_response
    original_prompt=play_module.build_prompt
    def build_prompt(*a,**kw):
        if doctor.get('enabled'):
            return original_prompt(*a,**kw)+doctor_protocol(config)
        return original_prompt(*a,**kw)+'''\n\n# Fixed experimental execution protocol\n
This is an EWAM-FIRST hybrid experiment, not a free-choice primitive-only run.
Inspect the task/RGB/state and form a meaningful manipulation subgoal. You may
use read-only perception first. Then let robo.ewam attempt ONE bounded segment
for real task progress BEFORE any analytic motion. Use its default 16 model
actions unless shorter execution is justified by an observable risk. Do not
make dummy/no-op calls just to meet a quota. End that Python cell after showing
fresh camera images with robo.show(...). Inspect the returned images in your
next model turn, then decide to continue EWAM or switch to primitives because
EWAM failed, stalled, or needs precise correction. Print a brief observable
progress/subgoal/switch reason. This ordering is enforced by the host. Every
physical action, including learned actions and corrections, shares the same
unmodified task native budget. No reward or task answer is exposed.\n'''
    play_module.build_prompt=build_prompt
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_gpus,args=(stop,out,config['gpus']),daemon=True)
    monitor.start()
    original_layout = base.lay_out
    def layout(*a,**kw):
        bench = original_layout(*a,**kw)
        (bench/'Assets').unlink()
        (bench/'Assets').symlink_to(RUNTIME/'runtime_assets/Assets',target_is_directory=True)
        script = bench/'scripts/eval_policy.sh'
        old = script.read_text()
        new = old.replace('KIT_ARGS=""',f'KIT_ARGS=" --/renderer/activeGpu={config["gpus"]["simulator"]} --/renderer/multiGpu/enabled=false"')
        if new==old:
            raise RuntimeError('Could not set run-local simulator GPU')
        script.write_text(new)
        write_json(out/'client_launcher_change.json',{'scope':'Run-local GPU launcher and asset path substitution only; no task, reward or observation changes',
                    'sim_gpu':config['gpus']['simulator'],'multi_gpu':False})
        return bench
    base.lay_out = layout
    original_boot = host.boot
    def boot(*a,**kw):
        kw['env_only'] = True
        toolkit,daemons = original_boot(*a,**kw)
        try:
            context.update(toolkit=toolkit,bridge=toolkit._primitives.env.bridge)
            # Keep simulator and segmentation on their admitted, separate devices.
            toolkit._primitives.sam3_client = RecordedSam3(LocalSam3(),context)
            env = RecordedEnv(toolkit._primitives.env,out)
            toolkit._primitives.env = env
            context['env'] = env
            context['ewam'].call('reset',session_id=context['session_id'],
                                seed=100000+args.eval_seed*1000+args.layout)
            context['lease_acquired'] = True
            return toolkit,daemons
        except BaseException:
            from pyrualean.hosts.common import stop_all
            stop_all(toolkit,daemons)
            raise
    host.boot = boot
    original_adapter = robot_module.ADAPTER
    robot_module.ADAPTER = dataclasses.replace(original_adapter,
        robot_cls=(doctor_robot_class(context) if doctor.get('enabled') else baseline_robot_class(context)),services_novla='a frozen EWAM learned motion policy and a SAM3 segmentation service')
    cell = host.Cell(suite='arx_x5',task=args.task,seed=args.layout,
                     cuda_device=config['gpus']['simulator'])
    started = time.monotonic()
    error = None
    result = None
    try:
        result = play_episode(cell,out,arm='cells',model=config['api']['model'],
            reasoning_effort=config['api']['reasoning_effort'],budget_s=config['budget_s'],
            max_turns=config['max_turns'],runtime='openai',robot='robodojo_xpl_rgb',
            primitives='no-vla',images=config['images'],feedback=config['feedback'],
            guides=config['guides'],base_url=config['api']['base_url'],max_attempts=1,
            boot_options={'source_root':config['source_root'],'eval_seed':args.eval_seed,'boot_timeout_s':900},
            openai_options={'max_tokens':config['api']['max_output_tokens'],
                'nudges':config['api']['nudges'],'request_timeout_s':config['api']['request_timeout_s'],
                'extra_body':{'reasoning_effort':config['api']['reasoning_effort']},'transport':transport})
    except BaseException:
        error = traceback.format_exc()
    finally:
        os.chdir(ROOT)
        stop.set();monitor.join(5)
        if context.get('lease_acquired'):
            with contextlib.suppress(Exception):
                context['ewam'].call('release',session_id=context['session_id'])
        if context.get('env'):
            context['env'].close_recording()
        transport.close()
    raw_results = []
    for path in (out/'bench/eval_result').glob('**/_result.json'):
        raw_results.append({'path':str(path),'value':json.loads(path.read_text())})
    details = [detail for row in raw_results for detail in row['value'].get('details',{}).values()
               if detail.get('layout_id')==args.layout]
    verdict = None
    if context.get('bridge'):
        with contextlib.suppress(Exception):
            verdict = context['bridge'].verdict(timeout_s=5)
    policy_status = (result or {}).get('status')
    valid = bool(result and not error and len(raw_results)==1 and len(details)==1 and verdict is not None
        and raw_results[0]['value'].get('eval_time')==1
        and policy_status not in ['boot_error','planner_error','error']
        and bool(details[0]['success'])==bool(verdict.get('success'))
        and bool(details[0]['success'])==bool(result.get('environment_success')))
    env = context.get('env')
    summary = {'task':args.task,'eval_seed':args.eval_seed,'layout':args.layout,
        'mode':'api_hybrid','benchmark_episode':True,'valid':valid,
        'success':bool(details[0]['success']) if valid else None,
        'official_score':details[0].get('score') if valid else None,
        'client_verdict':verdict,'client_results':raw_results,
        'policy_status':policy_status,'ewam_calls':context['ewam_calls'],
        'execution_strategy':config['execution_strategy'],
        'doctor_enabled':doctor.get('enabled',False),'doctor_phase':doctor.get('phase'),
        'diagnosis_count':len(context.get('diagnoses',[])),
        'ewam_executed_prefixes':context['ewam_attempts'],
        'ewam_attempts':context['ewam_attempts'],
        'ewam_first_attempted':context['ewam_attempts']>0,
        'first_native_action_owner':None if env is None else env.first_native_owner,
        'native_actions_issued':None if env is None else env.issued_actions,
        'native_actions_by_owner':None if env is None else env.actions_by_owner,
        'native_limit':None if env is None else env.initial_limit,
        'startup_s':(result or {}).get('startup_s'),'control_wall_s':(result or {}).get('policy_wall_s'),
        'total_wall_s':time.monotonic()-started,'error':error,
        'model':config['api']['model'],'reasoning_effort':config['api']['reasoning_effort'],
        'api_requests':transport.requests,'api_successful_responses':transport.responses,
        'context_compactions':transport.compactions,
        'efficiency':context['efficiency'],
        'generation':(result or {}).get('generation')}
    write_json(out/'episode_summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['task','eval_seed','valid','success','official_score','ewam_calls','native_actions_issued','policy_status']},ensure_ascii=False),flush=True)
    return 0 if valid else 1


if __name__=='__main__':
    raise SystemExit(main())
