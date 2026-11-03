#!/usr/bin/env python3
"""Direct current-window controller. No remote planner transport is used."""
from __future__ import annotations

# Export portability change: resolve external projects via PROJECTS_ROOT.
import os as _export_os
import argparse
import contextlib
import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECTS = Path(_export_os.environ.get("PROJECTS_ROOT", str(ROOT.parent)))
RUNTIME = PROJECTS / 'ewam_comparison'
CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')


def plain(value):
    if dataclasses.is_dataclass(value): return plain(dataclasses.asdict(value))
    if isinstance(value, dict): return {str(k): plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [plain(v) for v in value]
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    return value


def write_json(path,value):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(plain(value),indent=2,ensure_ascii=False)+'\n'); os.replace(tmp,path)


def append_json(path,value):
    with Path(path).open('a') as f: f.write(json.dumps(plain(value),ensure_ascii=False)+'\n')


class CountingEnv:
    """One unmodified native action budget across all action owners."""
    def __init__(self,env):
        self.env=env; self.issued_actions=0; self.owner='primitive'
        self.actions_by_owner={'primitive':0,'ewam':0}
    def get_obs(self): return self.env.get_obs()
    def get_status(self): return self.env.get_status()
    def episode_over(self): return self.env.episode_over()
    def conclude(self): return self.env.conclude()
    def step(self,action):
        status=self.get_status()
        if max(int(status.get('step',0)),self.issued_actions)>=int(status['step_limit']):
            from pyrualean._backend import EpisodeFinished
            raise EpisodeFinished('Shared native action budget exhausted')
        self.issued_actions+=1; self.actions_by_owner[self.owner]+=1
        return self.env.step(action)


def acknowledge_visual(context,latest,command):
    """Validate explicit window-agent provenance against the latest saved RGB."""
    env=context['env']
    if type(command.get('observation_index')) is not int or command['observation_index']!=latest['observation_index']:
        raise ValueError('Visual acknowledgement must name the exact latest observation_index')
    if (type(command.get('native_action_index')) is not int
            or command['native_action_index']!=env.issued_actions
            or latest['issued_actions']!=env.issued_actions):
        raise ValueError('Visual acknowledgement is stale at current native action index')
    images=command.get('image_sha256')
    if not isinstance(images,dict) or not images: raise ValueError('Name at least one actually viewed camera and its SHA256')
    for camera,digest in images.items():
        if camera not in latest['images']: raise ValueError('Unknown camera')
        actual=hashlib.sha256(Path(latest['images'][camera]).read_bytes()).hexdigest()
        if digest!=latest['image_sha256'][camera] or digest!=actual: raise ValueError('Camera SHA256 mismatch')
    if command.get('viewed_by')!='window_AI' or not str(command.get('evidence','')).strip():
        raise ValueError('Explicit viewed_by=window_AI and visible evidence are required')
    proposal=context.get('proposal')
    if proposal and proposal['native_action_index']!=env.issued_actions: raise ValueError('Proposal is stale')
    provenance={k:command[k] for k in ('observation_index','native_action_index','image_sha256','viewed_by','evidence')}
    provenance.update(command_id=command.get('id'),time=time.time())
    pending=context.get('awaiting_ewam_observation',False)
    context['awaiting_ewam_observation']=False
    if proposal:
        proposal['delivered_to_model']=True
        append_json(context['out']/'doctor_events.jsonl',dict(event='proposal_and_rgb_delivered_to_model',prediction_id=proposal['prediction_id'],**provenance))
    append_json(context['out']/'protocol_events.jsonl',dict(event='model_received_fresh_visual_response',after_ewam=pending,native_actions=env.issued_actions,**provenance))
    append_json(context['out']/'visual_acknowledgements.jsonl',provenance)
    return provenance


def window_robot_class(context):
    from doctor_robot import robot_class
    from pyrualean._backend import ToolError
    FrozenDoctor=robot_class(context)
    class WindowDoctor(FrozenDoctor):
        def _stateful(self,tool,kwargs,build):
            if not context.get('ewam_attempts',0):
                append_json(context['out']/'protocol_events.jsonl',dict(time=time.time(),event='analytic_motion_blocked_before_ewam',tool=tool))
                raise ToolError(tool,'Execute one meaningful bounded EWAM segment before primitive motion.')
            return super()._stateful(tool,kwargs,build)
    return WindowDoctor


def configure_environment(config):
    os.environ['CUDA_VISIBLE_DEVICES']=str(config['gpus']['sam3'])
    os.environ['PATH']=str(RUNTIME/'.venv-sim/bin')+os.pathsep+os.environ['PATH']
    os.environ['PYTHONPATH']=str(PROJECTS/'PyRUA-Lean/src')
    os.environ.update(OMNI_KIT_ACCEPT_EULA='YES',ROBODOJO_MAX_BASH_RETRIES='1',
        SAM3_CHECKPOINT_PATH=str(PROJECTS/'RoboDojo-resources/backbones/sam3/sam3.pt'),
        HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',
        MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',TRITON_CACHE_DIR=str(RUNTIME/'.triton-cache'),
        TORCH_EXTENSIONS_DIR=str(RUNTIME/'.torch-extensions'),VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',
        XDG_RUNTIME_DIR=str(RUNTIME/'xdg'),TMPDIR=str(RUNTIME/'runtime_tmp'),
        XDG_CACHE_HOME=str(RUNTIME/'.cache'),CUDA_CACHE_PATH=str(RUNTIME/'.cuda-cache'),NO_PROXY='*',no_proxy='*')
    for key in list(os.environ):
        if (key.lower().endswith('_proxy') and key.lower()!='no_proxy') or key in {
            'ROBODOJO_ORACLE_STATE','ROBODOJO_IK_FEEDBACK','ROBODOJO_TASK_STEP_LIM',
            'ROBODOJO_STEP_LIM','ROBODOJO_FIXED_INSTRUCTION','ROBODOJO_ORACLE_INSTRUCTION'}:
            os.environ.pop(key,None)
    sys.path[:0]=[str(ROOT),str(PROJECTS/'PyRUA-Lean/src'),str(RUNTIME)]


def source_snapshot(out,config):
    sources=list(ROOT.glob('*.py'))+[RUNTIME/'session.py',RUNTIME/'ewam_service.py']
    sources+=list((PROJECTS/'RoboDojo/XPolicyLab/policy/EWAM').rglob('*.py'))
    library=PROJECTS/'PyRUA-Lean/src/pyrualean'
    sources+=list(library.rglob('*.py'))+list((library/'knowledge').glob('*.md'))
    rows=[]
    for path in sources:
        dest=out/'implementation_snapshot'/path.relative_to(PROJECTS)
        dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,dest)
        rows.append(dict(path=str(path),snapshot=str(dest.relative_to(out)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write_json(out/'source_manifest.json',rows); write_json(out/'experiment_config.json',config)
    metadata=Path(config['service_root'])/'model.json'
    if not metadata.exists(): raise FileNotFoundError(f'Exact EWAM service metadata required: {metadata}')
    shutil.copyfile(metadata,out/'ewam_service_metadata.json')


class Session:
    def __init__(self,args,config):
        self.args=args; self.config=config; self.out=args.out.resolve()
        self.out.mkdir(parents=True,exist_ok=False)
        for name in ('commands','responses','workspace'): (self.out/name).mkdir()
        self.started=time.monotonic(); self.ready_at=None; self.cells=0; self.observations=0
        self.errors=[]; self.toolkit=None; self.daemons=[]; self.stop_reason=None; self.latest=None
        self.context=dict(out=self.out,session_id=self.out.name,doctor_phase='hybrid',
            doctor_memory=config.get('doctor',{}).get('frozen_memory'),
            efficiency=dict(config.get('efficiency',{}),sparse_supervision=args.variant=='sparse'),
            ewam_calls=0,ewam_attempts=0,awaiting_ewam_observation=False)

    def boot(self):
        from recording import RecordedEnv,RecordedSam3
        from ewam_service import EwamClient
        from pyrualean.hosts import xpolicylab_robodojo as base
        from pyrualean.hosts import xpolicylab_robodojo_rgb as host
        from pyrualean.hosts.sam3_local import LocalSam3
        from pyrualean.sandbox import make_builtins
        source_snapshot(self.out,self.config)
        original=base.lay_out
        def layout(*a,**kw):
            bench=original(*a,**kw)
            (bench/'Assets').unlink(); (bench/'Assets').symlink_to(RUNTIME/'runtime_assets/Assets',target_is_directory=True)
            script=bench/'scripts/eval_policy.sh'; old=script.read_text()
            new=old.replace('KIT_ARGS=""',f'KIT_ARGS=" --/renderer/activeGpu={self.config["gpus"]["simulator"]} --/renderer/multiGpu/enabled=false"')
            if new==old: raise RuntimeError('Could not set run-local simulator GPU')
            script.write_text(new)
            write_json(self.out/'client_launcher_change.json',dict(scope='Run-local asset path and GPU launcher only',sim_gpu=self.config['gpus']['simulator'],multi_gpu=False))
            return bench
        base.lay_out=layout
        cell=host.Cell(suite='arx_x5',task=self.args.task,seed=self.args.layout,cuda_device=self.config['gpus']['simulator'])
        self.toolkit,self.daemons=host.boot(cell,self.out,env_only=True,source_root=self.config.get('source_root',str(PROJECTS/'RoboDojo')),eval_seed=self.args.eval_seed,boot_timeout_s=900)
        self.bridge=self.toolkit._primitives.env.bridge
        self.env=RecordedEnv(self.toolkit._primitives.env,self.out); self.toolkit._primitives.env=self.env
        session=self
        class RecordedEwam:
            def __init__(self): self.client=EwamClient(session.config['service_root'])
            def call(self,op,**kwargs):
                started=time.monotonic(); error=None
                try: return self.client.call(op,**kwargs)
                except Exception:
                    error=traceback.format_exc(); session.errors.append(error); raise
                finally:
                    append_json(session.out/'ewam_service_events.jsonl',dict(time=time.time(),op=op,
                        session_id=kwargs.get('session_id'),wall_s=time.monotonic()-started,error=error))
        self.context.update(env=self.env,toolkit=self.toolkit,ewam=RecordedEwam())
        self.toolkit._primitives.sam3_client=RecordedSam3(LocalSam3(),self.context)
        self.context['ewam'].call('reset',session_id=self.context['session_id'],seed=100000+self.args.eval_seed*1000+self.args.layout)
        self.context['lease_acquired']=True
        self.robo=window_robot_class(self.context)(host.make_backend(self.toolkit),primitives='no-vla')
        self.namespace={'__builtins__':make_builtins(self.out/'workspace'),'robo':self.robo,'np':np}
        write_json(self.out/'run_config.json',dict(task=self.args.task,eval_seed=self.args.eval_seed,layout=self.args.layout,
            variant=self.args.variant,lane=self.args.lane,planner='window_AI',requested_planner='window_AI',
            exact_model='unverified',api_requests=0,token_usage='unavailable',native_limit=self.env.initial_limit,
            max_cells=100,wall_timeout_s=5400,max_model_actions=16,max_native_prefix=48,
            EWAM_seed=100000+self.args.eval_seed*1000+self.args.layout))
        self.snapshot(); self.ready_at=time.monotonic()

    def snapshot(self):
        from PIL import Image
        obs=self.env.get_obs(); folder=self.out/'observations'/f'{self.observations:04d}'; folder.mkdir(parents=True)
        paths={}; hashes={}
        for camera in CAMERAS:
            path=folder/(camera+'.png'); Image.fromarray(np.asarray(obs['vision'][camera]['color'])[...,:3].astype(np.uint8)).save(path)
            paths[camera]=str(path); hashes[camera]=hashlib.sha256(path.read_bytes()).hexdigest()
        status=self.env.get_status()
        value=dict(observation_index=self.observations,instruction=str(obs.get('instruction') or obs.get('instructions') or ''),
            robot_state=plain(self.robo.state()),step=int(status.get('step',0)),issued_actions=self.env.issued_actions,
            native_action_index=self.env.issued_actions,step_limit=self.env.initial_limit,
            native_budget_remaining=max(0,self.env.initial_limit-max(self.env.issued_actions,int(status.get('step',0)))),
            images=paths,image_sha256=hashes,python_cells=self.cells,
            awaiting_ewam_observation=self.context.get('awaiting_ewam_observation',False),
            proposal=None if not self.context.get('proposal') else {k:self.context['proposal'][k] for k in ('prediction_id','native_action_index','delivered_to_model')})
        self.observations+=1; self.latest=value
        write_json(folder/'observation.json',value); write_json(self.out/'latest.json',value)
        return value

    def python(self,code):
        from pyrualean.sandbox import audit_source,blocked_names
        from pyrualean._backend import EpisodeFinished
        if self.cells>=100: raise RuntimeError('100-cell budget exhausted')
        flags=audit_source(code,workspace=self.out/'workspace'); blocked=blocked_names(code)
        if flags or blocked: raise ValueError(f'Policy audit flags={flags}, blocked={blocked}')
        self.cells+=1; path=self.out/'workspace'/f'cell_{self.cells:03d}.py'; path.write_text(code)
        stdout,stderr=io.StringIO(),io.StringIO(); error=None
        with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
            try: exec(compile(code,str(path),'exec'),self.namespace)
            except EpisodeFinished as exc: self.stop_reason=exc.reason
            except Exception: error=traceback.format_exc()
        result=dict(stdout=stdout.getvalue(),stderr=stderr.getvalue(),error=error)
        append_json(self.out/'python_cells.jsonl',dict(index=self.cells,code=code,**result)); return result

    def loop(self):
        while not self.env.episode_over():
            if self.stop_reason: break
            if self.env.issued_actions>=self.env.initial_limit: self.stop_reason='native_budget'; break
            if time.monotonic()-self.ready_at>5400: self.stop_reason='wall_timeout'; break
            if not all(getattr(d,'alive',lambda:True)() for d in self.daemons): raise RuntimeError('Evaluation client exited before verdict')
            commands=sorted((self.out/'commands').glob('*.json'),key=lambda p:p.stat().st_mtime_ns)
            if not commands: time.sleep(.2); continue
            path=commands[0]; identifier=path.stem; raw_command=path.read_text()
            os.replace(path,path.with_suffix('.accepted'))
            response={}
            try:
                command=json.loads(raw_command)
                if not isinstance(command,dict): raise ValueError('Command must be an object')
                append_json(self.out/'commands.jsonl',dict(time=time.time(),**command))
                if not str(command.get('note','')).strip(): raise ValueError('Reviewable command note is required')
                if command.get('id')!=identifier: raise ValueError('Command ID must equal filename stem')
                op=command['op']
                if op=='python': response['output']=self.python(command['code'])
                elif op=='observe': response['output']=None
                elif op=='ack_visual': response['output']=acknowledge_visual(self.context,self.latest,command)
                elif op=='finish': self.stop_reason='planner_finish'; response['output']={'finishing':True}
                else: raise ValueError(f'Unknown operation: {op}')
            except Exception:
                response['error']=traceback.format_exc(); append_json(self.out/'command_errors.jsonl',dict(id=identifier,raw_command=raw_command,**response))
            response['observation']=self.snapshot()
            write_json(self.out/'responses'/(identifier+'.json'),response)

    def close(self):
        policy_end=time.monotonic(); verdict=None
        if self.toolkit is not None:
            try:
                self.toolkit.conclude(); verdict=self.bridge.verdict(timeout_s=60)
            except Exception: self.errors.append(traceback.format_exc())
        if self.context.get('lease_acquired'):
            try: self.context['ewam'].call('release',session_id=self.context['session_id'])
            except Exception: self.errors.append(traceback.format_exc())
        for cleanup in [lambda: self.env.close_recording() if hasattr(self,'env') else None,
                        lambda: self.robo._ledger.write_jsonl(self.out/'primitive_calls.jsonl') if hasattr(self,'robo') else None]:
            try: cleanup()
            except Exception: self.errors.append(traceback.format_exc())
        from pyrualean.hosts.common import stop_all
        stop_all(self.toolkit,self.daemons)
        raw=[dict(path=str(p),value=json.loads(p.read_text())) for p in (self.out/'bench/eval_result').glob('**/_result.json')]
        details=[d for row in raw for d in row['value'].get('details',{}).values() if d.get('layout_id')==self.args.layout]
        valid=bool(not self.errors and verdict is not None and len(raw)==1 and len(details)==1 and raw[0]['value'].get('eval_time')==1 and bool(details[0]['success'])==bool(verdict.get('success')))
        summary=dict(task=self.args.task,eval_seed=self.args.eval_seed,layout=self.args.layout,variant=self.args.variant,lane=self.args.lane,
            planner='window_AI',requested_planner='window_AI',exact_model='unverified',api_requests=0,token_usage='unavailable',
            valid=valid,success=bool(details[0]['success']) if valid else None,official_score=details[0].get('score') if valid else None,
            outcome_class='valid_task_success' if valid and details[0]['success'] else 'valid_task_failure' if valid else 'technical_interruption',
            client_verdict=verdict,client_results=raw,stop_reason=self.stop_reason,errors=self.errors,
            python_cells=self.cells,ewam_calls=self.context['ewam_calls'],ewam_attempts=self.context['ewam_attempts'],
            native_actions_issued=getattr(getattr(self,'env',None),'issued_actions',0),
            native_actions_by_owner=getattr(getattr(self,'env',None),'actions_by_owner',{}),
            first_native_action_owner=getattr(getattr(self,'env',None),'first_native_owner',None),
            startup_s=None if self.ready_at is None else self.ready_at-self.started,
            control_wall_s=None if self.ready_at is None else policy_end-self.ready_at,total_wall_s=time.monotonic()-self.started)
        write_json(self.out/'episode_summary.json',summary); write_json(self.out/'result.json',summary); return summary

    def run(self):
        try: self.boot(); self.loop()
        except BaseException:
            self.errors.append(traceback.format_exc()); self.stop_reason='infrastructure_error'
            print(self.errors[-1],flush=True)
        return self.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True); p.add_argument('--task',required=True)
    p.add_argument('--eval-seed',type=int,required=True); p.add_argument('--layout',type=int,default=0)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--variant',choices=['full','sparse'],required=True)
    p.add_argument('--lane',choices=['A','B','C'],required=True)
    a=p.parse_args(); config=json.loads(a.config.read_text())
    # A lane override can place SAM and simulator on any admitted device, including EWAM's.
    if a.lane in config.get('lanes',{}):
        lane=config['lanes'][a.lane]
        gpu=lane.get('gpu')
        override={k:lane[k] for k in ('simulator','sam3') if k in lane}
        if gpu is not None: override=dict({'simulator':gpu,'sam3':gpu},**override)
        config['gpus']=dict(config.get('gpus',{}),**lane.get('gpus',{}),**override)
        if 'service_root' in lane: config['service_root']=lane['service_root']
    for field,value in [('tasks',a.task),('eval_seeds',a.eval_seed),('layouts',a.layout)]:
        if field in config and value not in config[field]: p.error(f'{value} outside configured {field}')
    configure_environment(config); result=Session(a,config).run(); print(json.dumps(result),flush=True)
    return 0 if result['valid'] else 1


if __name__=='__main__': raise SystemExit(main())
