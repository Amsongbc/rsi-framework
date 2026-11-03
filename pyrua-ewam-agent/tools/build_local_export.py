#!/usr/bin/env python3
"""Recreate an allowlisted export from a local workspace; never writes upstream."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

DEST=Path(__file__).resolve().parents[1]
PROJECTS=Path(os.environ.get('PROJECTS_ROOT',str(DEST.parent)))
COHORTS=['ewam_comparison','ewam_doctor_integration','ewam_doctor_validation72','ewam_doctor_efficiency48','ewam_window_parallel24']
ALLOW={
'ewam_comparison': ['audit_pairs.py','control.py','ewam_service.py','prepare_runtime_assets.py','run_sequence.py','session.py','smoke_ewam.py','smoke_sam3.py','smoke_sim.py','summarize.py','test_episode_contract.py'],
'ewam_doctor_integration':['analyze_study.py','api_transport.py','audit_study.py','doctor_robot.py','hybrid_robot.py','learning.py','recording.py','run_episode.py','run_study.py','test_doctor.py'],
'ewam_doctor_validation72':['analyze_validation.py','api_transport.py','audit_study.py','doctor_robot.py','hybrid_robot.py','learning.py','metrics.py','recording.py','run_episode.py','run_validation.py','study_plan.py','test_doctor.py'],
'ewam_doctor_efficiency48':['analyze_efficiency.py','api_preflight.py','api_transport.py','audit_study.py','doctor_robot.py','hybrid_robot.py','learning.py','metrics.py','recording.py','run_episode.py','run_validation.py','study_plan.py','test_doctor.py','test_efficiency.py','test_plan.py'],
'ewam_window_parallel24':['doctor_robot.py','hybrid_robot.py','learning.py','recording.py','window_control.py','window_protocol_tests.py','window_session.py']}


def digest(b):return hashlib.sha256(b).hexdigest()
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def portable_source(text):
    # Keep original code in content-addressed provenance; executable copies accept
    # an external sibling-projects layout without altering control semantics.
    lines=text.splitlines(keepends=True);tree=ast.parse(text)
    cut=0
    for node in tree.body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str) and not cut:cut=node.end_lineno;continue
        if isinstance(node,ast.ImportFrom) and node.module=='__future__':cut=node.end_lineno;continue
        break
    if cut==0 and lines and lines[0].startswith('#!'):cut=1
    lines[cut:cut]=['\n# Export portability change: resolve external projects via PROJECTS_ROOT.\n','import os as _export_os\n']
    text=''.join(lines)
    text=re.sub(r'(?m)^(PROJECTS\s*=\s*)(HERE|ROOT)\.parent\s*$',r'\1Path(_export_os.environ.get("PROJECTS_ROOT", str(\2.parent)))',text)
    text=text.replace("Path(__file__).resolve().parents[1]/'PyRUA-Lean/src'","Path(_export_os.environ.get('PROJECTS_ROOT', str(Path(__file__).resolve().parents[1])))/'PyRUA-Lean/src'")
    text=text.replace("Path(__file__).resolve().parents[1]/'ewam_comparison'","Path(_export_os.environ.get('PROJECTS_ROOT', str(Path(__file__).resolve().parents[1])))/'ewam_comparison'")
    return text.rstrip()+'\n'


def sanitize_config(c):
    c=dict(c)
    c.pop('reference_baselines',None);c.pop('frozen_experience',None)
    if 'doctor' in c:c['doctor']=dict(enabled=True,phase='hybrid',frozen_memory=None)
    def walk(v,key=''):
        if isinstance(v,dict):return {k:walk(x,k) for k,x in v.items()}
        if isinstance(v,list):return [walk(x,key) for x in v]
        if isinstance(v,str):
            if key in ('key_file','api_key_file'):return '${API_KEY_FILE}'
            v=v.replace(str(PROJECTS),'${PROJECTS_ROOT}')
            if v.startswith('/mnt/') or v.startswith('/home/'):return '${LOCAL_PATH}'
        return v
    return walk(c)


def main():
    rows=[]; original_objects=set()
    for cohort,names in ALLOW.items():
        for name in names:
            src=PROJECTS/cohort/name
            if not src.exists():continue
            raw=src.read_bytes(); text=raw.decode()
            if re.search(r'/mnt/home/|/home/[A-Za-z0-9_.-]+/',text):raise RuntimeError('Absolute private source path requires explicit sanitizer: '+cohort+'/'+name)
            sha=digest(raw);original_objects.add(sha)
            original=DEST/'provenance/original_sources'/(sha+'.py')
            original.parent.mkdir(parents=True,exist_ok=True);original.write_bytes(raw)
            output=portable_source(text).encode()
            target=DEST/'projects'/cohort/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(output)
            rows.append(dict(original=cohort+'/'+name,original_sha256=sha,original_object=str(original.relative_to(DEST)),export=str(target.relative_to(DEST)),export_sha256=digest(output),changes='PROJECTS_ROOT portability import/path resolution; original content retained separately'))
        cfg=PROJECTS/cohort/'experiment_config.json'
        if cfg.exists():dump(DEST/'configs'/(cohort+'.example.json'),sanitize_config(json.loads(cfg.read_text())))
    for name in ['inference_constraints.txt','sim_constraints.txt']:
        shutil.copyfile(PROJECTS/'ewam_comparison'/name,DEST/name)
    for source,target in [('PyRUA-Lean/LICENSE','Apache-2.0.txt'),('PyRUA-Lean/NOTICE','PyRUA-Lean-NOTICE.txt'),('RoboDojo/LICENSE','RoboDojo-MIT.txt'),('RoboDojo/XPolicyLab/LICENSE','XPolicyLab-Apache-2.0.txt')]:
        shutil.copyfile(PROJECTS/source,DEST/'licenses'/target)
    dump(DEST/'provenance/source_manifest.json',rows)
    dump(DEST/'EXPORT_SCOPE.json',dict(schema=1,cohorts=COHORTS,source_files=len(rows),unique_exact_original_objects=len(original_objects),
        includes=['top-level allowlisted experiment/runtime/test/analysis Python','credential-free config templates','exact original source versions content-addressed and deduplicated','original and exported SHA256 manifest','upstream license notices'],
        excluded=['all runs/workspaces/observations','logs and raw/API records','images and videos','hidden source files and .env','keys/authkeys/sockets','environments and caches','model weights and checkpoints','downloaded assets and archives','historical frozen task memories/results','third-party flash-attention setup source','artifact packers with private key paths'],
        runtime_warning='External PyRUA-Lean, RoboDojo/XPolicyLab, licensed Isaac Sim runtime and separately obtained EWAM/SAM assets required. Historical orchestrators need fresh declarations/harnesses; no study outputs included.',
        snapshot_note='original_sources are exact original Python bytes, not sanitized config snapshots; private absolute source paths forbidden by exporter. No upstream sources or runtime files modified.'))
    print(json.dumps({'source_files':len(rows),'unique_originals':len(original_objects)}))


if __name__=='__main__':main()
