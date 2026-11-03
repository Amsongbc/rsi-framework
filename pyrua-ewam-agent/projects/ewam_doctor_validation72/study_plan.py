"""Fixed held-out cases, crossed GPU lanes and analysis commitments."""

# Export portability change: resolve external projects via PROJECTS_ROOT.
import os as _export_os
import hashlib
import json
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parent
CONTROL_FILES=('run_episode.py','hybrid_robot.py','doctor_robot.py','api_transport.py','recording.py','learning.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(config):
    pairs=[]
    for ti, task in enumerate(config['tasks']):
        for li, layout in enumerate(config['layouts']):
            for seed in config['eval_seeds']:
                pairs.append({'task':task,'seed':seed,'layout':layout,
                              'baseline_lane':'A' if (li+seed+ti)%2==0 else 'B'})
    random.Random(20261101).shuffle(pairs)
    pairs=[dict(pair_id=i+1,**p) for i,p in enumerate(pairs)]
    tests=[]
    for p in pairs:
        for arm in ('baseline','doctor'):
            lane=p['baseline_lane'] if arm=='baseline' else ('B' if p['baseline_lane']=='A' else 'A')
            tests.append({k:p[k] for k in ['task','seed','layout','pair_id']}|{'arm':arm,'lane':lane})
    assert len(pairs)==36 and len(tests)==72
    assert len({(t['task'],t['seed'],t['layout'],t['arm']) for t in tests})==72
    for task in config['tasks']:
        for arm in ('baseline','doctor'):
            rr=[r for r in tests if r['task']==task and r['arm']==arm]
            assert len(rr)==12
            assert sum(r['lane']=='A' for r in rr)==sum(r['lane']=='B' for r in rr)==6
            for seed in config['eval_seeds']:
                assert sum(r['lane']=='A' and r['seed']==seed for r in rr)==2
    hashes={name:digest(ROOT/name) for name in CONTROL_FILES}
    gates={task:{'experience_sha256':digest(ROOT/'harness'/task/'frozen_experience.json'),
                 'gate_sha256':digest(ROOT/'harness'/task/'gate.json')} for task in config['tasks']}
    return {'cohort_id':config['cohort_id'],'warmups':[],'pairs':pairs,'tests':tests,
        'planned_episodes':72,'planned_pairs':36,'episodes_per_task_per_arm':12,
        'config_sha256':digest(ROOT/'experiment_config.json'),'control_files':hashes,
        'frozen_experience':gates,'new_warmup_episodes':0,'new_learning_api_calls':0,
        'lanes':config['lanes'],
        'analysis_plan':{
            'primary_endpoint':'Official success-rate difference doctor minus baseline over all 36 paired cases, equal task weights',
            'primary_test':'Two-sided exact McNemar/binomial test on discordant pairs',
            'primary_interval':'95% percentile paired cluster bootstrap, layouts resampled within each task, all3 official seeds kept in each layout cluster',
            'bootstrap_repetitions':10000,'bootstrap_rng_seed':20261101,
            'overall_positive_evidence_rule':'Exact paired p<0.05 AND layout-cluster95% success-difference lower bound>0; report both even if they disagree',
            'cluster_caution':'Only four held-out layouts per task; report wide/degenerate intervals transparently',
            'secondary_endpoints':['Official score difference','All-attempt control and total wall time','Reported input/output/cache/reasoning tokens','Joint-success paired time/token comparison','EWAM/primitive action counts','Lane and API error/wait profiles'],
            'per_task_tests':'Exact paired success tests, Holm adjustment across three tasks; secondary/exploratory',
            'interim_inference':'Counts only; inferential results withheld until all72 fixed episodes finish',
            'early_stopping':'None based on outcomes; preserve valid failures, stop only for infrastructure inspection or user steering',
            'pilot_pooling':'Pilot12 test episodes excluded from primary statistics',
            'test_learning':'Frozen prior warmup guidance only; no test feedback into experience or controller',
            'timing_caution':'Paired concurrent execution with crossed GPU lanes; shared compute/network effects remain, time is secondary'} }
