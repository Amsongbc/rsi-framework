"""Predeclared new-layout2x2 efficiency ablation with balanced two-lane waves."""

# Export portability change: resolve external projects via PROJECTS_ROOT.
import os as _export_os
import hashlib
import itertools
import json
from pathlib import Path
import random
from collections import Counter

ROOT=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(config):
    arms=list(config['arms']);assert arms==['original','sparse','compact','combined']
    bases={0:['original','sparse','compact','combined'],
           1:['original','compact','sparse','combined'],
           2:['original','sparse','combined','compact']}
    blocks=[]
    for li,layout in enumerate(config['layouts']):
        for seed in config['eval_seeds']:
            base=bases[seed];order=base[li:]+base[:li]
            blocks.append({'task':'classify_objects','seed':seed,'layout':layout,'arm_order':order})
    random.Random(20261102).shuffle(blocks)
    pairs=[];tests=[]
    for bi,block in enumerate(blocks,1):
        block['case_id']=bi
        for wave in range(2):
            pid=len(pairs)+1
            names=block['arm_order'][wave*2:wave*2+2]
            pairs.append({k:block[k] for k in ['task','seed','layout','case_id']}|{'pair_id':pid,'wave':wave+1,'arms':names})
            for lane,arm in zip(['A','B'],names):
                tests.append({k:block[k] for k in ['task','seed','layout','case_id']}|
                    {'pair_id':pid,'wave':wave+1,'arm':arm,'lane':lane})
    assert len(blocks)==12 and len(pairs)==24 and len(tests)==48
    assert len({(t['seed'],t['layout'],t['arm']) for t in tests})==48
    for arm in arms:
        rows=[t for t in tests if t['arm']==arm];assert len(rows)==12
        for field in ['lane','wave']:
            assert set(Counter(t[field] for t in rows).values())=={6}
            for seed in config['eval_seeds']:
                assert set(Counter(t[field] for t in rows if t['seed']==seed).values())=={2}
    copairs=Counter(tuple(sorted(p['arms'])) for p in pairs)
    assert len(copairs)==6 and set(copairs.values())=={4}
    files={p.name:digest(p) for p in sorted(ROOT.glob('*.py'))}
    memory=ROOT/'harness/classify_objects'
    return {'cohort_id':config['cohort_id'],'planned_episodes':48,'planned_pairs':24,'planned_case_blocks':12,
        'blocks':blocks,'pairs':pairs,'tests':tests,'arms':config['arms'],'episodes_per_arm':12,
        'config_sha256':digest(ROOT/'experiment_config.json'),'control_files':files,
        'frozen_experience':{'classify_objects':{'experience_sha256':digest(memory/'frozen_experience.json'),
            'gate_sha256':digest(memory/'gate.json')}},'lanes':config['lanes'],
        'analysis_plan':{'primary_comparison':'combined vs original on12 matched new cases',
            'primary_resource_endpoints':['Total reported tokens including server compaction usage','Control wall time'],
            'paired_confidence_intervals':'95% paired layout-cluster percentile bootstrap,4 layouts, all3 seeds and4arms retained per cluster',
            'bootstrap_repetitions':10000,'bootstrap_rng_seed':20261102,
            'secondary_contrasts':['sparse vs original','compact vs original','sparse main effect averaged across compaction levels','compaction main effect averaged across sparse levels','factor interaction'],
            'success_scoring':'Official success and partial score, exact paired discordance p; Holm over3 arm-vs-original comparisons',
            'accuracy_limit':'12cases cannot establish formal noninferiority; report lost/won successes, do not equate nonsignificance with equivalent accuracy',
            'candidate_screen':'Both primary resource95%upper changes<0 AND no observed success lost to original; screening only, not proof of accuracy preservation',
            'interim_inference':'Counts only until all48 finish','early_stopping':'No outcome-based early stop or valid-case retries',
            'holdout':'Layouts6-9 excluded from all previous warmup/pilot/72episode tests; prior results used as development evidence only',
            'freeze':'No controller/experience/config tuning after declaration; no test feedback or online learning',
            'time_caution':'Crossed two-lane concurrency and balanced waves; network/shared CPU effects remain',
            'component_limit':'Sparse factor replaces GPT FK-preview review with deterministic robot-only FK/finite checks and retains fresh visual gate after motion; it is an ablation of supervision mode, not identical safety semantics'}}


if __name__=='__main__':
    c=json.loads((ROOT/'experiment_config.json').read_text());d=build(c)
    (ROOT/'study_declaration.json').write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'episodes':len(d['tests']),'blocks':len(d['blocks']),'paired_waves':len(d['pairs']),'balance':'passed'}))
