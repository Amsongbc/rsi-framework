"""Fixed-sample paired analysis; inferential outputs only after all72 episodes."""

# Export portability change: resolve external projects via PROJECTS_ROOT.
import os as _export_os
from collections import Counter
import csv
import json
import math
from pathlib import Path

import numpy as np

from metrics import aggregate,usage
from study_plan import ROOT,digest


def exact_p(wins,losses):
    n=wins+losses
    return min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.


def wilson(successes,n):
    if not n:return None
    z=1.959963984540054;p=successes/n;den=1+z*z/n
    middle=(p+z*z/(2*n))/den
    width=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [0. if successes==0 else 100*max(0,middle-width),
            100. if successes==n else 100*min(1,middle+width)]


def paired_stats(pairs):
    wins=sum(p['doctor_success'] and not p['baseline_success'] for p in pairs)
    losses=sum(p['baseline_success'] and not p['doctor_success'] for p in pairs)
    return {'pairs':len(pairs),'doctor_only_successes':wins,'baseline_only_successes':losses,
        'both_successes':sum(p['baseline_success'] and p['doctor_success'] for p in pairs),
        'both_failures':sum(not p['baseline_success'] and not p['doctor_success'] for p in pairs),
        'success_difference_percentage_points':100*(wins-losses)/len(pairs) if pairs else None,
        'exact_mcnemar_two_sided_p':exact_p(wins,losses)}


def cluster_intervals(pairs,tasks,layouts,repetitions=10000,seed=20261101):
    """Resample complete layout clusters within each fixed task; keep paired seeds."""
    rng=np.random.default_rng(seed);overall={};by_task={}
    fields={'success_difference_pp':lambda p:100*(int(p['doctor_success'])-int(p['baseline_success'])),
            'score_difference_points':lambda p:100*(p['doctor_score']-p['baseline_score']),
            'control_difference_min':lambda p:p['doctor_control_min']-p['baseline_control_min']}
    draws={}
    for task in tasks:
        index=rng.integers(0,len(layouts),size=(repetitions,len(layouts)))
        values={}
        for field,fn in fields.items():
            clusters=[]
            for layout in layouts:
                rr=[p for p in pairs if p['task']==task and p['layout']==layout]
                if len(rr)!=3:raise ValueError('A layout cluster must contain all3 official seeds')
                clusters.append(np.mean([fn(p) for p in rr]))
            boot=np.asarray(clusters)[index].mean(axis=1)
            values[field]=np.quantile(boot,[.025,.975]).tolist()
            draws.setdefault(field,[]).append(boot)
        by_task[task]=values
    for field,parts in draws.items():overall[field]=np.quantile(np.mean(parts,axis=0),[.025,.975]).tolist()
    return {'method':'Paired task-stratified layout-cluster percentile bootstrap; all3 seeds kept in cluster',
            'repetitions':repetitions,'rng_seed':seed,'overall_95_percent':overall,'per_task_95_percent':by_task}


def change(new,old):return 100*(new/old-1) if old else None


def main():
    config=json.loads((ROOT/'experiment_config.json').read_text())
    declaration=json.loads((ROOT/'study_declaration.json').read_text())
    expected={(c['task'],c['seed'],c['layout'],c['arm']):c for c in declaration['tests']}
    selected={};invalid=[]
    for path in sorted((ROOT/'runs').glob('*/episode_summary.json')):
        row=json.loads(path.read_text());cfg=json.loads((path.parent/'experiment_config.json').read_text())
        case=cfg['study_case'];key=(row['task'],row['eval_seed'],row['layout'],case['arm'])
        if key not in expected:raise ValueError('Undeclared episode found: '+str(path))
        row.update(arm=case['arm'],lane=case['lane'],pair_id=case['pair_id'],summary_path=str(path))
        row['http_status_counts']=dict(Counter(str(x.get('status')) for x in
            (json.loads(line) for line in (path.parent/'responses_events.jsonl').read_text().splitlines()))) if (path.parent/'responses_events.jsonl').exists() else {}
        if not row['valid']:invalid.append(row);continue
        if key in selected:raise ValueError('Duplicate valid outcome; selection forbidden')
        selected[key]=row
    rows=list(selected.values());groups={arm:aggregate([r for r in rows if r['arm']==arm]) for arm in ['baseline','doctor']}
    per_task=[]
    for task in config['tasks']:
        for arm in ['baseline','doctor']:per_task.append(dict(task=task,arm=arm,**aggregate([r for r in rows if r['task']==task and r['arm']==arm])))
    pairs=[]
    for case in declaration['pairs']:
        key=(case['task'],case['seed'],case['layout'])
        a=selected.get((*key,'baseline'));b=selected.get((*key,'doctor'))
        if a is None or b is None:continue
        dirs=[Path(x['summary_path']).parent for x in [a,b]]
        initial=[json.loads((p/'native_actions.jsonl').read_text().splitlines()[0]) for p in dirs]
        au=usage([dirs[0]/'responses_events.jsonl']);bu=usage([dirs[1]/'responses_events.jsonl'])
        pairs.append({'pair_id':case['pair_id'],'task':case['task'],'eval_seed':case['seed'],'layout':case['layout'],
            'baseline_lane':a['lane'],'doctor_lane':b['lane'],
            'baseline_success':a['success'],'doctor_success':b['success'],
            'baseline_score':a['official_score'],'doctor_score':b['official_score'],
            'baseline_control_min':a['control_wall_s']/60,'doctor_control_min':b['control_wall_s']/60,
            'baseline_total_min':a['total_wall_s']/60,'doctor_total_min':b['total_wall_s']/60,
            'baseline_tokens':au['total_tokens'],'doctor_tokens':bu['total_tokens'],
            'baseline_native_actions':a['native_actions_issued'],'doctor_native_actions':b['native_actions_issued'],
            'baseline_ewam_actions':a['native_actions_by_owner']['ewam'],'doctor_ewam_actions':b['native_actions_by_owner']['ewam'],
            'instruction_matches':initial[0]['instruction']==initial[1]['instruction'],
            'initial_robot_state_matches_exactly':initial[0]['robot_state']==initial[1]['robot_state'],
            'native_budget_matches':a['native_limit']==b['native_limit']})
    complete=len(selected)==len(expected)==72 and len(pairs)==36
    gates=[]
    for task in config['tasks']:
        path=ROOT/'harness'/task/'gate.json';g=json.loads(path.read_text())
        g['frozen_memory_hash_unchanged']=digest(path.parent/'frozen_experience.json')==g['sha256'];gates.append(g)
    report={'cohort_id':config['cohort_id'],'complete':complete,'valid':len(rows),'planned_episodes':72,
        'completed_pairs':len(pairs),'planned_pairs':36,'new_warmup_episodes':0,'new_learning_api_calls':0,
        'aggregates':groups,'per_task':per_task,'paired_tests':pairs,'gates':gates,'episodes':rows,'invalid_attempts':invalid,
        'lane_profiles':[dict(lane=lane,arm=arm,**aggregate([r for r in rows if r['lane']==lane and r['arm']==arm]))
                         for lane in ['A','B'] for arm in ['baseline','doctor']],
        'api_http_status_counts':{arm:dict(sum((Counter(r['http_status_counts']) for r in rows if r['arm']==arm),Counter())) for arm in ['baseline','doctor']},
        'inference_status':'final_fixed_sample' if complete else 'descriptive_interim_only_no_significance_claim',
        'limits':['Selected three-task held-out validation, not the full benchmark population.',
            'All72 predeclared episodes retained; pilot excluded from primary statistics.',
            'Prior warmup experience reused, no new training episodes or test feedback.',
            'Each arm uses each lane six times per task; paired concurrent execution may still affect wall-time through shared network/CPU resources.',
            'Within each lane SAM and simulator share one H20; EWAM runs on a separate dedicated H20.',
            'Only four layout clusters per task; cluster confidence intervals can remain wide.',
            'Whole mechanism comparison, not an ablation or a fresh pure-primitives/pure-EWAM comparison.',
            'Input includes cached tokens; output includes reasoning tokens; do not double count subcategories.',
            'Time is simulation-experiment wall time, not physical robot motion time.']}
    if complete:
        overall=paired_stats(pairs)
        overall.update(baseline_success_95_wilson=wilson(groups['baseline']['successes'],36),
                       doctor_success_95_wilson=wilson(groups['doctor']['successes'],36))
        intervals=cluster_intervals(pairs,config['tasks'],config['layouts'])
        task_stats={t:paired_stats([p for p in pairs if p['task']==t]) for t in config['tasks']}
        ordered=sorted(task_stats,key=lambda t:task_stats[t]['exact_mcnemar_two_sided_p']);previous=0
        for i,t in enumerate(ordered):
            adjusted=max(previous,min(1.,(3-i)*task_stats[t]['exact_mcnemar_two_sided_p']))
            task_stats[t]['holm_adjusted_p']=adjusted;previous=adjusted
        joint=[p for p in pairs if p['baseline_success'] and p['doctor_success']]
        joint_stats={'n':len(joint)}
        if joint:
            for arm in ['baseline','doctor']:
                joint_stats[arm+'_control_mean_min']=float(np.mean([p[arm+'_control_min'] for p in joint]))
                joint_stats[arm+'_mean_tokens']=float(np.mean([p[arm+'_tokens'] for p in joint]))
        report['statistics']={'overall':overall,'tasks':task_stats,'cluster_intervals':intervals,
            'joint_success_secondary':joint_stats,
            'favorable_under_prespecified_rule':bool(overall['exact_mcnemar_two_sided_p']<.05 and intervals['overall_95_percent']['success_difference_pp'][0]>0)}
        report['resource_change_percent']={
            'control_mean':change(groups['doctor']['mean_control_min'],groups['baseline']['mean_control_min']),
            'tokens':change(groups['doctor']['api']['total_tokens'],groups['baseline']['api']['total_tokens']),
            'native_actions':change(groups['doctor']['mean_native_actions'],groups['baseline']['mean_native_actions'])}
    (ROOT/'validation_results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    if pairs:
        with (ROOT/'paired_test_metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    table=[]
    for t in config['tasks']:
        aa=next(r for r in per_task if r['task']==t and r['arm']=='baseline');bb=next(r for r in per_task if r['task']==t and r['arm']=='doctor')
        table.append({'task':t,'baseline_n':aa['n'],'doctor_n':bb['n'],'baseline_successes':aa.get('successes'),
            'doctor_successes':bb.get('successes'),'baseline_score_percent':aa.get('mean_score_percent'),
            'doctor_score_percent':bb.get('mean_score_percent'),'baseline_control_min':aa.get('mean_control_min'),
            'doctor_control_min':bb.get('mean_control_min'),'baseline_tokens':aa.get('api',{}).get('total_tokens'),
            'doctor_tokens':bb.get('api',{}).get('total_tokens')})
    with (ROOT/'task_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    if complete:
        lines=['RoboDoctor机制扩大验证：72回合、36配对',
            '3任务，每方案每任务12回合；官方种子0/1/2；新布局2/3/4/5。',
            '复用冻结预热经验，无新增预热或测试期学习；旧试点未合并到主要统计。',
            '任务 | 旧混合成功/12 | 融合成功/12 | 旧官方均分 | 融合官方均分 | 旧/新控制分钟']
        lines += [f"{r['task']} | {r['baseline_successes']}/12 | {r['doctor_successes']}/12 | "
            f"{r['baseline_score_percent']:.2f} | {r['doctor_score_percent']:.2f} | {r['baseline_control_min']:.2f}/{r['doctor_control_min']:.2f}" for r in table]
        lines+=['','配对统计与置信区间：',json.dumps(report['statistics'],indent=2,ensure_ascii=False),
                '资源变化百分比：',json.dumps(report['resource_change_percent'],indent=2),'解释限制：',*report['limits']]
        (ROOT/'validation_report.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'complete':complete,'valid':len(rows),'pairs':len(pairs),
        'baseline_successes':groups['baseline'].get('successes'),'doctor_successes':groups['doctor'].get('successes')}),flush=True)


if __name__=='__main__':main()
