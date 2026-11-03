"""Four-arm matched2x2 analysis; no inferential output before all48 finish."""
from collections import Counter
import csv
import json
import math
from pathlib import Path
import numpy as np
from metrics import aggregate,usage
from study_plan import ROOT,digest


def save_csv(name,rows):
    if not rows:return
    with (ROOT/name).open('w',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def exact_p(wins,losses):
    n=wins+losses
    return min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.


def stats(blocks,layouts):
    arms=['original','sparse','compact','combined'];rng=np.random.default_rng(20261102)
    index=rng.integers(0,4,size=(10000,4))
    fields=['success','score','control_min','tokens','uncached_tokens','api_calls','native_actions']
    clusters={field:{arm:np.array([np.mean([b[arm][field] for b in blocks if b['layout']==layout])
              for layout in layouts]) for arm in arms} for field in fields}
    draws={f:{arm:values[index].mean(axis=1) for arm,values in aa.items()} for f,aa in clusters.items()}
    comparisons={}
    for arm in arms[1:]:
        wins=sum(b[arm]['success'] and not b['original']['success'] for b in blocks)
        losses=sum(b['original']['success'] and not b[arm]['success'] for b in blocks)
        result={'pairs':12,'wins':wins,'losses':losses,'exact_success_p':exact_p(wins,losses)}
        for field in fields:
            old=clusters[field]['original'].mean();new=clusters[field][arm].mean()
            if field in ['success','score']:
                point=100*(new-old);boot=100*(draws[field][arm]-draws[field]['original']);units='percentage_points'
            else:
                point=100*(new/old-1);boot=100*(draws[field][arm]/draws[field]['original']-1);units='percent_change_of_means'
            result[field]={'difference':float(point),'units':units,'cluster_95':np.quantile(boot,[.025,.975]).tolist()}
        both=[b for b in blocks if b[arm]['success'] and b['original']['success']]
        result['both_success_only']={'pairs':len(both)}
        if both:
            for field in ['control_min','tokens']:
                result['both_success_only'][field]={a:float(np.mean([b[a][field] for b in both])) for a in ['original',arm]}
        comparisons[arm]=result
    previous=0
    for i,arm in enumerate(sorted(comparisons,key=lambda a:comparisons[a]['exact_success_p'])):
        adjusted=max(previous,min(1.,(3-i)*comparisons[arm]['exact_success_p']))
        comparisons[arm]['holm_success_p']=adjusted;previous=adjusted
    factors={}
    for field in ['tokens','control_min','api_calls']:
        # Per-case log effects, then layout-cluster bootstrap; both levels retained.
        effects={'sparse':[],'compaction':[],'interaction':[]}
        for layout in layouts:
            subset=[b for b in blocks if b['layout']==layout]
            values={a:np.log([b[a][field] for b in subset]) for a in arms}
            effects['sparse'].append(float(np.mean(.5*(values['sparse']-values['original']+values['combined']-values['compact']))))
            effects['compaction'].append(float(np.mean(.5*(values['compact']-values['original']+values['combined']-values['sparse']))))
            effects['interaction'].append(float(np.mean(values['combined']-values['sparse']-values['compact']+values['original'])))
        factors[field]={}
        for label,value in effects.items():
            a=np.array(value);boot=100*np.expm1(a[index].mean(axis=1))
            factors[field][label]={'geometric_percent_effect':float(100*np.expm1(a.mean())),
                'cluster_95':np.quantile(boot,[.025,.975]).tolist()}
    primary=comparisons['combined']
    return {'comparisons_vs_original':comparisons,'factorial_log_resource_effects':factors,
        'bootstrap':{'repetitions':10000,'rng_seed':20261102,'clusters':4,'unit':'layout, all3seeds and4arms retained'},
        'combined_passes_resource_and_observed_success_screen':bool(primary['tokens']['cluster_95'][1]<0 and
            primary['control_min']['cluster_95'][1]<0 and primary['losses']==0),
        'accuracy_caution':'No observed losses is a screening criterion, not a powered noninferiority conclusion.'}


def main():
    cfg=json.loads((ROOT/'experiment_config.json').read_text());decl=json.loads((ROOT/'study_declaration.json').read_text())
    expected={(x['seed'],x['layout'],x['arm']):x for x in decl['tests']};selected={};invalid=[]
    for p in sorted((ROOT/'runs').glob('*/episode_summary.json')):
        try:row=json.loads(p.read_text())
        except json.JSONDecodeError:continue
        case=json.loads((p.parent/'experiment_config.json').read_text())['study_case']
        key=(row['eval_seed'],row['layout'],case['arm']);assert key in expected
        row.update(arm=case['arm'],lane=case['lane'],wave=case['wave'],case_id=case['case_id'],summary_path=str(p))
        if not row['valid']:invalid.append(row);continue
        if key in selected:raise ValueError('Duplicate valid outcome; selection prohibited')
        selected[key]=row
    rows=list(selected.values());groups={a:aggregate([r for r in rows if r['arm']==a]) for a in cfg['arms']}
    arm_table=[]
    for arm in cfg['arms']:
        rr=[r for r in rows if r['arm']==arm];g=groups[arm]
        if not rr:continue
        g['context_compactions']=sum(r.get('context_compactions',0) for r in rr)
        arm_table.append({'arm':arm,'n':g['n'],'successes':g['successes'],'score':g['mean_score_percent'],
            'mean_control_min':g['mean_control_min'],'mean_total_min':g['total_wall_min']/g['n'],
            'total_tokens':g['api']['total_tokens'],'uncached_input_tokens':g['api']['uncached_input_tokens'],
            'output_tokens':g['api']['output_tokens'],'api_calls':g['api']['successful_responses'],
            'ewam_inferences':g['ewam_inferences'],'compactions':g['context_compactions']})
    blocks=[];flat=[]
    for b in decl['blocks']:
        entries={a:selected.get((b['seed'],b['layout'],a)) for a in cfg['arms']}
        if not all(entries.values()):continue
        block={k:b[k] for k in ['task','seed','layout','case_id']};initials=[]
        for arm,row in entries.items():
            out=Path(row['summary_path']).parent
            first=json.loads((out/'native_actions.jsonl').read_text().splitlines()[0]);initials.append(first)
            u=usage([out/'responses_events.jsonl'])
            metrics={'success':row['success'],'score':row['official_score'],'control_min':row['control_wall_s']/60,
                'total_min':row['total_wall_s']/60,'tokens':u['total_tokens'],'uncached_tokens':u['uncached_input_tokens'],
                'api_calls':u['successful_responses'],'native_actions':row['native_actions_issued'],
                'ewam_calls':row['ewam_calls'],'compactions':row.get('context_compactions',0)}
            block[arm]=metrics;flat.append({k:block[k] for k in ['seed','layout','case_id']}|{'arm':arm,'lane':row['lane'],'wave':row['wave']}|metrics)
        block['instruction_matches']=all(x['instruction']==initials[0]['instruction'] for x in initials)
        block['initial_robot_state_matches_exactly']=all(x['robot_state']==initials[0]['robot_state'] for x in initials)
        block['native_budget_matches']=len({e['native_limit'] for e in entries.values()})==1
        blocks.append(block)
    complete=len(selected)==48 and len(blocks)==12
    statuses={}
    for arm in cfg['arms']:
        counts=Counter()
        for row in rows:
            if row['arm']!=arm:continue
            p=Path(row['summary_path']).parent/'responses_events.jsonl'
            counts.update(str(json.loads(x).get('status')) for x in p.read_text().splitlines())
        statuses[arm]=dict(counts)
    report={'cohort_id':cfg['cohort_id'],'complete':complete,'valid':len(rows),'planned_episodes':48,
        'complete_case_blocks':len(blocks),'planned_case_blocks':12,'aggregates':groups,'arm_table':arm_table,
        'case_blocks':blocks,'episodes':rows,'invalid_attempts':invalid,'http_status_counts':statuses,
        'new_warmup_episodes':0,'new_learning_calls':0,'inference_status':'final_fixed_sample' if complete else 'counts_only_interim',
        'limits':['Classify_objects only;12cases/arm,4layout clusters; cannot infer full-benchmark or formal noninferiority.',
            'New layouts6-9 only; no pooling with previous72/pilot/ten-task cohorts.',
            'Sparse mode changes proposal supervision and API wording; original motion primitives and postmotion visual gate retained.',
            'Server compaction costs included in actual API usage; cache subsets not double-counted; token ratio is not monetary fee ratio.',
            'Wall times include shared-network/CPU effects despite lane and wave balance.',
            'All valid task failures retained; no test-time learning or tuning.']}
    if complete:report['statistics']=stats(blocks,cfg['layouts'])
    (ROOT/'efficiency_results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    save_csv('arm_comparison.csv',arm_table);save_csv('matched_case_metrics.csv',flat)
    if complete:
        text=['分类任务效率消融：48回合/12个四组匹配案例','原版、只精简提案、只压缩历史、两项同时启用；每组12回合。',
            '新布局6/7/8/9，官方种子0/1/2；gpt-6.1-sol/high；复用冻结预热经验。',
            '方案 | 成功/12 | 官方均分 | 平均控制分钟 | 总token | 未缓存输入 | API调用 | 历史压缩次数']
        text += [f"{x['arm']} | {x['successes']}/12 | {x['score']:.2f} | {x['mean_control_min']:.2f} | {x['total_tokens']} | {x['uncached_input_tokens']} | {x['api_calls']} | {x['compactions']}" for x in arm_table]
        text += ['', '匹配比较及2x2因子效应：',json.dumps(report['statistics'],indent=2,ensure_ascii=False),'限制：',*report['limits']]
        (ROOT/'efficiency_report.txt').write_text('\n'.join(text)+'\n')
    print(json.dumps({'complete':complete,'valid':len(rows),'blocks':len(blocks),
        'successes':{a:g.get('successes',0) for a,g in groups.items()}}),flush=True)


if __name__=='__main__':main()
