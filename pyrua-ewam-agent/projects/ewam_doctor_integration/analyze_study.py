"""Descriptive paired results and full warmup/adaptation accounting."""

# Export portability change: resolve external projects via PROJECTS_ROOT.
import os as _export_os
import csv
import hashlib
import json
from pathlib import Path
import statistics as stats

ROOT = Path(__file__).resolve().parent


def usage(paths):
    totals = dict(input_tokens=0,cached_input_tokens=0,output_tokens=0,reasoning_output_tokens=0,
                  successful_responses=0,api_wait_s=0)
    for path in paths:
        if not path.exists():continue
        for line in path.read_text().splitlines():
            e=json.loads(line);totals['api_wait_s']+=e.get('latency_s',0)
            if e.get('status')!=200:continue
            u=(e.get('response') or {}).get('usage') or {}
            totals['successful_responses']+=1
            totals['input_tokens']+=u.get('input_tokens',0);totals['output_tokens']+=u.get('output_tokens',0)
            totals['cached_input_tokens']+=(u.get('input_tokens_details') or {}).get('cached_tokens',0)
            totals['reasoning_output_tokens']+=(u.get('output_tokens_details') or {}).get('reasoning_tokens',0)
    totals['total_tokens']=totals['input_tokens']+totals['output_tokens']
    totals['uncached_input_tokens']=totals['input_tokens']-totals['cached_input_tokens']
    return totals


def aggregate(rows):
    if not rows:return {'n':0}
    return {'n':len(rows),'successes':sum(r['success'] for r in rows),
            'success_rate_percent':100*stats.mean(r['success'] for r in rows),
            'mean_score_percent':100*stats.mean(r['official_score'] for r in rows),
            'mean_control_min':stats.mean(r['control_wall_s'] for r in rows)/60,
            'control_total_min':sum(r['control_wall_s'] for r in rows)/60,
            'total_wall_min':sum(r['total_wall_s'] for r in rows)/60,
            'mean_native_actions':stats.mean(r['native_actions_issued'] for r in rows),
            'ewam_inferences':sum(r['ewam_calls'] for r in rows),
            'ewam_executed_prefixes':sum(r['ewam_attempts'] for r in rows),
            'native_actions_by_owner':{k:sum(r['native_actions_by_owner'][k] for r in rows) for k in ['ewam','primitive']},
            'api':usage([Path(r['summary_path']).parent/'responses_events.jsonl' for r in rows])}


def main():
    declaration=json.loads((ROOT/'study_declaration.json').read_text())
    rows=[];invalid=[]
    for path in sorted((ROOT/'runs').glob('*/episode_summary.json')):
        r=json.loads(path.read_text());r['summary_path']=str(path)
        case=json.loads((path.parent/'experiment_config.json').read_text())['study_case'];r['arm']=case['arm']
        if r['valid']:rows.append(r)
        else:invalid.append(r)
    selected={(r['task'],r['eval_seed'],r['layout'],r['arm']):r for r in rows}
    if len(selected)!=len(rows):raise ValueError('Duplicate valid episode; no outcome selection allowed')
    aggregate_rows={arm:aggregate([r for r in rows if r['arm']==arm]) for arm in ['warmup','baseline','doctor']}
    pairs=[]
    for case in declaration['tests']:
        if case['arm']!='baseline':continue
        key=(case['task'],case['seed'],0)
        a=selected.get((*key,'baseline'));b=selected.get((*key,'doctor'))
        if not a or not b:continue
        paths=[Path(r['summary_path']).parent for r in [a,b]]
        initial=[json.loads((p/'native_actions.jsonl').read_text().splitlines()[0]) for p in paths]
        pairs.append({'task':key[0],'eval_seed':key[1],
            'baseline_success':a['success'],'doctor_success':b['success'],
            'baseline_score':a['official_score'],'doctor_score':b['official_score'],
            'baseline_control_min':a['control_wall_s']/60,'doctor_control_min':b['control_wall_s']/60,
            'baseline_tokens':usage([paths[0]/'responses_events.jsonl'])['total_tokens'],
            'doctor_tokens':usage([paths[1]/'responses_events.jsonl'])['total_tokens'],
            'instruction_matches':initial[0]['instruction']==initial[1]['instruction'],
            'initial_robot_state_matches':initial[0]['robot_state']==initial[1]['robot_state'],
            'native_budget_matches':a['native_limit']==b['native_limit']})
    adaptation_paths=list((ROOT/'learning').glob('*/api/responses_events.jsonl'))+list((ROOT/'harness').glob('*/api/responses_events.jsonl'))
    adaptation=usage(adaptation_paths)
    adaptation['wall_s']=sum(json.loads(p.read_text())['wall_s'] for p in
        list((ROOT/'learning').glob('*/api/offline_call_timing.json'))+list((ROOT/'harness').glob('*/api/offline_call_timing.json')))
    gates=[]
    for p in (ROOT/'harness').glob('*/gate.json'):
        g=json.loads(p.read_text());g['frozen_memory_hash_unchanged']=hashlib.sha256((p.parent/'frozen_experience.json').read_bytes()).hexdigest()==g['sha256'];gates.append(g)
    complete=all((c['task'],c['seed'],c['layout'],c['arm']) in selected for c in declaration['warmups']+declaration['tests'])
    per=[]
    for task in json.loads((ROOT/'experiment_config.json').read_text())['tasks']:
        for arm in ['warmup','baseline','doctor']:
            per.append(dict(task=task,arm=arm,**aggregate([r for r in rows if r['task']==task and r['arm']==arm])))
    warm=aggregate_rows['warmup'];test=aggregate_rows['doctor']
    preparation={'warmup_plus_learning_wall_min':warm.get('total_wall_min',0)+adaptation['wall_s']/60,
                 'warmup_plus_learning_tokens':warm.get('api',{}).get('total_tokens',0)+adaptation['total_tokens']}
    all_in={'doctor_test_plus_preparation_wall_min':test.get('total_wall_min',0)+preparation['warmup_plus_learning_wall_min'],
            'doctor_test_plus_preparation_tokens':test.get('api',{}).get('total_tokens',0)+preparation['warmup_plus_learning_tokens']}
    report={'complete':complete,'planned_episodes':21,'valid':len(rows),'invalid_attempts':invalid,
            'aggregates':aggregate_rows,'offline_learning':adaptation,'preparation':preparation,'all_in_doctor':all_in,
            'gates':gates,'per_task':per,'paired_tests':pairs,'episodes':rows,
            'limits':['Exploratory 3-task, 2-test-seed-per-arm study; no significance or full-benchmark claim.',
                'Bundled proposal supervision, warmup gate and memory; not a component ablation.',
                'Warmup uses layout1 only; test layout0 has frozen experience and no outcome feedback.',
                'Gate below 30 selects a fresh direct-primitives branch without RSI memory, as in RoboDoctor.',
                'PyRUA motion primitives are retained; RoboDoctor 1-5-step correction restriction is not imported.',
                'Original RGB-only observation contract is retained; FK is robot-only, not object truth.',
                'All warmup and summary/harness API costs are included separately; token usage is not a monetary invoice.',
                'Same official API, model/effort, GPUs and budgets in fresh paired arms; shared GPU activity can still vary.']}
    (ROOT/'study_results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    if pairs:
        with (ROOT/'paired_test_metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    if complete:
        lines=['RoboDoctor机制迁移：小规模配对实验',
               '模型 gpt-6.1-sol/high；3任务；预热 layout1 seeds0/1/2；测试 layout0 seeds0/1。',
               '控制组：原固定EWAM先行框架；实验组：提案检查/前缀放行/预热路由/冻结经验。',
               '完整统计见 study_results.json；逐样本见 paired_test_metrics.csv。','',
               json.dumps(aggregate_rows,indent=2,ensure_ascii=False),'预热与总结成本：',
               json.dumps(preparation,indent=2),'实验组连同准备成本：',json.dumps(all_in,indent=2),
               '门控结果：',json.dumps(gates,indent=2),'解释限制：',*report['limits']]
        (ROOT/'study_report.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'complete':complete,'valid':len(rows),'paired_tests':len(pairs),
                      'baseline_successes':aggregate_rows['baseline'].get('successes'),
                      'doctor_successes':aggregate_rows['doctor'].get('successes')}),flush=True)


if __name__=='__main__':main()
