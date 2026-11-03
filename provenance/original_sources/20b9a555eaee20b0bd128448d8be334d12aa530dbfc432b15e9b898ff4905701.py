"""Descriptive paired results and full warmup/adaptation accounting."""
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

