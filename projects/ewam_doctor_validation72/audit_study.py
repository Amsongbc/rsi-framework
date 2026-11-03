"""Verify recorded native scores, action plans, proposal gates and test isolation."""

# Export portability change: resolve external projects via PROJECTS_ROOT.
import os as _export_os
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def main():
    output=[];invalid=[]
    for p in sorted((ROOT/'runs').glob('*/episode_summary.json')):
        s=json.loads(p.read_text());base=p.parent
        if not s['valid']:invalid.append(str(p));continue
        config=json.loads((base/'experiment_config.json').read_text())
        a=lines(base/'native_actions.jsonl');n=s['native_actions_issued']
        checks={'frame_action_count':len(a)==n+1,
            'contiguous_frame_indices':all(row['frame_index']==i for i,row in enumerate(a)),
            'native_budget':n<=s['native_limit'],
            'owner_counts':all(sum(row['owner']==owner for row in a[1:])==count
                               for owner,count in s['native_actions_by_owner'].items())}
        official=next(iter(s['client_results'][0]['value']['details'].values()))
        checks['official_score_matches']=s['official_score']==official['score']
        checks['official_success_matches']=s['success']==bool(official['success'])
        videos=[]
        for v in sorted((base/'native_video').glob('*.mp4')):
            cap=cv2.VideoCapture(str(v));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            first,_=cap.read();cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,count-1));last,_=cap.read();cap.release()
            videos.append(count==n+1 and first and last)
        checks['three_readable_aligned_videos']=len(videos)==3 and all(videos)
        if s['doctor_enabled']:
            plans={r['prediction_id']:r for r in lines(base/'ewam_plans.jsonl')}
            seen={r['prediction_id']:r['time'] for r in lines(base/'doctor_events.jsonl')
                  if r['event']=='proposal_and_rgb_delivered_to_model'}
            releases=lines(base/'ewam_calls.jsonl');valid=True
            for event in releases:
                plan=plans[event['prediction_id']];start=plan['native_action_index'];count=event['native_actions']
                valid=valid and event['prediction_id'] in seen
                if count:
                    actual=a[start+1:start+1+count]
                    valid=valid and len(actual)==count and seen.get(event['prediction_id'],float('inf'))<=actual[0]['time']
                    for row,target in zip(actual,plan['result']['actions'][:count]):
                        valid=valid and row['owner']=='ewam'
                        valid=valid and all(np.allclose(row['action'][key],value,rtol=0,atol=1e-8) for key,value in target.items())
                valid=valid and event['discarded_native_actions']==len(plan['result']['actions'])-count
            checks['proposal_review_and_unchanged_contiguous_prefixes']=bool(valid)
            checks['release_count_matches']=len(releases)==s['ewam_attempts']
            if s['doctor_phase']=='direct':
                checks['direct_branch_no_ewam']=s['ewam_calls']==0 and s['native_actions_by_owner']['ewam']==0
                checks['direct_branch_no_rsi_memory']=config['doctor']['frozen_memory'] is None
            elif config['study_case']['arm']=='doctor':
                frozen=ROOT/'harness'/s['task']/'frozen_experience.json'
                checks['test_memory_frozen']=json.loads(frozen.read_text())==config['doctor']['frozen_memory']
        else:
            checks['baseline_original_ewam_first']=s['first_native_action_owner']=='ewam'
        output.append({'episode':base.name,'checks':checks})
    declaration=json.loads((ROOT/'study_declaration.json').read_text())
    frozen={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value
            for name,value in declaration['control_files'].items()}
    report={'valid_episodes_checked':len(output),'invalid_attempts_preserved':invalid,
            'frozen_controller_hashes':frozen,
            'all_pass':all(frozen.values()) and all(all(r['checks'].values()) for r in output),
            'episodes':output}
    (ROOT/'integrity_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='episodes'}))
    if not report['all_pass']:raise SystemExit('Integrity audit failed; inspect integrity_audit.json')


if __name__=='__main__':main()
