"""Verify official scoring, native recordings, sparse review and real API pruning."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent


def lines(p):return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def main():
    results=[];invalid=[]
    declaration=json.loads((ROOT/'study_declaration.json').read_text())
    old_manifest=next((ROOT.parent/'ewam_doctor_validation72/runs').glob('*/source_manifest.json'))
    old_sources={x['path']:x['sha256'] for x in json.loads(old_manifest.read_text())
                 if '/PyRUA-Lean/src/pyrualean/' in x['path']}
    for p in sorted((ROOT/'runs').glob('*/episode_summary.json')):
        s=json.loads(p.read_text());base=p.parent
        if not s['valid']:invalid.append(str(p));continue
        cfg=json.loads((base/'experiment_config.json').read_text());eff=cfg['efficiency']
        a=lines(base/'native_actions.jsonl');n=s['native_actions_issued']
        official=next(iter(s['client_results'][0]['value']['details'].values()))
        checks={'frame_action_count':len(a)==n+1,'frame_indices':all(r['frame_index']==i for i,r in enumerate(a)),
            'native_budget':n<=s['native_limit'],'official_score':s['official_score']==official['score'],
            'official_success':s['success']==bool(official['success']),
            'owner_counts':all(sum(r['owner']==owner for r in a[1:])==count for owner,count in s['native_actions_by_owner'].items()),
            'variant_flags_match':all(s['efficiency'][k]==eff[k] for k in ['sparse_supervision','context_compaction']),
            'frozen_memory':cfg['doctor']['frozen_memory']==json.loads((ROOT/'harness/classify_objects/frozen_experience.json').read_text())}
        sources={x['path']:x['sha256'] for x in json.loads((base/'source_manifest.json').read_text()) if x['path'] in old_sources}
        checks['original_pyrua_library_unchanged']=sources==old_sources
        videos=[]
        for video in sorted((base/'native_video').glob('*.mp4')):
            cap=cv2.VideoCapture(str(video));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            first,_=cap.read();cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,count-1));last,_=cap.read();cap.release()
            videos.append(count==n+1 and first and last)
        checks['three_aligned_readable_videos']=len(videos)==3 and all(videos)
        plans={x['prediction_id']:x for x in lines(base/'ewam_plans.jsonl')}
        events=lines(base/'doctor_events.jsonl');releases=lines(base/'ewam_calls.jsonl')
        delivered={x['prediction_id']:x for x in events if x['event']=='proposal_and_rgb_delivered_to_model'}
        host={x['prediction_id']:x for x in events if x['event']=='host_fk_review'}
        visual=lines(base/'protocol_events.jsonl');prefix_ok=True;gate_ok=True
        for release in releases:
            ident=release['prediction_id'];plan=plans[ident];start=plan['native_action_index'];count=release['native_actions']
            rows=a[start+1:start+count+1]
            prefix_ok=prefix_ok and len(rows)==count and release['discarded_native_actions']==len(plan['result']['actions'])-count
            if eff['sparse_supervision']:
                review=host.get(ident,{})
                prefix_ok=prefix_ok and review.get('passed',False) and review.get('finite_actions',False)
                prefix_ok=prefix_ok and review.get('native_action_index')==start
                prefix_ok=prefix_ok and plan['preview']['fk_position_check_passed']
                prefix_ok=prefix_ok and release.get('review_mode')=='host_fk_screen_no_gpt_preview'
                prefix_ok=prefix_ok and all(np.isfinite(np.asarray(v,float)).all() for row in plan['result']['actions'] for v in row.values())
            else:review=delivered.get(ident,{});prefix_ok=prefix_ok and bool(review)
            if count:prefix_ok=prefix_ok and review.get('time',float('inf'))<=rows[0]['time']
            for row,target in zip(rows,plan['result']['actions'][:count]):
                prefix_ok=prefix_ok and row['owner']=='ewam' and all(np.allclose(row['action'][k],v,rtol=0,atol=1e-8) for k,v in target.items())
            end=start+count
            if count and end<n:
                gate_ok=gate_ok and any(x.get('after_ewam') and x.get('native_actions')==end and
                    rows[-1]['time']<=x['time']<=a[end+1]['time'] for x in visual)
        checks['review_and_unchanged_contiguous_prefixes']=bool(prefix_ok)
        checks['every_ewam_prefix_followed_by_visual_before_next_motion']=bool(gate_ok)
        checks['executed_prefix_count']=len(releases)==s['ewam_attempts']
        if eff['sparse_supervision']:checks['no_fake_gpt_preview_event']=not delivered
        responses=lines(base/'responses_events.jsonl');by_request={e['request']:e for e in responses}
        requests={int(q.stem):json.loads(q.read_text()) for q in (base/'api_requests').glob('*.json')}
        comp=lines(base/'compaction_events.jsonl')
        checks['compaction_counter']=len(comp)==s.get('context_compactions',0)
        checks['api_compaction_mode']=all(('context_management' in request)==eff['context_compaction'] for request in requests.values())
        prune_ok=True
        for e in responses:
            output=(e.get('response') or {}).get('output',[])
            indices=[i for i,x in enumerate(output) if x.get('type')=='compaction']
            if not eff['context_compaction'] or e.get('status')!=200 or not indices:continue
            following=requests.get(e['request']+1)
            if following is None:continue
            static=[x for x in requests[1]['input'] if x.get('role') in ['system','developer']]
            expected=static+output[indices[-1]:]
            prune_ok=prune_ok and following['input'][:len(expected)]==expected
            prune_ok=prune_ok and following['input'][len(static)]['type']=='compaction'
        checks['actual_api_history_pruning_preserves_opaque_state_and_tool_chain']=bool(prune_ok)
        if not eff['context_compaction']:checks['uncompressed_arm_never_compacted']=not comp
        results.append({'episode':base.name,'checks':checks})
    frozen={k:hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in declaration['control_files'].items()}
    report={'valid_episodes_checked':len(results),'invalid_attempts_preserved':invalid,
        'frozen_controller_hashes':frozen,'all_pass':all(frozen.values()) and all(all(r['checks'].values()) for r in results),'episodes':results}
    (ROOT/'integrity_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='episodes'}))
    if not report['all_pass']:raise SystemExit('Integrity audit failed')


if __name__=='__main__':main()
