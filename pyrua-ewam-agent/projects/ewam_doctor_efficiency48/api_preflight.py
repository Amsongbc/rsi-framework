"""Non-benchmark API contract probe; never exposes the private credential."""

# Export portability change: resolve external projects via PROJECTS_ROOT.
import os as _export_os
import json
import random
from pathlib import Path

from api_transport import ResponsesTransport

ROOT=Path(__file__).resolve().parent

def main():
    api=json.loads((ROOT/'experiment_config.json').read_text())['api']
    api.update(context_compaction=True,compact_threshold=32768,max_output_tokens=2048)
    out=ROOT/'preflight'/'tool_context_probe'
    if out.exists():raise RuntimeError('Preserve previous preflight evidence')
    transport=ResponsesTransport(api,out)
    tool={'type':'function','function':{'name':'probe','description':'Return the probe marker.',
        'parameters':{'type':'object','properties':{'marker':{'type':'string'}},'required':['marker'],'additionalProperties':False}}}
    rng=random.Random(20261102)
    padding=' '.join('probe%08x'%rng.getrandbits(32) for _ in range(9000))
    messages=[{'role':'system','content':'You are testing API conversation compaction. '
        'Follow each new instruction. The probe tool produces irrelevant synthetic diagnostic padding.'},
        {'role':'user','content':'Call probe with marker READY.'}]
    body={'model':api['model'],'messages':messages,'tools':[tool],'tool_choice':'auto'}
    try:
        code,data=transport('',body,600)
        if code!=200:
            raise RuntimeError('API probe status '+str(code)+': '+json.dumps(data)[:1600])
        calls=data['choices'][0]['message']['tool_calls']
        if not calls:raise RuntimeError('Probe did not return the required function call')
        messages.append(data['choices'][0]['message'])
        messages.extend({'role':'tool','tool_call_id':c['id'],'content':'READY acknowledged. Irrelevant diagnostics, discard after reading: '+padding} for c in calls)
        messages.append({'role':'user','content':'Ignore all diagnostic padding. Call probe with marker SECOND.'})
        code2,data2=transport('',body,600)
        if code2!=200:raise RuntimeError('Continuation status '+str(code2)+': '+json.dumps(data2)[:1600])
        calls2=data2['choices'][0]['message']['tool_calls']
        if not calls2:raise RuntimeError('Second probe did not return a tool call')
        messages.append(data2['choices'][0]['message'])
        messages.extend({'role':'tool','tool_call_id':c['id'],'content':'SECOND complete.'} for c in calls2)
        messages.append({'role':'user','content':'Probes finished. Reply only DONE, without tools.'})
        code3,data3=transport('',body,600)
        if code3!=200:raise RuntimeError('Final continuation failed: '+str(code3))
        if data3['usage']['prompt_tokens']>=data2['usage']['prompt_tokens']*.5:
            raise RuntimeError('Actual followup input tokens did not shrink after server compaction')
        report={'benchmark_episode':False,'model':api['model'],'reasoning_effort':api['reasoning_effort'],
            'statuses':[code,code2,code3],'compactions':transport.compactions,'function_continuation_passed':True,
            'usage_first':data['usage'],'usage_second':data2['usage'],
            'usage_third':data3['usage'],'final_text':data3['choices'][0]['message']['content']}
        if transport.compactions<1:raise RuntimeError('Server accepted config but no actual compaction was observed')
        (ROOT/'api_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    finally:transport.close()

if __name__=='__main__':main()
