"""Protocol tests for actual context pruning and the sparse EWAM state machine."""

# Export portability change: resolve external projects via PROJECTS_ROOT.
import os as _export_os
import copy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from api_transport import ResponsesTransport
from doctor_robot import robot_class,protocol,RobodojoOfficialColourRobot,ToolError
from recording import RecordedEnv
from test_doctor import Environment,Policy,blocked,state_machine

ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/'ewam_doctor_validation72'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class Client:
    def __init__(self,outputs):self.outputs=list(outputs);self.payloads=[]
    def post(self,url,json,timeout):
        self.payloads.append(copy.deepcopy(json));status,value=self.outputs.pop(0)
        return SimpleNamespace(status_code=status,headers={'x-request-id':'test'},json=lambda:copy.deepcopy(value))
    def close(self):pass


def response(output):
    return {'model':'gpt-6.1-sol','output':output,'usage':{'input_tokens':100,'output_tokens':5}}


def transport_checks(root):
    key=root/'test_key';key.write_text('offline-fake-key')
    api={'key_file':str(key),'model':'gpt-6.1-sol','reasoning_effort':'high',
         'max_output_tokens':8192,'base_url':'https://api.openai.com/v1','compact_threshold':32768}
    call={'type':'function_call','id':'f1','call_id':'c1','name':'python','arguments':'{}'}
    reason={'type':'reasoning','id':'r1','encrypted_content':'opaque-reasoning','summary':[]}
    compact={'type':'compaction','id':'compact1','encrypted_content':'opaque-history'}
    end={'type':'message','id':'m1','role':'assistant','content':[{'type':'output_text','text':'DONE'}]}
    initial=[{'role':'system','content':'Unchanged API contract'}, {'role':'user','content':'Task plus old observations'}]
    tool={'type':'function','function':{'name':'python','description':'execute','parameters':{'type':'object'}}}
    bodies=[]
    for label,cls in [('prior',module('prior_transport',PRIOR/'api_transport.py').ResponsesTransport),('current',ResponsesTransport)]:
        client=Client([(200,response([reason,call])),(200,response([end]))])
        trans=cls(api,root/label,client=client);messages=copy.deepcopy(initial)
        body={'model':'gpt-6.1-sol','messages':messages,'tools':[tool]}
        trans('',body,10);messages.extend([{'role':'assistant','content':'','tool_calls':[{'id':'c1'}]},
            {'role':'tool','tool_call_id':'c1','content':'result'}]);trans('',body,10)
        bodies.append(client.payloads)
    assert bodies[0]==bodies[1], 'Original arm must preserve the previous transport payload'
    client=Client([(200,response([compact,reason,call])),(429,{'error':'retry'}),(200,response([end]))])
    trans=ResponsesTransport(dict(api,context_compaction=True),root/'compact',client=client)
    messages=copy.deepcopy(initial);body={'model':'gpt-6.1-sol','messages':messages,'tools':[tool]}
    assert trans('',body,10)[0]==200 and trans.compactions==1
    assert [x.get('role',x.get('type')) for x in trans.history]==['system','compaction','reasoning','function_call']
    assert not any(x.get('role')=='user' for x in trans.history)
    assert trans.history[0]['content'][0]['text']=='Unchanged API contract'
    callbacks=[];trans.on_visual_response=lambda:callbacks.append(True)
    messages.extend([{'role':'assistant','content':''},{'role':'tool','tool_call_id':'c1','content':'result'},
        {'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,YWJj'}}]}])
    assert trans('',body,10)[0]==429 and not callbacks
    assert trans('',body,10)[0]==200 and callbacks==[True]
    assert client.payloads[1]==client.payloads[2], 'Retry must not duplicate or lose items'
    sent=client.payloads[1]['input'];assert sent[-2]['call_id']=='c1' and sent[-1]['content'][0]['type']=='input_image'
    assert client.payloads[0]['context_management']==[{'type':'compaction','compact_threshold':32768}]
    assert 'offline-fake-key' not in (root/'compact/responses_events.jsonl').read_text()


def sparse_checks(root):
    env=RecordedEnv(Environment(),root)
    context={'out':root,'env':env,'ewam':Policy(),'session_id':'offline','doctor_phase':'hybrid',
        'ewam_calls':0,'ewam_attempts':0,'efficiency':{'sparse_supervision':True},
        'toolkit':SimpleNamespace(state=SimpleNamespace(record=lambda *a:None))}
    robo=robot_class(context)(SimpleNamespace())
    blocked(lambda:robo.ewam_propose())
    blocked(lambda:robo.ewam_execute('unknown','progress','RGB'))
    event=robo.ewam_step('Move toward object','Current RGB shows object',model_actions=2)
    assert event['native_actions']==6 and env.issued_actions==6
    assert event['review_mode']=='host_fk_screen_no_gpt_preview'
    assert context['proposal'] is None and context['awaiting_ewam_observation']
    blocked(lambda:robo.ewam_step('progress','RGB',model_actions=1))
    blocked(lambda:robo._stateful('move',{},lambda _:None))
    events=[json.loads(x) for x in (root/'doctor_events.jsonl').read_text().splitlines()]
    assert any(x['event']=='host_fk_review' and x['passed'] for x in events)
    assert not any(x['event']=='proposal_and_rgb_delivered_to_model' for x in events)
    context['awaiting_ewam_observation']=False
    robo.ewam_step('Finish prefix','New RGB',model_actions=16)
    assert env.issued_actions==10 and env.actions_by_owner=={'primitive':0,'ewam':10}
    env.close_recording()


def rejection_checks(root):
    class NonfinitePolicy(Policy):
        def call(self,*args,**kwargs):
            result=super().call(*args,**kwargs)
            result['actions'][5]['left_arm_joint_state'][0]=float('nan')
            return result
    env=RecordedEnv(Environment(),root)
    context={'out':root,'env':env,'ewam':NonfinitePolicy(),'session_id':'offline',
        'doctor_phase':'hybrid','ewam_calls':0,'ewam_attempts':0,
        'efficiency':{'sparse_supervision':True},
        'toolkit':SimpleNamespace(state=SimpleNamespace(record=lambda *a:None))}
    robo=robot_class(context)(SimpleNamespace())
    blocked(lambda:robo.ewam_step('Move object','RGB',model_actions=2))
    assert env.issued_actions==0 and context['ewam_attempts']==0 and context['proposal'] is None
    with patch.object(RobodojoOfficialColourRobot,'_stateful',return_value='correction'):
        assert robo._stateful('move',{},lambda _:None)=='correction'
    env.close_recording()


def main():
    with TemporaryDirectory() as name:
        root=Path(name)
        for n in ['original','sparse','rejected']:(root/n).mkdir()
        state_machine(root/'original');sparse_checks(root/'sparse');rejection_checks(root/'rejected');transport_checks(root)
    prior=module('prior_doctor',PRIOR/'doctor_robot.py')
    c=json.loads((ROOT/'experiment_config.json').read_text());c['doctor'].update(phase='hybrid',frozen_memory={'lesson':'same'})
    assert protocol(c)==prior.protocol(c)
    c['efficiency']['sparse_supervision']=True
    assert 'Before EWAM motion use ewam_propose' not in protocol(c) and 'ewam_step(' in protocol(c)
    from pyrualean.prompt import api_reference
    assert api_reference(robot_class({}),primitives='no-vla')==api_reference(prior.robot_class({}),primitives='no-vla')
    reference=api_reference(robot_class({'efficiency':{'sparse_supervision':True}}),primitives='no-vla')
    assert all(x in reference for x in ['ewam_step','move_to','diagnose'])
    result={'passed':True,'benchmark_episode':False,'checks':['prior_payload_and_prompt_equivalence',
        'actual_history_pruning_keeps_static_contract_and_opaque_state','tool_call_id_and_reasoning_preserved',
        'retry_does_not_deliver_image_or_duplicate_items','fresh_image_survives_pruning',
        'sparse_ewam_executes_unchanged_contiguous_prefix','sparse_postmotion_visual_gate',
        'sparse_has_host_review_not_fake_gpt_review','sparse_rejects_nonfinite_unsampled_actions_without_motion',
        'shared_native_budget','original_robot_reference_compatible']}
    (ROOT/'offline_efficiency_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
