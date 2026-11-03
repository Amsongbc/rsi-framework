#!/usr/bin/env python3
"""Offline protocol tests with synthetic public RGB/state and no simulator/API."""
import copy
import hashlib
import json
import os
import time
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT.parent/'PyRUA-Lean/src'),str(ROOT.parent/'ewam_comparison')]
from window_session import CountingEnv,acknowledge_visual,window_robot_class,Session
from doctor_robot import preview
from hybrid_robot import RobodojoOfficialColourRobot
from pyrualean._backend import ToolError,EpisodeFinished
from pyrualean.hosts import robodojo_kinematics as kin


class FakeNative:
    def __init__(self,limit=70):
        self.n=0; self.limit=limit
        state={}
        for arm in ('left','right'):
            q=np.zeros(6); xyz=kin.fk(arm,q)[:3,3]
            state[arm+'_arm_joint_state']=q
            state[arm+'_ee_joint_state']=np.array([.5])
            state[arm+'_ee_pose']=np.r_[xyz,[0,0,0,1]]
        self.obs=dict(instruction='Move the visible block',state=state,vision={c:{'color':np.zeros((8,8,3),np.uint8)} for c in ('cam_head','cam_left_wrist','cam_right_wrist')})
    def get_obs(self): return self.obs
    def get_status(self): return dict(step=self.n,step_limit=self.limit)
    def episode_over(self): return False
    def step(self,action): self.n+=1; return self.obs,0,False,{}
    def conclude(self): pass


class FakeEwam:
    def __init__(self,native): self.native=native
    def call(self,op,**kwargs):
        assert op=='infer'
        action={k:v.copy() for k,v in self.native.obs['state'].items() if 'pose' not in k}
        return dict(actions=[copy.deepcopy(action) for _ in range(48)],inference_s=.01,instruction='Move the visible block')


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.out=Path(self.temp.name)
        native=FakeNative(); env=CountingEnv(native); env.initial_limit=native.limit
        self.ctx=dict(out=self.out,env=env,session_id='offline',ewam=FakeEwam(native),doctor_phase='hybrid',efficiency={},ewam_calls=0,ewam_attempts=0,awaiting_ewam_observation=False,
            toolkit=types.SimpleNamespace(state=types.SimpleNamespace(record=lambda *a:None)))
        self.native=native; self.env=env
        self.make_robot()
        p=self.out/'head.png'; p.write_bytes(b'offline image bytes')
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        self.latest=dict(observation_index=3,issued_actions=0,images={'cam_head':str(p)},image_sha256={'cam_head':h})
        self.ack=dict(id='ack1',observation_index=3,native_action_index=0,image_sha256={'cam_head':h},viewed_by='window_AI',evidence='Visible synthetic test scene')

    def make_robot(self,sparse=False):
        self.ctx['efficiency']['sparse_supervision']=sparse
        cls=window_robot_class(self.ctx); self.robot=cls.__new__(cls)
        self.robot._last_state=None

    def do_ack(self):
        self.latest['issued_actions']=self.env.issued_actions
        self.ack['native_action_index']=self.env.issued_actions
        return acknowledge_visual(self.ctx,self.latest,self.ack)

    def test_ack_rejects_wrong_index_native_hash_and_missing_provenance(self):
        self.ctx['awaiting_ewam_observation']=True
        for field,value in [('observation_index',2),('native_action_index',1),('image_sha256',{'cam_head':'0'*64}),('viewed_by','script'),('evidence','')]:
            bad=dict(self.ack,**{field:value})
            with self.assertRaises(ValueError): acknowledge_visual(self.ctx,self.latest,bad)
            self.assertTrue(self.ctx['awaiting_ewam_observation'])
        self.env.issued_actions=1
        with self.assertRaises(ValueError): acknowledge_visual(self.ctx,self.latest,self.ack)

    def test_ack_checks_actual_file_bytes_and_logs_provenance(self):
        self.ctx['awaiting_ewam_observation']=True
        self.do_ack(); self.assertFalse(self.ctx['awaiting_ewam_observation'])
        row=json.loads((self.out/'visual_acknowledgements.jsonl').read_text())
        self.assertEqual(row['command_id'],'ack1'); self.assertEqual(row['image_sha256'],self.ack['image_sha256'])
        Path(self.latest['images']['cam_head']).write_bytes(b'changed')
        with self.assertRaises(ValueError): self.do_ack()

    def test_first_motion_gate_is_equal_in_full_and_sparse(self):
        for sparse in (False,True):
            self.make_robot(sparse)
            with self.assertRaises(ToolError): self.robot._stateful('move_to',{},lambda x:x)
            self.assertEqual(self.env.issued_actions,0)

    def test_full_proposal_needs_ack_then_prefix_and_fresh_feedback(self):
        proposal=self.robot.ewam_propose()
        with self.assertRaises(ToolError): self.robot.ewam_execute(proposal['prediction_id'],'Useful progress','Visible target',2)
        self.assertEqual(self.env.issued_actions,0)
        self.do_ack(); self.assertTrue(self.ctx['proposal']['delivered_to_model'])
        report=self.robot.ewam_execute(proposal['prediction_id'],'Useful progress','Visible target',2)
        self.assertEqual(report['native_actions'],6); self.assertEqual(report['discarded_native_actions'],42)
        self.assertTrue(self.ctx['awaiting_ewam_observation']); self.assertIsNone(self.ctx['proposal'])
        with self.assertRaises(ToolError): self.robot.ewam_propose()
        with self.assertRaises(ToolError): self.robot._stateful('move_to',{},lambda x:x)
        with self.assertRaises(ValueError): acknowledge_visual(self.ctx,self.latest,self.ack)
        self.do_ack()
        with patch.object(RobodojoOfficialColourRobot,'_stateful',return_value='allowed'):
            self.assertEqual(self.robot._stateful('move_to',{},lambda x:x),'allowed')
        with self.assertRaises(ToolError): self.robot.ewam_execute(proposal['prediction_id'],'Retry','Visible',2)

    def test_sparse_host_review_and_maximum_prefix(self):
        self.make_robot(True)
        with self.assertRaises(ToolError): self.robot.ewam_propose()
        with self.assertRaises(ToolError): self.robot.ewam_execute('x','Reason','Evidence')
        report=self.robot.ewam_step('Useful bounded progress','Visible target',model_actions=16)
        self.assertEqual(self.env.issued_actions,48); self.assertEqual(report['native_actions'],48)
        rows=[json.loads(l) for l in (self.out/'doctor_events.jsonl').read_text().splitlines()]
        review=next(r for r in rows if r['event']=='host_fk_review')
        self.assertTrue(review['passed']); self.assertTrue(review['finite_actions'])
        with self.assertRaises(ToolError): self.robot.ewam_step('Again','Target')

    def test_sparse_rejects_bad_fk_without_native_motion(self):
        self.make_robot(True)
        self.native.obs['state']['left_ee_pose'][0]+=.01
        with self.assertRaises(ToolError): self.robot.ewam_step('Progress','Visible')
        self.assertEqual(self.env.issued_actions,0); self.assertEqual(self.ctx['ewam_attempts'],0)
        self.assertIsNone(self.ctx['proposal'])

    def test_model_prefix_bounds_and_native_limit(self):
        self.make_robot(True)
        for bad in (0,17,True,1.5):
            with self.assertRaises(ValueError): self.robot.ewam_step('Progress','Visible',model_actions=bad)
        self.env.initial_limit=5; self.native.limit=5
        report=self.robot.ewam_step('Progress','Visible',model_actions=16)
        self.assertEqual(report['native_actions'],5)
        with self.assertRaises(EpisodeFinished): self.env.step({})
        self.assertEqual(self.env.issued_actions,5)
        self.assertEqual(self.env.actions_by_owner,{'primitive':0,'ewam':5})

    def test_cell_limit_prevents_execution(self):
        s=Session.__new__(Session); s.cells=100; s.out=self.out
        with self.assertRaisesRegex(RuntimeError,'100-cell'): s.python('print(123)')

    def test_mailbox_every_command_has_snapshot_and_persistent_python(self):
        args=types.SimpleNamespace(out=self.out/'mailbox',variant='full',lane='A')
        session=Session(args,{})
        session.env=self.env; session.context=self.ctx
        session.robo=types.SimpleNamespace(state=lambda: {'visible_robot_state':True})
        session.namespace={'__builtins__':{'print':print},'robo':session.robo,'np':np}
        session.ready_at=time.monotonic(); session.snapshot()
        commands=[dict(op='python',code='remembered = 17'),
                  dict(op='python',code='print(remembered + 1)'),
                  dict(op='ack_visual',observation_index=0,native_action_index=0,
                       image_sha256=self.ack['image_sha256'],viewed_by='window_AI',evidence='Old frame'),
                  dict(op='finish')]
        for index,command in enumerate(commands):
            identifier=f'command{index}'
            path=session.out/'commands'/(identifier+'.json')
            path.write_text(json.dumps(dict(id=identifier,note='Offline mailbox check',**command)))
            os.utime(path,ns=(1000000+index,1000000+index))
        session.loop()
        responses=[json.loads((session.out/'responses'/f'command{i}.json').read_text()) for i in range(4)]
        self.assertEqual(responses[1]['output']['stdout'],'18\n')
        self.assertIn('exact latest observation_index',responses[2]['error'])
        self.assertEqual([r['observation']['observation_index'] for r in responses],[1,2,3,4])
        self.assertEqual(len(list((session.out/'observations').glob('*/*.png'))),15)
        self.assertEqual(session.stop_reason,'planner_finish')
        self.assertEqual(json.loads((session.out/'latest.json').read_text())['observation_index'],4)

    def test_frozen_factory_is_identical(self):
        old=ROOT.parent/'ewam_doctor_efficiency48/doctor_robot.py'
        self.assertEqual((ROOT/'doctor_robot.py').read_bytes(),old.read_bytes())


if __name__=='__main__': unittest.main(verbosity=2)
