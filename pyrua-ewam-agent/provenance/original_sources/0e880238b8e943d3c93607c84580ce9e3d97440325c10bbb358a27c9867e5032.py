"""EWAM extension to the unchanged PyRUA RGB robot API."""
from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'PyRUA-Lean/src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ewam_comparison'))
from pyrualean.robodojo_xpl_rgb import RobodojoOfficialColourRobot
from pyrualean._backend import ToolError
from ewam_service import public_observation
from recording import append_json


def robot_class(context):
    class HybridRobot(RobodojoOfficialColourRobot):
        def _stateful(self,tool,kwargs,build):
            if not context.get('ewam_attempts',0):
                append_json(context['out']/'protocol_events.jsonl',{
                    'time':time.time(),'event':'analytic_motion_blocked_before_ewam','tool':tool})
                raise ToolError(tool,'This cohort requires an EWAM attempt first. Read RGB/state/locate if needed, then call robo.ewam for real task progress before analytic motion.')
            if context.get('awaiting_ewam_observation'):
                append_json(context['out']/'protocol_events.jsonl',{
                    'time':time.time(),'event':'analytic_motion_blocked_before_visual_feedback','tool':tool})
                raise ToolError(tool,'First return fresh RGB with robo.show and end this cell. Inspect that image in the next model turn before issuing primitive corrections.')
            return super()._stateful(tool,kwargs,build)

        def ewam(self, instruction: str | None = None, chunks: int = 1,
                 model_actions: int = 16) -> list[dict]:
            """Execute bounded manipulation using the frozen learned EWAM policy.

            instruction=None uses the benchmark instruction. You may provide an
            observable subtask instruction. Each chunk replans from fresh three-camera
            RGB and joint state. 16 model actions yield 48 native actions; choose
            1..16 model actions and exactly one chunk. Every action spends the same native
            budget as move_to/home/set_gripper. No reward, task answer or object ground
            truth is available. The returned report contains action counts and timing.

            This experiment requires EWAM FIRST for a meaningful task-progress segment,
            before any analytic motion. Read-only observation/perception may precede it.
            After the segment, call robo.show(...) and END THE CELL so the model can see
            fresh RGB. Only in a subsequent model turn choose more EWAM or geometric
            correction if the learned action failed/stalled/needs precise adjustment.
            Primitive motion and a second EWAM call are blocked until visual feedback
            has reached the model. Do not use dummy/no-op calls just to unlock primitives.
            Do not blindly repeat an unsuccessful segment. Print concise
            observable progress, object/subgoal selection and reasons for switching
            execution methods, so the recovery can be analyzed later. Final success
            is determined only by the benchmark client.
            """
            if chunks!=1:
                raise ValueError('The EWAM-first cohort uses exactly one chunk before visual feedback')
            if context.get('awaiting_ewam_observation'):
                raise ToolError('ewam','Return robo.show images and end this cell before another learned segment.')
            if not isinstance(model_actions,int) or not 1<=model_actions<=16:
                raise ValueError('model_actions must be 1..16')
            if instruction is not None and not isinstance(instruction,str):
                raise ValueError('instruction must be a string or None')
            env = context['env']
            toolkit = context['toolkit']
            start = time.monotonic()
            before = env.issued_actions
            previous_owner,env.owner = env.owner,'ewam'
            reports = []
            record = self._ledger.open('ewam',{'instruction':instruction,'chunks':chunks,
                                               'model_actions':model_actions},stateful=True)
            try:
                for _ in range(chunks):
                    if env.episode_over() or env.issued_actions>=env.initial_limit:
                        break
                    context['ewam_attempts']=context.get('ewam_attempts',0)+1
                    result = context['ewam'].call('infer',session_id=context['session_id'],
                        obs=public_observation(env.get_obs(),instruction))
                    context['ewam_calls'] += 1
                    report = {k:v for k,v in result.items() if k!='actions'}
                    report.update(planned_native_actions=len(result['actions']),executed_native_actions=0)
                    append_json(context['out']/'ewam_plans.jsonl',{
                        'time':time.time(),'call':context['ewam_calls'],
                        'start_native_action':env.issued_actions,
                        'instruction_override':instruction,'model_actions_requested':model_actions,
                        'result':result})
                    for action in result['actions'][:model_actions*3]:
                        if env.episode_over() or env.issued_actions>=env.initial_limit:
                            break
                        _,_,done,_ = env.step(action)
                        report['executed_native_actions'] += 1
                        if done:
                            break
                    reports.append(report)
                    if env.episode_over():
                        break
                record.ok = True
                return reports
            except BaseException as exc:
                record.error = type(exc).__name__+': '+str(exc)
                raise
            finally:
                env.owner = previous_owner
                context['awaiting_ewam_observation']=True
                record.env_steps = env.issued_actions-before
                record.elapsed_s = time.monotonic()-start
                record.summary = {'reports':reports,'native_actions':record.env_steps}
                # Keep the original robot state cache/observation protocol coherent.
                toolkit.state.record(env.get_obs(),env.get_status())
                self._last_state = None
                append_json(context['out']/'ewam_calls.jsonl',{
                    'time':time.time(),'instruction_override':instruction,
                    'model_actions':model_actions,'chunks':chunks,'reports':reports,
                    'native_actions':record.env_steps,'wall_s':record.elapsed_s,
                    'error':record.error})
    return HybridRobot
