"""RoboDoctor-inspired proposal supervision; original PyRUA primitives remain intact.

Adapted concepts: policy proposals, contiguous prefixes, fresh-observation gates,
grounded diagnosis memory, and calibrated policy/direct routing. This module
does not import RoboDoctor's action implementation or change policy weights.
"""

# Export portability change: resolve external projects via PROJECTS_ROOT.
import os as _export_os
import time
import uuid

import numpy as np

from hybrid_robot import RobodojoOfficialColourRobot, ToolError, public_observation
from recording import append_json
from pyrualean.hosts import robodojo_kinematics as kin


def preview(actions, state):
    """Robot-only FK, checked against measured EEF; no object state is queried."""
    if not actions:
        raise ValueError('Empty EWAM prediction')
    measured = {}
    for arm in ('left', 'right'):
        q = np.asarray(state[arm + '_arm_joint_state'], float).reshape(6)
        pose = np.asarray(state[arm + '_ee_pose'], float).reshape(7)
        error = float(np.linalg.norm(kin.fk(arm, q)[:3, 3] - pose[:3]))
        measured[arm] = {'joint_state': q.tolist(), 'measured_link6_pose': pose.tolist(),
                         'fk_position_error_m': error}
    samples = []
    for index in sorted(set([0, min(11, len(actions)-1), min(23, len(actions)-1), len(actions)-1])):
        row = {'native_prefix_steps': index + 1, 'arms': {}}
        for arm in ('left', 'right'):
            q = np.asarray(actions[index][arm + '_arm_joint_state'], float).reshape(6)
            if not np.isfinite(q).all():
                raise ValueError('Nonfinite EWAM joint proposal')
            pose = kin.fk(arm, q)
            row['arms'][arm] = {'predicted_link6_xyz': pose[:3, 3].tolist(),
                                'predicted_tcp_xyz': kin.tcp(pose).tolist(),
                                'gripper': np.asarray(actions[index][arm + '_ee_joint_state']).tolist()}
        samples.append(row)
    return {'kind': 'robot-only kinematic prediction, NOT observed object/contact outcome',
            'measured_start': measured,
            'fk_position_check_passed': all(x['fk_position_error_m'] < .003 for x in measured.values()),
            'samples': samples}


def robot_class(context):
    class DoctorRobot(RobodojoOfficialColourRobot):
        # Inherit the robot documentation: PyRUA's reference renderer applies
        # exact no-VLA substitutions to that established class contract.
        def _stateful(self, tool, kwargs, build):
            if context.get('awaiting_ewam_observation'):
                raise ToolError(tool, 'Show fresh camera RGB and end the cell; inspect it next turn before more motion.')
            if context.get('proposal') and not context['proposal']['delivered_to_model']:
                raise ToolError(tool, 'Inspect the new proposal and RGB in the next model turn before motion.')
            # Any attempted primitive invalidates the entire old policy proposal.
            context['proposal'] = None
            return super()._stateful(tool, kwargs, build)

        def ewam_propose(self, instruction: str | None = None) -> dict:
            """Predict an EWAM chunk WITHOUT moving. Inspect fresh RGB and robot-only
            FK preview before release. A prediction is not evidence of grasp/contact.
            End this cell after showing RGB; next model turn execute a prefix or use
            the unchanged primitives. Proposals expire after ANY physical motion.
            The direct branch disables EWAM. Do not repeat a rejected unchanged
            proposal at the same state; use grounded perception or correction.
            """
            if context['doctor_phase'] == 'direct':
                raise ToolError('ewam_propose', 'Warmup gate selected direct primitives; EWAM is disabled in this branch.')
            if context.get('awaiting_ewam_observation'):
                raise ToolError('ewam_propose', 'Observe previous execution in the next model turn first.')
            env = context['env']
            if env.episode_over() or env.issued_actions >= env.initial_limit:
                raise ToolError('ewam_propose', 'Episode has ended or its action budget is exhausted.')
            previous = context.get('last_proposal_native_step')
            if previous == env.issued_actions:
                raise ToolError('ewam_propose', 'A prediction already exists at this state; execute it or correct with primitives.')
            result = context['ewam'].call('infer', session_id=context['session_id'],
                                         obs=public_observation(env.get_obs(), instruction))
            context['ewam_calls'] += 1
            identifier = uuid.uuid4().hex
            diagnostics = preview(result['actions'], env.get_obs()['state'])
            proposal = {'prediction_id': identifier, 'native_action_index': env.issued_actions,
                        'call': context['ewam_calls'], 'result': result,
                        'instruction_override': instruction, 'preview': diagnostics,
                        'delivered_to_model': False}
            context['proposal'] = proposal
            context['last_proposal_native_step'] = env.issued_actions
            append_json(context['out']/'ewam_plans.jsonl', dict(time=time.time(), **proposal))
            append_json(context['out']/'doctor_events.jsonl', {
                'event': 'proposal_created', 'time': time.time(), 'prediction_id': identifier,
                'native_action_index': env.issued_actions, 'inference_s': result['inference_s']})
            return {'prediction_id': identifier, 'native_action_index': env.issued_actions,
                    'native_horizon': len(result['actions']), 'max_model_actions': 16,
                    'robot_only_fk': diagnostics, 'physical_steps': 0,
                    'instruction': result.get('instruction'),
                    'next': 'Show RGB, END CELL, inspect preview and RGB next turn; execute a prefix or use primitives.'}

        def ewam_execute(self, prediction_id: str, reason: str, evidence: str,
                         model_actions: int = 16) -> dict:
            """Release 1..16 contiguous EWAM model actions (3 native steps each).
            Explain expected progress and current visual evidence. Use shorter
            prefixes near contact/occlusion/uncertainty. No policy suffix may be
            resumed: after execution show RGB, end cell, and replan from new state.
            """
            env = context['env']
            proposal = context.get('proposal')
            if (not proposal or prediction_id != proposal['prediction_id']
                    or proposal['native_action_index'] != env.issued_actions):
                raise ToolError('ewam_execute', 'Missing or stale prediction. Infer from the current state.')
            if not proposal['delivered_to_model']:
                raise ToolError('ewam_execute', 'First show RGB and end the proposal cell; inspect the proposal next turn.')
            if context.get('awaiting_ewam_observation'):
                raise ToolError('ewam_execute', 'Observe previous physical execution first.')
            if type(model_actions) is not int or not 1 <= model_actions <= 16:
                raise ValueError('model_actions must be 1..16')
            if not isinstance(reason, str) or not reason.strip() or not isinstance(evidence, str) or not evidence.strip():
                raise ValueError('A release requires reason and current observed evidence')
            context['proposal'] = None  # discard suffix even on an execution error
            context['ewam_attempts'] += 1
            before = env.issued_actions
            started = time.monotonic()
            previous_owner, env.owner = env.owner, 'ewam'
            error = None
            try:
                for action in proposal['result']['actions'][:model_actions*3]:
                    if env.episode_over() or env.issued_actions >= env.initial_limit:
                        break
                    _, _, done, _ = env.step(action)
                    if done:
                        break
            except BaseException as exc:
                error = type(exc).__name__ + ': ' + str(exc)
                raise
            finally:
                env.owner = previous_owner
                context['awaiting_ewam_observation'] = True
                context['toolkit'].state.record(env.get_obs(), env.get_status())
                self._last_state = None
                event = {'time': time.time(), 'prediction_id': prediction_id,
                         'call': proposal['call'], 'model_actions': model_actions,
                         'reason': reason, 'evidence': evidence,
                         'native_actions': env.issued_actions-before,
                         'discarded_native_actions': len(proposal['result']['actions'])-(env.issued_actions-before),
                         'wall_s': time.monotonic()-started, 'error': error}
                append_json(context['out']/'ewam_calls.jsonl', event)
                append_json(context['out']/'doctor_events.jsonl', dict(event='policy_prefix_executed', **event))
            return event

        def diagnose(self, subgoal: str, observation: str, decision: str,
                     expected_change: str, outcome: str = 'unverified') -> dict:
            """Record concise episode memory grounded in CURRENT RGB/robot state.
            decision: continue_policy, primitive_correction, or verify.
            State the observed failure/progress, next expected visible change, and
            observed outcome. Do not claim contact or success from a proposal alone.
            Read prior notes with doctor_memory(). Notes do not move the robot.
            """
            if decision not in ('continue_policy', 'primitive_correction', 'verify'):
                raise ValueError('Unknown diagnosis decision')
            values = [subgoal, observation, expected_change, outcome]
            if not all(isinstance(x, str) and 0 < len(x) <= 1200 for x in values):
                raise ValueError('Diagnosis fields must be concise nonempty strings, max1200 chars')
            row = {'time': time.time(), 'native_action_index': context['env'].issued_actions,
                   'subgoal': subgoal, 'observation': observation, 'decision': decision,
                   'expected_change': expected_change, 'outcome': outcome,
                   'evidence_kind': 'model observation claim, not official evaluation'}
            context.setdefault('diagnoses', []).append(row)
            append_json(context['out']/'diagnoses.jsonl', row)
            return row

        def doctor_memory(self) -> dict:
            """Read current compressed scratch notes and frozen warmup experience.
            Prior lessons are hypotheses: reobserve and re-ground in this scene.
            Old coordinates/progress are not current facts. No test outcomes enter
            frozen experience. Host gate score is not a live task reward.
            """
            return {'phase': context['doctor_phase'], 'warmup_experience': context.get('doctor_memory'),
                    'recent_diagnoses': context.get('diagnoses', [])[-4:]}

    return DoctorRobot


def protocol(config):
    doctor = config['doctor']
    memory = doctor.get('frozen_memory')
    return '''\n\n# RoboDoctor-inspired supervision with unchanged PyRUA primitives
You retain the full existing geometric/grasp/move primitives. Learned weights
are fixed. Default to a useful learned subgoal when current evidence supports
it; precision, wrong-object motion, unsafe contact or repeated stalling justify
the original primitives. Do not call EWAM merely to satisfy a quota.
Before EWAM motion use ewam_propose, show current RGB, and END THE CELL. Inspect
the robot-only FK proposal next turn together with RGB; release only a useful
contiguous prefix with ewam_execute or reject it and use primitives. A proposal
does not prove contact/grasp. Shorten the prefix near uncertainty. After any
learned execution, show new RGB and END CELL before more physical action.
All old suffixes are discarded after motion; reinfer on fresh observations.
Use diagnose sparingly at a meaningful failure/progress/switch, with an expected
visible effect and verified or explicitly unverified outcome. Keep a compact
subgoal/progress/failure hypothesis in working memory. Do not blindly repeat a
failed grasp or unchanged geometry; re-ground and revise it. A successful
subinteraction may be useful even if a later step fails. Preserve successful
support/grasp and move only a genuinely idle arm for better visibility.
Experience is qualitative, object-relative advice, never a coordinate replay or
proof of current progress. Task success comes only from official termination.
All EWAM and primitive actions share the unchanged official native budget.
''' + '\nSelected branch: ' + doctor['phase'] + (
        '\nThe three warmup trials scored below the fixed gate: use direct primitives. EWAM is disabled.\n'
        if doctor['phase'] == 'direct' else
        '\nUse the supervised EWAM/primitive hybrid; real visual risk or clear stalling permits rejecting a proposal.\n') + (
        '\nFrozen experience/harness from independent warmup layouts:\n' + __import__('json').dumps(memory, ensure_ascii=False)
        if memory else '\nStart with zero task-specific prior experience.\n')
