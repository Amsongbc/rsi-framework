"""Agent + VLA protocol for structured decisions (a new, separately named protocol).

`robo.vla(language, model_actions)` asks the VLA backend for a chunk from the
current public observation and executes a contiguous prefix of
`model_actions * stride` native actions; the rest is discarded. After any motion the
next motion waits until the policy has received the new observation. Optional
`vla_first` blocks primitive motion before the first VLA segment, like the
original EWAM-first cohorts. All actions share the environment's native budget.
"""
import time

from rsi_framework.core.io import append_json
from rsi_framework.core.registry import register


class VLAPrefixProtocol:
    kind = 'protocol'
    evidence = 'real'
    requires_vla = True
    accepts_vla_call = True

    def __init__(self, *, max_model_actions=16, default_model_actions=16, vla_first=False,
                 observe_after_motion=True, max_cells=100, wall_budget_s=5400):
        self.max_model_actions = max_model_actions
        self.default_model_actions = default_model_actions
        self.vla_first = vla_first
        self.observe_after_motion = observe_after_motion
        self.max_cells = max_cells
        self.wall_budget_s = wall_budget_s

    def describe(self):
        return {'block': 'vla_prefix', 'max_model_actions': self.max_model_actions, 'vla_first': self.vla_first,
                'observe_after_motion': self.observe_after_motion, 'wall_budget_s': self.wall_budget_s}

    def initial_context(self, run):
        return dict(out=run.out, session_id=run.out.name, awaiting_observation=False,
                    vla_calls=0, vla_attempts=0, doctor_memory=run.memory.legacy_value())

    def _gate(self, tool):
        ToolError = self.run.tools.tool_error
        if self.observe_after_motion and self.context.get('awaiting_observation'):
            raise ToolError(tool, 'Observe the result of the previous motion before moving again.')
        if self.vla_first and tool != 'vla' and not self.context.get('vla_attempts'):
            raise ToolError(tool, 'This profile requires one VLA segment before primitive motion.')

    def build_robot(self, run):
        base, protocol = run.tools.base_robot_class(), self
        self.context, self.run = run.context, run

        class AgentVLARobot(base):
            def _stateful(self, tool, kwargs, build):
                protocol._gate(tool)
                try:
                    return super()._stateful(tool, kwargs, build)
                finally:
                    if protocol.observe_after_motion:
                        protocol.context['awaiting_observation'] = True

            def vla(self, language: str | None = None, model_actions: int = 16) -> dict:
                """Give `language` (None: the task instruction) to the VLA and execute the first
                `model_actions` of its chunk; the remainder is discarded. Shorten near contact or
                uncertainty. The VLA sees only public RGB, joint state and this language."""
                return protocol.execute(self, language, model_actions)
        return AgentVLARobot(run.tools.backend(run), primitives=run.tools.primitives)

    def execute(self, robot, language, model_actions):
        run, context = self.run, self.context
        if type(model_actions) is not int or not 1 <= model_actions <= self.max_model_actions:
            raise ValueError(f'model_actions must be 1..{self.max_model_actions}')
        if language is not None and (not isinstance(language, str) or not language.strip()):
            raise ValueError('language must be a non-empty string or None')
        self._gate('vla')
        env = run.env
        if env.episode_over() or env.issued_actions >= env.initial_limit:
            raise run.tools.tool_error('vla', 'Episode has ended or its action budget is exhausted.')
        context['vla_calls'] += 1
        call = context['vla_calls']
        start = env.issued_actions
        result = run.vla.infer(run, env.get_obs(), language)
        stride = run.vla.contract['stride']
        append_json(run.out / 'vla_plans.jsonl', dict(time=time.time(), call=call, call_id=f'vla{call:04d}',
                    start_native_action=start, language=language, model_actions=model_actions, result=result))
        context['vla_attempts'] += 1
        started, error = time.monotonic(), None
        previous, env.owner = env.owner, getattr(env, 'vla_owner', 'ewam')
        try:
            for action in result['actions'][:model_actions * stride]:
                if env.episode_over() or env.issued_actions >= env.initial_limit:
                    break
                _, _, done, _ = env.step(action)
                if done:
                    break
        except BaseException as exc:
            error = type(exc).__name__ + ': ' + str(exc)
            raise
        finally:
            env.owner = previous
            if self.observe_after_motion:
                context['awaiting_observation'] = True
            run.env_block.after_motion(run)
            robot._last_state = None
            event = dict(time=time.time(), call=call, call_id=f'vla{call:04d}', language=language,
                         instruction=result.get('instruction'), model_actions=model_actions,
                         start_native_action=start, native_actions=env.issued_actions - start,
                         discarded_native_actions=len(result['actions']) - (env.issued_actions - start),
                         wall_s=time.monotonic() - started, error=error)
            append_json(run.out / 'vla_calls.jsonl', event)
        return {k: event[k] for k in ('call_id', 'native_actions', 'discarded_native_actions', 'instruction')}

    def vla_call(self, run, language, model_actions):
        return run.robo.vla(language, self.default_model_actions if model_actions is None else model_actions)

    def observation_delivered(self, run):
        run.context['awaiting_observation'] = False

    def acknowledge(self, run, command):
        raise ValueError('ack_visual is not part of the vla_prefix protocol')

    def observation_fields(self, run):
        return dict(awaiting_observation=run.context.get('awaiting_observation', False),
                    vla_calls=run.context.get('vla_calls', 0))


@register('protocol', 'vla_prefix', description='Agent chooses language for the VLA or a tool; prefix release')
def make(**options):
    return VLAPrefixProtocol(**options)
