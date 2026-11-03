"""LLM policy protocol: the policy drives the original primitives directly, no VLA.

Optional `observe_after_motion` requires the result of a motion to reach the policy
before the next motion (the policy reports delivery each time it hands over the
latest observation). Primitive action generation stays inside PyRUA.
"""
from rsi_framework.core.registry import register


class ToolsOnlyProtocol:
    kind = 'protocol'
    evidence = 'real'
    requires_vla = False
    accepts_vla_call = False

    def __init__(self, *, max_cells=100, wall_budget_s=5400, observe_after_motion=False):
        self.max_cells = max_cells
        self.wall_budget_s = wall_budget_s
        self.observe_after_motion = observe_after_motion

    def describe(self):
        return {'block': 'tools_only', 'max_cells': self.max_cells, 'wall_budget_s': self.wall_budget_s,
                'observe_after_motion': self.observe_after_motion}

    def initial_context(self, run):
        return dict(out=run.out, session_id=run.out.name, awaiting_observation=False,
                    doctor_memory=run.memory.legacy_value())

    def build_robot(self, run):
        ToolError = run.tools.tool_error
        base, context, protocol = run.tools.base_robot_class(), run.context, self

        class ToolsOnlyRobot(base):
            def _stateful(self, tool, kwargs, build):
                if protocol.observe_after_motion and context.get('awaiting_observation'):
                    raise ToolError(tool, 'Observe the result of the previous motion before moving again.')
                try:
                    return super()._stateful(tool, kwargs, build)
                finally:
                    if protocol.observe_after_motion:
                        context['awaiting_observation'] = True
        return ToolsOnlyRobot(run.tools.backend(run), primitives=run.tools.primitives)

    def observation_delivered(self, run):
        run.context['awaiting_observation'] = False

    def acknowledge(self, run, command):
        raise ValueError('ack_visual is not part of the tools_only protocol')

    def vla_call(self, run, language, model_actions):
        raise ValueError('This profile has no VLA; use the tools')

    def observation_fields(self, run):
        return dict(awaiting_observation=run.context.get('awaiting_observation', False))


@register('protocol', 'tools_only', description='LLM drives PyRUA primitives directly; no VLA')
def make(**options):
    return ToolsOnlyProtocol(**options)
