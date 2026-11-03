"""The original current-window protocol (full / sparse), used as is.

The robot class and the visual acknowledgement are the functions in
projects/ewam_window_parallel24/window_session.py, so EWAM-first, proposal expiry,
prefix release, the sparse host screen and the acknowledgement checks are the
original code paths, not re-implementations.
"""
from rsi_framework.compat import paths
from rsi_framework.core.registry import register


class WindowProtocol:
    kind = 'protocol'
    evidence = 'real'
    requires_vla = True
    requires_tools = ('pyrua',)
    accepts_vla_call = False   # EWAM is reached through Python cells (robo.ewam_*)

    def __init__(self, *, variant, efficiency=None, doctor_phase='hybrid', max_cells=100, wall_budget_s=5400):
        if variant not in ('full', 'sparse'):
            raise ValueError('window protocol variant must be full or sparse')
        self.variant = variant
        self.efficiency = dict(efficiency or {})
        self.doctor_phase = doctor_phase
        self.max_cells = max_cells
        self.wall_budget_s = wall_budget_s

    def describe(self):
        return {'block': f'window_{self.variant}', 'efficiency': self.efficiency, 'doctor_phase': self.doctor_phase,
                'max_cells': self.max_cells, 'wall_budget_s': self.wall_budget_s}

    def initial_context(self, run):
        return dict(out=run.out, session_id=run.out.name, doctor_phase=self.doctor_phase,
                    doctor_memory=run.memory.legacy_value(),
                    efficiency=dict(self.efficiency, sparse_supervision=self.variant == 'sparse'),
                    ewam_calls=0, ewam_attempts=0, awaiting_ewam_observation=False)

    def build_robot(self, run):
        paths.ensure_importable()
        from window_session import window_robot_class
        return window_robot_class(run.context)(run.tools.backend(run), primitives='no-vla')

    def acknowledge(self, run, command):
        from window_session import acknowledge_visual
        return acknowledge_visual(run.context, run.latest, command)

    def vla_call(self, run, language, model_actions):
        raise ValueError('The window protocol releases EWAM only through robo.ewam_* in Python cells')

    def observation_fields(self, run):
        proposal = run.context.get('proposal')
        return dict(awaiting_ewam_observation=run.context.get('awaiting_ewam_observation', False),
                    proposal=None if not proposal else {k: proposal[k] for k in ('prediction_id', 'native_action_index', 'delivered_to_model')})


@register('protocol', 'window_full', description='Original window protocol, full proposal review')
def make_full(**options):
    return WindowProtocol(variant='full', **options)


@register('protocol', 'window_sparse', description='Original window protocol, sparse host FK screen')
def make_sparse(**options):
    return WindowProtocol(variant='sparse', **options)
