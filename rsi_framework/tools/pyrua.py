"""The unchanged PyRUA RGB robot primitives plus SAM3 perception (PyRUA environment only).

Perception is wrapped by the original RecordedSam3. Motion tools spend the shared
native budget through PyRUA's own `_stateful` path; the framework never generates
primitive actions itself. Python cells use PyRUA's own sandbox.
"""
from rsi_framework.compat import paths
from rsi_framework.core.registry import register
from rsi_framework.tools.base import ToolSet

READONLY = ('state', 'show', 'locate', 'doctor_memory')


class PyRUATools(ToolSet):
    requires_environment = ('robodojo',)

    def __init__(self, *, perception='sam3_local', primitives='no-vla', expose=None, readonly=READONLY):
        self.perception = perception
        self.primitives = primitives
        self.expose = tuple(expose) if expose else None
        self.readonly = tuple(readonly)

    def describe(self):
        return {'block': 'pyrua', 'perception': self.perception, 'primitives': self.primitives,
                'expose': self.expose}

    @property
    def tool_error(self):
        from pyrualean._backend import ToolError
        return ToolError

    def sandbox(self):
        from pyrualean.sandbox import audit_source, blocked_names, make_builtins
        from pyrualean._backend import EpisodeFinished
        return make_builtins, audit_source, blocked_names, (EpisodeFinished,)

    def attach(self, run):
        paths.ensure_importable()
        if self.perception == 'sam3_local':
            from recording import RecordedSam3
            from pyrualean.hosts.sam3_local import LocalSam3
            run.toolkit._primitives.sam3_client = RecordedSam3(LocalSam3(), run.context)

    def backend(self, run):
        return run.env_block.robot_backend(run)

    def base_robot_class(self):
        paths.ensure_importable()
        from pyrualean.robodojo_xpl_rgb import RobodojoOfficialColourRobot
        return RobodojoOfficialColourRobot


@register('tools', 'pyrua', description='Unchanged PyRUA RGB primitives + recorded SAM3 (PyRUA env)')
def make(**options):
    return PyRUATools(**options)
