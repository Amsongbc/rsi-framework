"""ToolSet contract shared by every tool block.

  base_robot_class()   class(backend, primitives=...) with _stateful, _ledger, _last_state, state()
  backend(run)         what that class is constructed with
  attach(run)          perception or other services (optional)
  sandbox()            (make_builtins, audit_source, blocked_names, finished_exceptions) for Python cells
  tool_error           exception class protocols raise for refusals
  requires_environment environment block names it works with (None: any)
Public methods of the robot class are the tools; specs() and call() expose them to
structured policies, Python cells call them as robo.<name>(...).
"""
import inspect

from rsi_framework.core.contracts import ToolSpec

HIDDEN = ('ewam', 'ewam_propose', 'ewam_execute', 'ewam_step', 'vla', 'diagnose')


class ToolSet:
    kind = 'tools'
    evidence = 'real'
    requires_environment = None
    readonly = ()
    expose = None

    def attach(self, run):
        pass

    def specs(self, robot):
        out = []
        for name in sorted(dir(type(robot))):
            member = getattr(type(robot), name)
            if name.startswith('_') or not callable(member) or name in HIDDEN:
                continue
            if self.expose is not None and name not in self.expose:
                continue
            signature = inspect.signature(member)
            params = {p: ('required' if v.default is inspect._empty else repr(v.default))
                      for p, v in signature.parameters.items() if p != 'self'}
            out.append(ToolSpec(name, 'readonly' if name in self.readonly else 'motion',
                                inspect.getdoc(member) or '', params))
        return out

    def call(self, run, name, args):
        allowed = {spec.name for spec in self.specs(run.robo)}
        if name not in allowed:
            raise ValueError(f'Unknown or unavailable tool: {name}')
        if not isinstance(args, dict):
            raise ValueError('Tool arguments must be an object')
        return getattr(run.robo, name)(**args)
