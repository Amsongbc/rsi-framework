"""Robot API reference renderer with the upstream entry name (TEST DOUBLE)."""
import inspect


def api_reference(robot_cls, primitives='no-vla'):
    lines = [inspect.getdoc(robot_cls) or '']
    for name in sorted(dir(robot_cls)):
        member = getattr(robot_cls, name)
        if name.startswith('_') or not callable(member):
            continue
        lines.append(f'robo.{name}{inspect.signature(member)}')
        lines.append(inspect.getdoc(member) or '')
    return '\n'.join(lines)
