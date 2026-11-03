"""Cell sandbox for tool sets that do not bring their own (PyRUA tools use PyRUA's).

Only an explicit builtin subset is available; imports, dunder attribute access and
code-execution builtins are refused before the cell runs.
"""
import ast
import builtins

SAFE = ('abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'Exception', 'filter', 'float', 'getattr', 'hasattr',
        'int', 'isinstance', 'KeyError', 'len', 'list', 'map', 'max', 'min', 'print', 'range', 'repr', 'reversed',
        'round', 'RuntimeError', 'set', 'sorted', 'str', 'sum', 'tuple', 'type', 'TypeError', 'ValueError', 'zip')
BLOCKED = frozenset({'eval', 'exec', 'compile', 'open', '__import__', 'globals', 'locals', 'vars', 'input',
                     'breakpoint', 'exit', 'quit', 'setattr', 'delattr'})


def make_builtins(workspace):
    return {name: getattr(builtins, name) for name in SAFE}


def audit_source(code, workspace=None):
    flags = set()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            flags.add('import')
        if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
            flags.add('dunder_attribute:' + node.attr)
    return sorted(flags)


def blocked_names(code):
    return sorted({n.id for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Name) and n.id in BLOCKED})
