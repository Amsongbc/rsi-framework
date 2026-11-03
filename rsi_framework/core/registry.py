"""Name -> factory registry for every swappable block.

A block module registers its factories at import time:

    @register('vla', 'ewam_service', evidence='real', description='...')
    def make(**options): ...

Profiles name blocks by (kind, name). External plugins are plain modules listed in a
profile's `plugins`; importing them registers their blocks the same way.
"""
import dataclasses
import importlib

KINDS = ('environment', 'robot', 'vla', 'llm_backend', 'policy', 'tools', 'protocol', 'memory',
         'recorder', 'dataset', 'trainer', 'evaluator')
EVIDENCE = ('real', 'replay', 'software_synthetic')

BUILTIN_MODULES = (
    'rsi_framework.robots.arx_x5',
    'rsi_framework.environments.pyrua_robodojo',
    'rsi_framework.vla_backends.ewam',
    'rsi_framework.llm_backends.command_queue',
    'rsi_framework.llm_backends.scripted',
    'rsi_framework.llm_backends.http_chat',
    'rsi_framework.policies.commands',
    'rsi_framework.policies.vla_only',
    'rsi_framework.tools.pyrua',
    'rsi_framework.tools.joint',
    'rsi_framework.protocols.window',
    'rsi_framework.protocols.tools_only',
    'rsi_framework.protocols.vla_prefix',
    'rsi_framework.memory.manager',
    'rsi_framework.compat.window_v1',
    'rsi_framework.core.recorders',
    'rsi_framework.data.dataset',
    'rsi_framework.trainers.base',
    'rsi_framework.evaluation.candidates',
)


@dataclasses.dataclass(frozen=True)
class Entry:
    kind: str
    name: str
    factory: object
    evidence: str
    description: str
    provides: tuple
    requires: tuple
    aliases: tuple = ()


class Registry:
    def __init__(self):
        self._entries = {}

    def register(self, kind, name, *, evidence='real', description='', provides=(), requires=(), aliases=()):
        if kind not in KINDS:
            raise ValueError(f'Unknown block kind {kind!r}; expected one of {KINDS}')
        if evidence not in EVIDENCE:
            raise ValueError(f'Unknown evidence label {evidence!r}')

        def decorate(factory):
            entry = Entry(kind, name, factory, evidence, description, tuple(provides), tuple(requires), tuple(aliases))
            for key in [(kind, n) for n in (name, *aliases)]:
                if key in self._entries and self._entries[key].factory is not factory:
                    raise ValueError(f'Block {kind}:{key[1]} is already registered')
                self._entries[key] = entry
            return factory
        return decorate

    def entry(self, kind, name):
        try:
            return self._entries[(kind, name)]
        except KeyError:
            raise KeyError(f'No {kind} block named {name!r}; available: {self.names(kind)}') from None

    def create(self, kind, name, **options):
        return self.entry(kind, name).factory(**options)

    def canonical(self, kind, name):
        return self.entry(kind, name).name

    def names(self, kind):
        return sorted({e.name for (k, _), e in self._entries.items() if k == kind})

    def entries(self):
        unique = {(e.kind, e.name): e for e in self._entries.values()}
        return [unique[key] for key in sorted(unique)]


REGISTRY = Registry()
register = REGISTRY.register


def load_builtin():
    for module in BUILTIN_MODULES:
        importlib.import_module(module)


def load_plugins(modules):
    for module in modules or ():
        importlib.import_module(module)
