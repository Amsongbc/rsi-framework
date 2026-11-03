"""Assemble blocks from a profile and refuse incompatible combinations.

Modes:
  llm_policy  LLM (or external controller) drives tools; no VLA in the execution path.
  agent_vla   the policy may hand language to a VLA or call tools.
  pure_vla    the VLA alone, task instruction, no LLM.
  compat      an original cohort entry (see rsi_framework.compat); not assembled here.
Learning (VLA training) and Memory are separate switches; a frozen profile never
creates a trainer. After the blocks exist, their contracts are checked against each
other: the environment publishes a RobotSpec (or the profile names one), the VLA's
cameras / state keys / action keys must fit that robot, and a tool set must support
the environment. A mismatch names the adapter to change.
"""
import dataclasses

from rsi_framework.core.registry import REGISTRY, load_builtin, load_plugins

MODES = ('llm_policy', 'agent_vla', 'pure_vla')
ORDER = ('environment', 'robot', 'vla', 'llm_backend', 'tools', 'memory', 'protocol', 'policy', 'recorder')
RANK = {'real': 0, 'replay': 1, 'software_synthetic': 2}


class AssemblyError(ValueError):
    pass


def _spec(profile, kind):
    spec = profile.get(kind)
    if spec is None:
        return None
    if isinstance(spec, str):
        return {'block': spec, 'options': {}}
    return {'block': spec['block'], 'options': dict(spec.get('options', {}))}


@dataclasses.dataclass
class Assembly:
    profile: dict
    blocks: dict

    def describe(self):
        return {kind: (block.describe() if hasattr(block, 'describe') else type(block).__name__)
                for kind, block in self.blocks.items() if block is not None}

    def evidence(self):
        labels = {}
        for kind, block in self.blocks.items():
            if block is None:
                continue
            labels[kind] = getattr(block, 'evidence_label', None) or getattr(block, 'evidence', 'real')
        overall = max(labels.values(), key=RANK.__getitem__) if labels else 'real'
        return {'overall': overall, 'blocks': labels}

    def require_evidence(self):
        evidence = self.evidence()
        if evidence['overall'] != 'real' and not self.profile.get('allow_stand_ins', False):
            stand_ins = {k: v for k, v in evidence['blocks'].items() if v != 'real'}
            raise AssemblyError(f'Stand-in blocks {stand_ins} need allow_stand_ins=true; '
                                'their results are labelled and never reported as real runs')


def validate(profile):
    load_builtin()
    load_plugins(profile.get('plugins'))
    mode = profile.get('mode')
    if mode == 'compat':
        raise AssemblyError('compat profiles run through rsi_framework.compat, not the generic assembly')
    if mode not in MODES:
        raise AssemblyError(f'Unknown mode {mode!r}; expected one of {MODES} or compat')
    for kind in ('environment', 'tools', 'protocol', 'policy'):
        if _spec(profile, kind) is None:
            raise AssemblyError(f'{mode} needs a {kind} block')
    vla, llm = _spec(profile, 'vla'), _spec(profile, 'llm_backend')
    protocol = REGISTRY.entry('protocol', _spec(profile, 'protocol')['block'])
    policy = _spec(profile, 'policy')
    ops = tuple(policy['options'].get('ops', ()))
    if mode == 'llm_policy':
        if vla is not None:
            raise AssemblyError('llm_policy executes without a VLA; remove the vla block '
                                '(offline VLA learning is configured under learning)')
        if 'vla' in ops:
            raise AssemblyError('llm_policy cannot offer the vla operation')
        if llm is None:
            raise AssemblyError('llm_policy needs an llm_backend (an API, a self-hosted model or command_queue)')
    if mode == 'agent_vla':
        if vla is None:
            raise AssemblyError('agent_vla needs a vla block')
        if llm is None:
            raise AssemblyError('agent_vla needs an llm_backend')
    if mode == 'pure_vla':
        if vla is None or policy['block'] != 'vla_only':
            raise AssemblyError('pure_vla needs a vla block and the vla_only policy')
        if llm is not None:
            raise AssemblyError('pure_vla has no LLM backend')
        if _spec(profile, 'memory') and _spec(profile, 'memory')['block'] != 'off':
            raise AssemblyError('pure_vla keeps Memory off')
    if vla is None and protocol.factory(**_spec(profile, 'protocol')['options']).requires_vla:
        raise AssemblyError(f'protocol {protocol.name} requires a VLA')
    if protocol.name.startswith('window_') and (llm is None or llm['block'] != 'command_queue'
                                                or not llm['options'].get('mailbox', True)):
        raise AssemblyError('The window protocol records visual acknowledgements as window_AI; an automatic '
                            'backend needs its own confirmation protocol, not a relabelled window one')
    if 'ack_visual' in ops and not protocol.name.startswith('window_'):
        raise AssemblyError('ack_visual belongs to the window protocol')
    learning = profile.get('learning') or {}
    if learning.get('enabled'):
        for key in ('dataset', 'trainer', 'evaluation', 'release'):
            if not learning.get(key):
                raise AssemblyError(f'learning.enabled needs learning.{key}')
        if learning['trainer'].get('block') in (None, 'none'):
            raise AssemblyError('learning.enabled needs a real trainer block, not none')
        if mode == 'llm_policy' and learning.get('serve_candidate'):
            raise AssemblyError('llm_policy trains offline only; a trained VLA needs a new agent_vla release')
    elif learning.get('trainer') and learning['trainer'].get('block') not in (None, 'none'):
        raise AssemblyError('A frozen profile (learning.enabled=false) cannot name a trainer')
    memory = _spec(profile, 'memory') or {'block': 'off', 'options': {}}
    if memory['block'] in ('frozen', 'learn') and not memory['options'].get('snapshot'):
        raise AssemblyError(f'memory {memory["block"]} needs a snapshot')
    if memory['block'] == 'learn' and not memory['options'].get('store'):
        raise AssemblyError('memory learn needs a store for new versions')
    return mode


def assemble(profile):
    validate(profile)
    blocks = {}
    for kind in ORDER:
        spec = _spec(profile, kind)
        if kind == 'memory' and spec is None:
            spec = {'block': 'off', 'options': {}}
        if kind == 'recorder' and spec is None:
            spec = {'block': 'generic', 'options': {}}
        if kind == 'robot' and spec is None:
            if not blocks['environment'].robot:
                raise AssemblyError('The environment publishes no RobotSpec; name one under robot')
            spec = {'block': blocks['environment'].robot, 'options': {}}
        if spec is None:
            blocks[kind] = None
            continue
        options = spec['options']
        if kind == 'policy' and blocks.get('llm_backend') is not None:
            options = dict(options, channel=blocks['llm_backend'])
        blocks[kind] = REGISTRY.create(kind, spec['block'], **options)
    check_contracts(profile, blocks)
    return Assembly(profile, blocks)


def check_contracts(profile, blocks):
    env, robot, vla, tools = blocks['environment'], blocks['robot'], blocks.get('vla'), blocks['tools']
    env_name = REGISTRY.canonical('environment', _spec(profile, 'environment')['block'])
    problems = []
    if env.robot and REGISTRY.canonical('robot', env.robot) != REGISTRY.canonical('robot', robot.name):
        problems.append(f'environment {env_name} publishes robot {env.robot}, profile names {robot.name}')
    protocol = blocks['protocol']
    tools_name = REGISTRY.canonical('tools', _spec(profile, 'tools')['block'])
    if getattr(protocol, 'requires_tools', None) and tools_name not in protocol.requires_tools:
        problems.append(f'protocol {_spec(profile, "protocol")["block"]} is built on tools {list(protocol.requires_tools)}, not {tools_name}')
    if tools.requires_environment and env_name not in tools.requires_environment:
        problems.append(f'tools {_spec(profile, "tools")["block"]} work with {list(tools.requires_environment)}, '
                        f'not {env_name}; use tools "joint" or write a tool set for this environment')
    if vla is not None:
        contract, name = vla.contract, _spec(profile, 'vla')['block']
        missing_cameras = [c for c in contract.get('cameras', ()) if c not in robot.cameras]
        missing_state = [k for k in contract.get('state_keys', ()) if k not in robot.state_keys]
        actions = dict(contract.get('action_keys', {}))
        if missing_cameras:
            problems.append(f'vla {name} needs cameras {missing_cameras} that robot {robot.name} does not publish')
        if missing_state:
            problems.append(f'vla {name} needs state {missing_state} that robot {robot.name} does not publish')
        if actions != dict(robot.action_keys):
            problems.append(f'vla {name} outputs actions {actions}, robot {robot.name} takes {dict(robot.action_keys)}; '
                            'map them in the VLA adapter')
        if 'stride' not in contract:
            problems.append(f'vla {name} contract lacks stride (native actions per model action)')
    if problems:
        raise AssemblyError('; '.join(problems))
