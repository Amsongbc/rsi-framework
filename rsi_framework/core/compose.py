"""Readable profiles: name the pieces, defaults do the rest.

  env         robodojo | <plugin>          task, seeds and runtime options come from configs/envs/<env>.json
  vla         ewam | none | <plugin>       none = the LLM alone (code as policy); options from configs/vlas/
  controller  window | api | replay | none who decides; window = file queue (Codex CLI ...), none = VLA alone
Optional: robot, tools, decision (code | calls), protocol, memory, and *_options /
llm / replies to override defaults.

Defaults: decision is `calls` for an API agent with a VLA, otherwise `code`; the
protocol is tools_only without a VLA, the original window_full on RoboDojo with the
window controller, vla_prefix otherwise. A combination that is an original cohort
(compat/presets.py) runs that cohort; implementation=generic runs it on framework
blocks instead.
"""
from rsi_framework.compat import presets
from rsi_framework.core import defaults
from rsi_framework.core.config import merge
from rsi_framework.core.registry import load_builtin, load_plugins

READABLE = ('env', 'robot', 'vla', 'tools', 'controller', 'decision', 'protocol', 'memory', 'study')
CONTROLLERS = ('window', 'api', 'replay', 'none', 'api_responses')
ALIASES = {'pyrua_robodojo': 'robodojo', 'ewam_service': 'ewam'}


def is_readable(profile):
    return 'env' in profile


def normalize(profile):
    r = {k: profile.get(k) for k in READABLE}
    r['env'] = ALIASES.get(r['env'], r['env'])
    r['vla'] = ALIASES.get(r['vla'] or 'none', r['vla'] or 'none')
    r['controller'] = r['controller'] or 'window'
    if r['controller'] not in CONTROLLERS:
        raise ValueError(f'controller must be one of {CONTROLLERS}')
    r['tools'] = r['tools'] or ('pyrua' if r['env'] == 'robodojo' else 'joint')
    if r['decision'] is None and r['controller'] != 'none':
        r['decision'] = 'calls' if r['controller'] in ('api', 'replay') and r['vla'] != 'none' else 'code'
    if r['decision'] not in (None, 'code', 'calls'):
        raise ValueError('decision must be code or calls')
    if r['protocol'] is None:
        if r['vla'] == 'none':
            r['protocol'] = 'tools_only'
        elif r['controller'] == 'window' and r['tools'] == 'pyrua' and r['decision'] == 'code':
            r['protocol'] = 'window_full'
        elif r['controller'] != 'none':
            r['protocol'] = 'vla_prefix'
    return r


def with_defaults(profile):
    load_builtin()
    load_plugins(profile.get('plugins'))
    env = defaults.load('environment', ALIASES.get(profile['env'], profile['env']))
    vla_name = ALIASES.get(profile.get('vla') or 'none', profile.get('vla') or 'none')
    vla = defaults.load('vla', vla_name) if vla_name != 'none' else {}
    controller = profile.get('controller') or 'window'
    ctrl = defaults.load('controller', controller) if controller in ('api', 'replay') else {}
    out = dict(profile)
    out['env_options'] = merge(env.get('options', {}), profile.get('env_options') or {})
    out['vla_options'] = merge(vla.get('options', {}), profile.get('vla_options') or {})
    if controller == 'api':
        out['llm'] = merge(ctrl.get('llm', {}), profile.get('llm') or {})
    out['policy_options'] = merge(ctrl.get('policy_options', {}), profile.get('policy_options') or {})
    out['case_defaults'] = merge({k: env[k] for k in ('task', 'eval_seed', 'layout') if k in env}, profile.get('case') or {})
    service = merge(vla.get('service', {}), profile.get('vla_service') or {})
    out['_defaults'] = {'env': dict(env, options=out['env_options']),
                        'vla': dict(vla, options=out['vla_options'], service=service)}
    return out


def resolve(profile):
    """A readable profile -> a compat profile (original cohort) or a full block profile."""
    if not is_readable(profile):
        return profile
    profile = with_defaults(profile)
    r = normalize(profile)
    preset = presets.match(r)
    implementation = profile.get('implementation') or ('original' if preset else 'generic')
    if preset and implementation != 'generic':
        return dict(presets.to_compat(preset, profile, implementation), case_defaults=profile['case_defaults'])
    if implementation != 'generic':
        raise ValueError(f'implementation {implementation} exists only for original combinations; '
                         'see python -m rsi_framework presets')
    return dict(blocks(profile, r), case_defaults=profile['case_defaults'])


def blocks(profile, r):
    if r['controller'] == 'api_responses' or r['protocol'] in ('doctor', 'ewam_tool', 'ewam_baseline', 'smoke'):
        raise ValueError(f'controller/protocol {r["controller"]}/{r["protocol"]} exist only as original cohorts')
    controller, vla = r['controller'], r['vla']
    protocol = r['protocol'] or 'vla_prefix'
    env_options = dict(profile.get('env_options') or {})
    environment = {'block': r['env'], 'options': env_options}
    if 'perception_gpu' in env_options:
        environment['perception_gpu'] = env_options.pop('perception_gpu')
    protocol_options = dict(profile.get('protocol_options') or {})
    if controller == 'window' and protocol in ('vla_prefix', 'tools_only'):
        protocol_options.setdefault('observe_after_motion', False)   # the file queue has no delivery confirmation yet
    out = {'name': profile.get('name', '_'.join(x for x in (r['env'], vla, controller, protocol) if x)),
           'mode': 'pure_vla' if controller == 'none' else 'llm_policy' if vla == 'none' else 'agent_vla',
           'allow_stand_ins': profile.get('allow_stand_ins', False), 'plugins': profile.get('plugins', []),
           'environment': environment,
           'vla': None if vla == 'none' else {'block': vla, 'options': dict(profile.get('vla_options') or {})},
           'tools': {'block': r['tools'], 'options': dict(profile.get('tools_options') or {})},
           'protocol': {'block': protocol, 'options': protocol_options},
           'memory': {'block': r['memory'] or 'off', 'options': dict(profile.get('memory_options') or {})},
           'learning': profile.get('learning') or {'enabled': False}}
    if r['robot']:
        out['robot'] = {'block': r['robot'], 'options': {}}
    if profile.get('cases'):
        out['cases'] = profile['cases']
    if controller == 'none':
        out['policy'] = {'block': 'vla_only', 'options': dict(profile.get('policy_options') or {})}
        return out
    if r['decision'] == 'code':
        ops = ['python', 'observe'] + (['ack_visual'] if protocol.startswith('window_') else []) + ['finish']
    else:
        ops = ['tool'] + (['vla'] if vla != 'none' else []) + ['observe', 'finish']
    out['llm_backend'] = {'window': {'block': 'command_queue', 'options': {'mailbox': True}},
                          'api': {'block': 'http_chat', 'options': dict(profile.get('llm') or {})},
                          'replay': {'block': 'scripted', 'options': {'replies': profile.get('replies', [])}}}[controller]
    out['policy'] = {'block': 'commands', 'options': dict({'ops': ops, 'require_id': controller == 'window'},
                                                          **(profile.get('policy_options') or {}))}
    return out
