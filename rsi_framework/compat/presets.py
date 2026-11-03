"""The original cohorts written as readable combinations.

When a readable profile (env / vla / controller / protocol ...) is exactly one of
these, it runs that original entry: unchanged (implementation=original, default) or
block-built where available (modular). The entry's own arguments come from
`original` in the profile or flags, else from the env / VLA defaults (`sources`).
"""
from pathlib import Path

from rsi_framework import ORIGINAL_ROOT
import dataclasses

ORIGINAL_ARGS = {'config': 'config', 'lane': 'lane', 'sim_gpu': 'sim_gpu', 'perception_gpu': 'perception_gpu',
                 'ewam_service': 'service', 'timeout': 'timeout'}


@dataclasses.dataclass(frozen=True)
class Preset:
    name: str
    readable: dict
    cohort: str
    entry: str
    variant: str | None
    original: tuple
    implementations: tuple = ('original',)
    default_protocol: bool = False
    sources: tuple = ()

    def describe(self):
        return {'readable': self.readable, 'runs': f'projects/{self.cohort}/{self.entry}.py'
                + (f' ({self.variant})' if self.variant else ''), 'original_args': list(self.original),
                'implementations': list(self.implementations)}


BASE = {'env': 'robodojo', 'vla': 'ewam', 'tools': 'pyrua'}
WINDOW = (('config', 'env', 'original', 'window_config'), ('lane', 'env', 'original', 'lane'))
SESSION = (('sim_gpu', 'env', 'options', 'simulator_gpu'), ('perception_gpu', 'env', 'options', 'perception_gpu'),
           ('ewam_service', 'vla', 'options', 'service_root'))
PRESETS = (
    Preset('robodojo_ewam_window_full', dict(BASE, controller='window', decision='code', protocol='window_full'),
           'ewam_window_parallel24', 'window_session', 'full', ('config', 'lane'), ('original', 'modular'), sources=WINDOW),
    Preset('robodojo_ewam_window_sparse', dict(BASE, controller='window', decision='code', protocol='window_sparse'),
           'ewam_window_parallel24', 'window_session', 'sparse', ('config', 'lane'), ('original', 'modular'), sources=WINDOW),
    Preset('robodojo_ewam_baseline', dict(BASE, controller='none', protocol='ewam_baseline'),
           'ewam_comparison', 'session', 'baseline', ('sim_gpu', 'perception_gpu', 'ewam_service'),
           default_protocol=True, sources=SESSION),
    Preset('robodojo_ewam_conversation', dict(BASE, controller='window', decision='code', protocol='ewam_tool'),
           'ewam_comparison', 'session', 'hybrid', ('sim_gpu', 'perception_gpu', 'ewam_service'), sources=SESSION),
    Preset('robodojo_ewam_smoke', dict(BASE, controller='none', protocol='smoke'),
           'ewam_comparison', 'session', 'smoke', ('sim_gpu', 'perception_gpu', 'ewam_service'), sources=SESSION),
) + tuple(
    Preset(f'robodojo_ewam_api_{study}', dict(BASE, controller='api_responses', decision='code', protocol='doctor', study=study),
           cohort, 'run_episode', None, ('config',), sources=(('config', 'env', 'original', f'{study}_config'),))
    for study, cohort in (('integration', 'ewam_doctor_integration'), ('validation72', 'ewam_doctor_validation72'),
                          ('efficiency48', 'ewam_doctor_efficiency48')))


def match(readable):
    """The preset this readable combination denotes, or None."""
    for preset in PRESETS:
        wanted = dict(preset.readable)
        if preset.default_protocol and readable.get('protocol') is None:
            wanted.pop('protocol')
        if all(readable.get(k) == v for k, v in wanted.items()) and readable.get('study') == preset.readable.get('study'):
            return preset
    return None


def to_compat(preset, profile, implementation):
    if implementation not in preset.implementations:
        raise ValueError(f'{preset.name} runs as {list(preset.implementations)}, not {implementation}')
    if profile.get('robot') not in (None, 'arx_x5', 'arx_x5_bimanual'):
        raise ValueError(f'{preset.name} is the ARX X5 cohort; robot {profile["robot"]} does not apply')
    if profile.get('memory') not in (None, 'off'):
        raise ValueError(f'{preset.name} takes its Memory from the original config, not from memory=')
    given = {k: v for k, v in (profile.get('original') or {}).items() if v not in (None, '')}
    found = profile.get('_defaults') or {}
    for key, block, section, field in preset.sources:
        value = ((found.get(block) or {}).get(section) or {}).get(field)
        if key not in given and value not in (None, ''):
            given[key] = value
    missing = [k for k in preset.original if given.get(k) in (None, '')]
    if missing:
        raise ValueError(f'{preset.name} needs original arguments: ' + ', '.join('--' + k.replace('_', '-') for k in missing))
    render = None
    template = ORIGINAL_ROOT / 'configs' / f'{preset.cohort}.example.json'
    if 'config' in given and not Path(given['config']).is_file():
        if not (str(given['config']).endswith('.local.json') and template.is_file()):
            raise ValueError(f'{preset.name}: original config {given["config"]} does not exist')
        render = {'template': str(template), 'target': str(given['config'])}
    unknown = [k for k in given if k not in ORIGINAL_ARGS]
    if unknown:
        raise ValueError(f'Unknown original arguments {unknown}; known: {sorted(ORIGINAL_ARGS)}')
    args = {ORIGINAL_ARGS[k]: v for k, v in given.items() if v not in (None, '')}
    return {'name': profile.get('name', preset.name), 'mode': 'compat', 'preset': preset.name,
            'allow_stand_ins': profile.get('allow_stand_ins', False),
            'compat': {'cohort': preset.cohort, 'entry': preset.entry, 'variant': preset.variant,
                       'implementation': implementation, 'args': args},
            'services': {'render': render, 'ewam': (found.get('vla') or {}).get('service')}}
