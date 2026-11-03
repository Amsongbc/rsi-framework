"""compat profiles: a profile file that names an original entry and its arguments.

{"mode": "compat", "compat": {"cohort": ..., "entry": ..., "implementation": "original"|"modular",
                              "variant": ..., "args": {...}}}

The argv is built exactly as the original entry expects; `--out` and the case come
from the command line. implementation=original executes the unchanged script.
"""
from rsi_framework.compat.cohorts import entry

CASE_ARGS = {
    ('ewam_window_parallel24', 'window_session'): lambda c, case, out: [
        '--config', c['args']['config'], '--task', case['task'], '--eval-seed', str(case['eval_seed']),
        '--layout', str(case['layout']), '--out', str(out), '--variant', c['variant'], '--lane', c['args']['lane']],
    ('ewam_comparison', 'session'): lambda c, case, out: [
        '--mode', c['variant'], '--task', case['task'], '--layout', str(case['layout']),
        '--eval-seed', str(case['eval_seed']), '--out', str(out)] + sum(
            ([f'--{k.replace("_", "-")}', str(v)] for k, v in c['args'].items()), []),
}
for cohort in ('ewam_doctor_integration', 'ewam_doctor_validation72', 'ewam_doctor_efficiency48'):
    CASE_ARGS[(cohort, 'run_episode')] = lambda c, case, out: [
        '--config', c['args']['config'], '--task', case['task'], '--eval-seed', str(case['eval_seed']),
        '--layout', str(case['layout']), '--out', str(out)]


def argv(profile, case, out):
    c = profile['compat']
    key = (c['cohort'], c['entry'])
    spec = entry(*key)
    if spec.variants and c.get('variant') not in spec.variants:
        raise ValueError(f'{key} variant must be one of {spec.variants}')
    if c.get('implementation', 'original') not in spec.implementations:
        raise ValueError(f'{key} has implementations {spec.implementations}')
    if key not in CASE_ARGS:
        raise ValueError(f'{key} is not an episode entry; run it with: python -m rsi_framework compat {key[0]} {key[1]} -- ...')
    return CASE_ARGS[key](c, case, out)
