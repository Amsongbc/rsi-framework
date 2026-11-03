"""Run one episode of a (non-compat) profile in this process.

Episodes are one per process, like the original sessions: the environment adapter
patches host module state at boot. Orchestrators call this through the CLI in a
subprocess for each episode.
"""
from pathlib import Path

from rsi_framework.core.assembly import assemble
from rsi_framework.core.runner import EpisodeRunner


def check_case(profile, case):
    rules = profile.get('cases') or {}
    for field, key in (('tasks', 'task'), ('eval_seeds', 'eval_seed'), ('layouts', 'layout')):
        if field in rules and case[key] not in rules[field]:
            raise ValueError(f'{case[key]} outside configured {field}')


def run_episode(profile, case, out, release=None):
    check_case(profile, case)
    environment = profile['environment']
    from rsi_framework.core.registry import REGISTRY, load_builtin, load_plugins
    load_builtin()
    load_plugins(profile.get('plugins'))
    if REGISTRY.canonical('environment', environment['block']) == 'robodojo':
        from rsi_framework.environments.pyrua_robodojo import prepare_process
        llm = profile.get('llm_backend') or {}
        prepare_process(environment.get('perception_gpu', environment.get('options', {}).get('simulator_gpu', 0)),
                        keep_proxy=llm.get('block') == 'http_chat')
    assembly = assemble(profile)
    if release is not None:
        from rsi_framework.releases.manifest import verify_assembly
        problems = verify_assembly(release, assembly)
        if problems:
            raise RuntimeError('Assembly does not match the release manifest: ' + '; '.join(problems))
    return EpisodeRunner(assembly, case, Path(out)).run()
