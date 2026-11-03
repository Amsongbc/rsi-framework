"""Command line: python -m rsi_framework <command> ...

  baseline [--projects DIR]          original files unchanged? (fingerprint)
  inventory                          original cohorts, entries, variants, behaviours
  blocks                             registered blocks by kind
  suites [--upstream DIR]            original offline tests in a temporary copy
  compat COHORT ENTRY [--implementation original|modular] -- ARGS
                                     run an original entry (unchanged) or its block-built version
  presets                            original cohorts written as readable combinations
  run (--profile P | --env E --vla V --controller C --protocol P ...) --task T --eval-seed S --out DIR
                                     one episode; an original combination runs the original entry, e.g.
                                     run --env robodojo --vla ewam --controller window --protocol window_full
                                         --config cfg.json --lane A --task classify_objects --eval-seed 0 --out runs/x
  check-profile (P | readable flags) show what a profile resolves to
  equivalence REF CAND               compare two run directories under the fixed rules
  dataset --episodes DIR... --out DIR [--selection success_only|valid_all|all_executed]
  round --plan PLAN --out DIR        one improvement round (collect, learn, evaluate, release)
  release list|activate|rollback --store DIR [--id ID] [--reason TEXT]
"""
import argparse
import json
from pathlib import Path
import sys


READABLE_FLAGS = ('env', 'robot', 'vla', 'tools', 'controller', 'decision', 'protocol', 'memory', 'study',
                  'implementation')
ORIGINAL_FLAGS = ('config', 'lane', 'sim_gpu', 'perception_gpu', 'ewam_service', 'timeout')


def readable_flags(p):
    p.add_argument('--profile', type=Path, help='profile file (readable or block form)')
    for name in READABLE_FLAGS:
        p.add_argument('--' + name.replace('_', '-'))
    for name in ORIGINAL_FLAGS:
        p.add_argument('--' + name.replace('_', '-'), help='argument of the original entry')
    p.add_argument('--plugin', action='append', default=[], help='module registering extra blocks')
    p.add_argument('--allow-stand-ins', action='store_true')


def build_profile(parser, a):
    from rsi_framework.core import compose
    from rsi_framework.core.config import load_profile
    profile = load_profile(a.profile) if a.profile else {}
    flags = {k: getattr(a, k) for k in READABLE_FLAGS if getattr(a, k) is not None}
    original = {k: getattr(a, k) for k in ORIGINAL_FLAGS if getattr(a, k) is not None}
    if flags or original:
        if profile and not compose.is_readable(profile):
            parser.error('readable flags combine only with a readable profile')
        profile.update(flags)
        profile['original'] = dict(profile.get('original') or {}, **original)
    if a.plugin:
        profile['plugins'] = list(profile.get('plugins', [])) + a.plugin
    if a.allow_stand_ins:
        profile['allow_stand_ins'] = True
    if not profile:
        parser.error('give --profile or readable flags such as --env robodojo --vla ewam')
    try:
        return compose.resolve(profile)
    except ValueError as exc:
        parser.error(str(exc))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough = []
    if '--' in argv:
        index = argv.index('--')
        argv, passthrough = argv[:index], argv[index + 1:]
    parser = argparse.ArgumentParser(prog='python -m rsi_framework', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('baseline'); p.add_argument('--projects', type=Path)
    sub.add_parser('inventory')
    sub.add_parser('blocks')
    p = sub.add_parser('suites'); p.add_argument('--upstream', type=Path)
    p = sub.add_parser('compat'); p.add_argument('cohort'); p.add_argument('entry')
    p.add_argument('--implementation', choices=['original', 'modular'], default='original')
    sub.add_parser('presets')
    p = sub.add_parser('run')
    readable_flags(p)
    p.add_argument('--task', help='default: the env file'); p.add_argument('--eval-seed', type=int)
    p.add_argument('--layout', type=int)
    p.add_argument('--out', type=Path, help='default: runs/<combination>-<time>'); p.add_argument('--release', type=Path)
    p.add_argument('--no-start-services', action='store_true', help='do not start EWAM for an original run')
    p = sub.add_parser('check-profile')
    readable_flags(p)
    p = sub.add_parser('equivalence'); p.add_argument('reference', type=Path); p.add_argument('candidate', type=Path)
    p = sub.add_parser('dataset'); p.add_argument('--episodes', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--selection', default='success_only')
    p = sub.add_parser('round'); p.add_argument('--plan', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('release'); p.add_argument('action', choices=['list', 'activate', 'rollback'])
    p.add_argument('--store', type=Path, required=True); p.add_argument('--id'); p.add_argument('--reason', default='manual')
    a = parser.parse_args(argv)

    if a.command == 'baseline':
        from rsi_framework.compat import baseline
        report = baseline.check(projects=a.projects) if a.projects else baseline.check()
        print(json.dumps(report, indent=2))
        return 0 if report['passed'] else 1
    if a.command == 'inventory':
        from rsi_framework.compat.cohorts import inventory
        print(json.dumps(inventory(), indent=2, ensure_ascii=False))
        return 0
    if a.command == 'blocks':
        from rsi_framework.core.registry import REGISTRY, load_builtin
        load_builtin()
        for entry in REGISTRY.entries():
            print(f'{entry.kind:12s} {entry.name:20s} [{entry.evidence}] {entry.description}')
        return 0
    if a.command == 'suites':
        from rsi_framework.compat import original_suites
        report = original_suites.run(a.upstream) if a.upstream else original_suites.run()
        print(json.dumps(report, indent=2))
        return 0 if report['passed'] else 1
    if a.command == 'compat':
        from rsi_framework.compat import launch
        from rsi_framework.compat.cohorts import entry
        if a.implementation == 'modular':
            if 'modular' not in entry(a.cohort, a.entry).implementations:
                parser.error(f'{a.cohort}:{a.entry} has no modular implementation yet; use original')
            from rsi_framework.compat import window_v1
            return window_v1.main(passthrough)
        launch.exec_original(a.cohort, a.entry, passthrough)
    if a.command == 'presets':
        from rsi_framework.compat.presets import PRESETS
        from rsi_framework.core.compose import normalize
        for preset in PRESETS:
            full = normalize(dict(preset.readable))
            flags = dict(preset.readable)
            for key in ('tools', 'decision', 'protocol', 'controller'):
                trial = {k: v for k, v in flags.items() if k != key}
                if key in flags and all(normalize(trial).get(k) == full.get(k) for k in preset.readable
                                        if not (preset.default_protocol and k == 'protocol')):
                    flags = trial
            if preset.default_protocol:
                flags.pop('protocol', None)
            impl = '' if preset.implementations == ('original',) else '  [--implementation modular]'
            print(f"{preset.name}\n  {' '.join(f'--{k} {v}' for k, v in flags.items())}{impl}\n"
                  f"  -> {preset.describe()['runs']}")
        return 0
    if a.command == 'check-profile':
        profile = build_profile(parser, a)
        if profile.get('mode') == 'compat':
            print(json.dumps({k: v for k, v in profile.items() if not k.startswith('_')}, indent=2))
            return 0
        from rsi_framework.core.assembly import assemble
        print(json.dumps(assemble(profile).describe(), indent=2, default=str))
        return 0
    if a.command == 'run':
        from rsi_framework.core.episode import run_episode
        profile = build_profile(parser, a)
        case = dict(profile.pop('case_defaults', None) or {})
        case.update({k: v for k, v in dict(task=a.task, eval_seed=a.eval_seed, layout=a.layout).items() if v is not None})
        missing = [k for k in ('task', 'eval_seed', 'layout') if k not in case]
        if missing:
            parser.error(f'no default {missing} for this environment; pass --' + ' --'.join(m.replace('_', '-') for m in missing))
        if a.out is None:
            import time
            a.out = Path('runs') / f"{profile.get('preset') or profile.get('name', 'run')}-{time.strftime('%Y%m%d-%H%M%S')}"
            print(f'output: {a.out}', flush=True)
        if profile.get('mode') == 'compat':
            from rsi_framework.compat import launch, profiles, services, window_v1
            compat = profile['compat']
            started = services.prepare(profile, start=not a.no_start_services)
            args = profiles.argv(profile, case, a.out)
            if compat['cohort'] == 'ewam_window_parallel24':
                print(f"controller: python projects/ewam_window_parallel24/window_control.py --session {a.out} "
                      "--note '<what you see>' python --file cell.py", flush=True)
            try:
                if compat.get('implementation', 'original') == 'modular':
                    return window_v1.main(args + (['--allow-stand-ins'] if profile.get('allow_stand_ins') else []))
                if not started:
                    launch.exec_original(compat['cohort'], compat['entry'], args)
                return launch.run(compat['cohort'], compat['entry'], args)
            finally:
                services.stop(started)
        release = json.loads(a.release.read_text()) if a.release else None
        if release is not None:
            profile = release['profile']
        result = run_episode(profile, case, a.out, release=release)
        print(json.dumps({k: result.get(k) for k in ('profile', 'mode', 'valid', 'success', 'official_score',
                                                     'stop_reason', 'evidence')}), flush=True)
        return 0 if result['valid'] else 1
    if a.command == 'equivalence':
        from rsi_framework.evaluation.equivalence import compare
        report = compare(a.reference, a.candidate)
        print(json.dumps(report, indent=2))
        return 0 if report['equal'] else 1
    if a.command == 'dataset':
        from rsi_framework.data.dataset import build
        print(json.dumps(build(a.episodes, a.out, selection=a.selection), indent=2))
        return 0
    if a.command == 'round':
        from rsi_framework.core.iteration import run_round
        print(json.dumps(run_round(a.plan, a.out), indent=2, default=str))
        return 0
    if a.command == 'release':
        from rsi_framework.releases.manifest import ReleaseStore
        store = ReleaseStore(a.store)
        if a.action == 'list':
            print(json.dumps(store.state(), indent=2))
        elif a.action == 'rollback':
            print(json.dumps(store.rollback(a.reason), indent=2))
        else:
            parser.error('activate needs an evaluation; use the round command, which activates only on a passed gate')
        return 0
    return 2


if __name__ == '__main__':
    sys.exit(main())
