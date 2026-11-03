"""window_session.py rebuilt from framework blocks (implementation: modular).

The original experiment config stays the source of truth: blocks are configured
from it exactly as window_session.Session uses it. Only the recorder is specific
to this cohort (source snapshot, run_config.json and episode_summary.json fields).
tests/test_equivalence_window.py runs this and the original on identical inputs and
compares every output file.

Known record differences, all outside the control path:
- implementation_snapshot/ and source_manifest.json additionally list the
  framework sources that produced the run (original rows first, unchanged);
- traceback frames that point into the runner show framework file paths instead of
  window_session.py; exception types, messages and the original-code frames match.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

from rsi_framework import REPO_ROOT
from rsi_framework.compat import baseline, paths
from rsi_framework.core.assembly import Assembly
from rsi_framework.core.io import sha256_file, write_json
from rsi_framework.core.recorders import framework_sources, official_outcome, snapshot
from rsi_framework.core.registry import register
from rsi_framework.core.runner import EpisodeRunner
from rsi_framework.environments.pyrua_robodojo import PyRUARoboDojo
from rsi_framework.llm_backends.command_queue import CommandQueue
from rsi_framework.memory.manager import LegacyDoctorMemory
from rsi_framework.policies.commands import WINDOW_OPS, CommandLoop
from rsi_framework.protocols.window import WindowProtocol
from rsi_framework.robots.arx_x5 import ARX_X5
from rsi_framework.tools.pyrua import PyRUATools
from rsi_framework.vla_backends.ewam import EwamService, episode_seed


class WindowV1Recorder:
    kind = 'recorder'
    evidence = 'real'

    def __init__(self, *, config, variant, lane):
        self.config, self.variant, self.lane = config, variant, lane

    def describe(self):
        return {'block': 'window_v1'}

    def prepare(self, run):
        import window_session
        projects, root, runtime = window_session.PROJECTS, window_session.ROOT, window_session.RUNTIME
        sources = list(root.glob('*.py')) + [runtime / 'session.py', runtime / 'ewam_service.py']
        sources += list((projects / 'RoboDojo/XPolicyLab/policy/EWAM').rglob('*.py'))
        library = projects / 'PyRUA-Lean/src/pyrualean'
        sources += list(library.rglob('*.py')) + list((library / 'knowledge').glob('*.md'))
        rows = []
        for path in sources:
            dest = run.out / 'implementation_snapshot' / path.relative_to(projects)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            rows.append(dict(path=str(path), snapshot=str(dest.relative_to(run.out)), sha256=sha256_file(path)))
        for path in framework_sources():
            dest = run.out / 'implementation_snapshot' / path.relative_to(REPO_ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            rows.append(dict(path=str(path), snapshot=str(dest.relative_to(run.out)), sha256=sha256_file(path)))
        write_json(run.out / 'source_manifest.json', rows)
        write_json(run.out / 'experiment_config.json', self.config)
        metadata = Path(self.config['service_root']) / 'model.json'
        if not metadata.exists():
            raise FileNotFoundError(f'Exact EWAM service metadata required: {metadata}')
        shutil.copyfile(metadata, run.out / 'ewam_service_metadata.json')

    def run_config(self, run):
        case = run.case
        write_json(run.out / 'run_config.json', dict(
            task=case['task'], eval_seed=case['eval_seed'], layout=case['layout'], variant=self.variant, lane=self.lane,
            planner='window_AI', requested_planner='window_AI', exact_model='unverified', api_requests=0,
            token_usage='unavailable', native_limit=run.env.initial_limit, max_cells=100, wall_timeout_s=5400,
            max_model_actions=16, max_native_prefix=48, EWAM_seed=episode_seed(case['eval_seed'], case['layout'])))

    def snapshot(self, run):
        return snapshot(run)

    def summarize(self, run, verdict, policy_end):
        import time
        raw, details, valid = official_outcome(run, verdict)
        case = run.case
        summary = dict(task=case['task'], eval_seed=case['eval_seed'], layout=case['layout'], variant=self.variant,
                       lane=self.lane, planner='window_AI', requested_planner='window_AI', exact_model='unverified',
                       api_requests=0, token_usage='unavailable', valid=valid,
                       success=bool(details[0]['success']) if valid else None,
                       official_score=details[0].get('score') if valid else None,
                       outcome_class='valid_task_success' if valid and details[0]['success'] else
                       'valid_task_failure' if valid else 'technical_interruption',
                       client_verdict=verdict, client_results=raw, stop_reason=run.stop_reason, errors=run.errors,
                       python_cells=run.cells, ewam_calls=run.context['ewam_calls'],
                       ewam_attempts=run.context['ewam_attempts'],
                       native_actions_issued=getattr(getattr(run, 'env', None), 'issued_actions', 0),
                       native_actions_by_owner=getattr(getattr(run, 'env', None), 'actions_by_owner', {}),
                       first_native_action_owner=getattr(getattr(run, 'env', None), 'first_native_owner', None),
                       startup_s=None if run.ready_at is None else run.ready_at - run.started,
                       control_wall_s=None if run.ready_at is None else policy_end - run.ready_at,
                       total_wall_s=time.monotonic() - run.started)
        write_json(run.out / 'episode_summary.json', summary)
        write_json(run.out / 'result.json', summary)
        return summary


def assemble(config, variant, lane, allow_stand_ins=False):
    paths.ensure_importable()
    import window_session
    blocks = dict(
        environment=PyRUARoboDojo(simulator_gpu=config['gpus']['simulator'],
                                  source_root=config.get('source_root', str(window_session.PROJECTS / 'RoboDojo')),
                                  runtime_root=window_session.RUNTIME),
        robot=ARX_X5,
        vla=EwamService(service_root=config['service_root']),
        llm_backend=None,
        tools=PyRUATools(),
        memory=LegacyDoctorMemory(value=config.get('doctor', {}).get('frozen_memory')),
        protocol=WindowProtocol(variant=variant, efficiency=config.get('efficiency', {})),
        recorder=WindowV1Recorder(config=config, variant=variant, lane=lane))
    blocks['llm_backend'] = CommandQueue(mailbox=True)
    blocks['policy'] = CommandLoop(channel=blocks['llm_backend'], ops=WINDOW_OPS, require_note=True,
                                   require_id=True, max_cells=100)
    profile = {'name': f'compat_window_{variant}', 'mode': 'compat', 'allow_stand_ins': allow_stand_ins,
               'compat': {'cohort': paths.WINDOW, 'entry': 'window_session', 'implementation': 'modular',
                          'variant': variant, 'lane': lane}}
    return Assembly(profile, blocks)


def parse(argv):
    p = argparse.ArgumentParser(description='window_session.py rebuilt from framework blocks')
    p.add_argument('--config', type=Path, required=True); p.add_argument('--task', required=True)
    p.add_argument('--eval-seed', type=int, required=True); p.add_argument('--layout', type=int, default=0)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--variant', choices=['full', 'sparse'], required=True)
    p.add_argument('--lane', choices=['A', 'B', 'C'], required=True)
    p.add_argument('--allow-stand-ins', action='store_true')
    return p, p.parse_args(argv)


def main(argv=None):
    p, a = parse(argv)
    config = json.loads(a.config.read_text())
    if a.lane in config.get('lanes', {}):
        lane = config['lanes'][a.lane]
        gpu = lane.get('gpu')
        override = {k: lane[k] for k in ('simulator', 'sam3') if k in lane}
        if gpu is not None:
            override = dict({'simulator': gpu, 'sam3': gpu}, **override)
        config['gpus'] = dict(config.get('gpus', {}), **lane.get('gpus', {}), **override)
        if 'service_root' in lane:
            config['service_root'] = lane['service_root']
    for field, value in [('tasks', a.task), ('eval_seeds', a.eval_seed), ('layouts', a.layout)]:
        if field in config and value not in config[field]:
            p.error(f'{value} outside configured {field}')
    baseline.require_unchanged(projects=paths.original_projects())
    paths.ensure_importable()
    from window_session import configure_environment
    configure_environment(config)
    assembly = assemble(config, a.variant, a.lane, allow_stand_ins=a.allow_stand_ins)
    case = dict(task=a.task, eval_seed=a.eval_seed, layout=a.layout)
    result = EpisodeRunner(assembly, case, a.out).run()
    print(json.dumps(result), flush=True)
    return 0 if result['valid'] else 1


@register('recorder', 'window_v1', description='Original window_session records (compat modular)')
def make_recorder(**options):
    return WindowV1Recorder(**options)


if __name__ == '__main__':
    sys.exit(main())
