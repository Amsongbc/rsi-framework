"""RoboDojo through the PyRUA host, with the original run-local launcher change.

Boot follows window_session.Session.boot: patch the host's lay_out (runtime asset
link, single simulator GPU), boot the evaluation client, wrap the native env in the
original RecordedEnv (shared native budget, owner counts, video and native actions).
"""
import json
import os
from pathlib import Path
import sys

from rsi_framework.compat import paths
from rsi_framework.core.io import write_json
from rsi_framework.core.registry import register
from rsi_framework.environments.base import EnvironmentAdapter


class PyRUARoboDojo(EnvironmentAdapter):
    robot = 'arx_x5'

    def __init__(self, *, simulator_gpu, source_root=None, runtime_root=None, suite='arx_x5',
                 boot_timeout_s=900):
        self.simulator_gpu = int(simulator_gpu)
        self.source_root = source_root or str(paths.projects_root() / 'RoboDojo')
        self.runtime_root = Path(runtime_root) if runtime_root else paths.runtime_root()
        self.suite = suite
        self.boot_timeout_s = boot_timeout_s
        self.evidence = 'real'

    def describe(self):
        return {'block': 'robodojo', 'robot': self.robot, 'suite': self.suite, 'simulator_gpu': self.simulator_gpu,
                'source_root': self.source_root, 'evidence': self.evidence}

    def prepare(self, run):
        run.toolkit = None
        run.daemons = []

    def boot(self, run):
        paths.ensure_importable()
        from recording import RecordedEnv
        import pyrualean
        from pyrualean.hosts import xpolicylab_robodojo as base
        from pyrualean.hosts import xpolicylab_robodojo_rgb as host
        if getattr(pyrualean, 'IS_TEST_DOUBLE', False):
            self.evidence = 'software_synthetic'
        original = base.lay_out
        gpu, assets, out = self.simulator_gpu, self.runtime_root / 'runtime_assets/Assets', run.out

        def layout(*a, **kw):
            bench = original(*a, **kw)
            (bench / 'Assets').unlink()
            (bench / 'Assets').symlink_to(assets, target_is_directory=True)
            script = bench / 'scripts/eval_policy.sh'
            old = script.read_text()
            new = old.replace('KIT_ARGS=""', f'KIT_ARGS=" --/renderer/activeGpu={gpu} --/renderer/multiGpu/enabled=false"')
            if new == old:
                raise RuntimeError('Could not set run-local simulator GPU')
            script.write_text(new)
            write_json(out / 'client_launcher_change.json', dict(scope='Run-local asset path and GPU launcher only',
                                                                 sim_gpu=gpu, multi_gpu=False))
            return bench
        base.lay_out = layout
        cell = host.Cell(suite=self.suite, task=run.case['task'], seed=run.case['layout'], cuda_device=gpu)
        run.toolkit, run.daemons = host.boot(cell, run.out, env_only=True, source_root=self.source_root,
                                             eval_seed=run.case['eval_seed'], boot_timeout_s=self.boot_timeout_s)
        run.bridge = run.toolkit._primitives.env.bridge
        run.env = RecordedEnv(run.toolkit._primitives.env, run.out)
        run.toolkit._primitives.env = run.env

    def booted(self, run):
        return run.toolkit is not None

    def context_entries(self, run):
        return {'env': run.env, 'toolkit': run.toolkit}

    def processes_alive(self, run):
        return all(getattr(d, 'alive', lambda: True)() for d in run.daemons)

    def finish(self, run):
        run.toolkit.conclude()
        return run.bridge.verdict(timeout_s=60)

    def outcome(self, run, verdict):
        """The official RoboDojo client result files, judged as window_session.Session.close does."""
        raw = self.official_results(run.out)
        details = [d for row in raw for d in row['value'].get('details', {}).values() if d.get('layout_id') == run.case['layout']]
        valid = bool(not run.errors and verdict is not None and len(raw) == 1 and len(details) == 1
                     and raw[0]['value'].get('eval_time') == 1 and bool(details[0]['success']) == bool(verdict.get('success')))
        return raw, details, valid

    def after_motion(self, run):
        run.toolkit.state.record(run.env.get_obs(), run.env.get_status())

    def robot_backend(self, run):
        from pyrualean.hosts import xpolicylab_robodojo_rgb as host
        return host.make_backend(run.toolkit)

    def stop(self, run):
        from pyrualean.hosts.common import stop_all
        stop_all(run.toolkit, run.daemons)

    @staticmethod
    def official_results(out):
        return [dict(path=str(p), value=json.loads(p.read_text())) for p in (Path(out) / 'bench/eval_result').glob('**/_result.json')]


ORACLE_VARIABLES = {'ROBODOJO_ORACLE_STATE', 'ROBODOJO_IK_FEEDBACK', 'ROBODOJO_TASK_STEP_LIM', 'ROBODOJO_STEP_LIM',
                    'ROBODOJO_FIXED_INSTRUCTION', 'ROBODOJO_ORACLE_INSTRUCTION'}


def prepare_process(perception_gpu, keep_proxy):
    """Process environment of the original entries for a framework episode.

    Same variables as window_session.configure_environment. keep_proxy=True keeps
    HTTP(S) proxies for an API backend, as the historical API cohorts do
    (NO_PROXY limited to localhost); keep_proxy=False removes them like the window
    cohort. Hidden-state RoboDojo variables are always removed.
    """
    projects, runtime = paths.projects_root(), paths.runtime_root()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(perception_gpu)
    os.environ['PATH'] = str(runtime / '.venv-sim/bin') + os.pathsep + os.environ['PATH']
    os.environ['PYTHONPATH'] = os.pathsep.join([str(projects / 'PyRUA-Lean/src'), os.environ.get('PYTHONPATH', '')]).rstrip(os.pathsep)
    no_proxy = 'localhost,127.0.0.1,::1' if keep_proxy else '*'
    os.environ.update(OMNI_KIT_ACCEPT_EULA='YES', ROBODOJO_MAX_BASH_RETRIES='1',
                      SAM3_CHECKPOINT_PATH=str(projects / 'RoboDojo-resources/backbones/sam3/sam3.pt'),
                      HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                      OPENBLAS_NUM_THREADS='8', TRITON_CACHE_DIR=str(runtime / '.triton-cache'),
                      TORCH_EXTENSIONS_DIR=str(runtime / '.torch-extensions'),
                      VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json', XDG_RUNTIME_DIR=str(runtime / 'xdg'),
                      TMPDIR=str(runtime / 'runtime_tmp'), XDG_CACHE_HOME=str(runtime / '.cache'),
                      CUDA_CACHE_PATH=str(runtime / '.cuda-cache'), NO_PROXY=no_proxy, no_proxy=no_proxy)
    (runtime / 'runtime_tmp').mkdir(parents=True, exist_ok=True)
    for key in list(os.environ):
        if key in ORACLE_VARIABLES or (not keep_proxy and key.lower().endswith('_proxy') and key.lower() != 'no_proxy'):
            os.environ.pop(key)
    paths.ensure_importable()
    for path in (str(projects / 'PyRUA-Lean/src'),):
        if path not in sys.path:
            sys.path.insert(0, path)


@register('environment', 'robodojo', aliases=('pyrua_robodojo',), description='RoboDojo via the PyRUA host, original RecordedEnv')
def make(**options):
    return PyRUARoboDojo(**options)
