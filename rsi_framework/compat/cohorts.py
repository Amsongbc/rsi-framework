"""S0 inventory: every original cohort, its entries, variants, config template and checks.

`implementations` says how the framework reproduces an entry:
- original: the unchanged script is launched with the same argv, cwd and environment;
- modular: the episode is rebuilt from framework blocks and compared against the
  original under identical inputs (tests/test_equivalence_window.py).
"""
import dataclasses


@dataclasses.dataclass(frozen=True)
class Entry:
    script: str
    purpose: str
    args: tuple
    implementations: tuple = ('original',)
    variants: tuple = ()


@dataclasses.dataclass(frozen=True)
class Cohort:
    name: str
    role: str
    config_template: str | None
    entries: dict
    offline_tests: tuple
    behaviours: tuple
    needs: tuple


COHORTS = {
    'ewam_window_parallel24': Cohort(
        name='ewam_window_parallel24',
        role='Current-window controller: file queue, persistent Python namespace, explicit visual '
             'acknowledgement, full/sparse supervision, EWAM-first hybrid. Main compatibility target.',
        config_template='pyrua-ewam-agent/configs/ewam_window_parallel24.example.json',
        entries={
            'window_session': Entry('window_session.py', 'One episode driven by the window controller',
                                    ('--config', '--task', '--eval-seed', '--layout', '--out', '--variant', '--lane'),
                                    ('original', 'modular'), ('full', 'sparse')),
            'window_control': Entry('window_control.py', 'Submit one command to a running session',
                                    ('--session', '--note', 'op ...')),
        },
        offline_tests=('window_protocol_tests.py',),
        behaviours=('EWAM-first before primitive motion', 'fresh visual acknowledgement after EWAM',
                    'full: propose -> FK preview -> ack -> contiguous prefix',
                    'sparse: ewam_step with host finite/FK screen', 'proposal expires after any motion',
                    'shared native budget', '<=16 model / 48 native actions per segment',
                    '100 Python cells, 5400 s control time', 'window_AI identity unverified, api_requests=0'),
        needs=('PyRUA-Lean', 'RoboDojo/XPolicyLab', 'Isaac Sim runtime', 'EWAM service', 'SAM3')),
    'ewam_comparison': Cohort(
        name='ewam_comparison',
        role='EWAM service, public observation filter, pure EWAM baseline / current-conversation '
             'hybrid / connectivity smoke, runtime asset preparation.',
        config_template=None,
        entries={
            'ewam_service': Entry('ewam_service.py', 'Serve one frozen EWAM model over a Unix socket',
                                  ('--root', '--gpu', '--preallocate-gib')),
            'session': Entry('session.py', 'Baseline / hybrid / smoke episode',
                             ('--mode', '--task', '--layout', '--eval-seed', '--sim-gpu', '--perception-gpu',
                              '--service', '--out', '--timeout'), ('original',), ('baseline', 'hybrid', 'smoke')),
            'control': Entry('control.py', 'Submit a command to a hybrid session', ('--session', '--note', 'op ...')),
            'run_sequence': Entry('run_sequence.py', 'Fixed sequence of sessions', ('--wait-for', '--cases')),
            'prepare_runtime_assets': Entry('prepare_runtime_assets.py', 'Host-specific asset paths', ()),
            'summarize': Entry('summarize.py', 'Summaries of saved verdicts', ()),
            'audit_pairs': Entry('audit_pairs.py', 'Paired initial-observation audit', ()),
            'smoke_ewam': Entry('smoke_ewam.py', 'Real EWAM forward check', ()),
            'smoke_sam3': Entry('smoke_sam3.py', 'Real SAM3 check', ()),
            'smoke_sim': Entry('smoke_sim.py', 'Simulator connectivity check', ()),
        },
        offline_tests=('test_episode_contract.py',),
        behaviours=('EWAM input = 3 RGB + 4 joint keys + instruction', 'stride 3: 16 model -> 48 native',
                    'baseline keeps the benchmark instruction and full chunks', 'shared native budget'),
        needs=('PyRUA-Lean', 'RoboDojo/XPolicyLab', 'Isaac Sim runtime', 'EWAM weights', 'SAM3 (hybrid)')),
    'ewam_doctor_integration': Cohort(
        name='ewam_doctor_integration',
        role='Historical API controller, proposal supervision, warmup experience and frozen harness.',
        config_template='pyrua-ewam-agent/configs/ewam_doctor_integration.example.json',
        entries={
            'run_episode': Entry('run_episode.py', 'One API-planned episode',
                                 ('--config', '--task', '--eval-seed', '--layout', '--out')),
            'run_study': Entry('run_study.py', '9 warmups + 12 paired tests', ()),
            'analyze_study': Entry('analyze_study.py', 'Study analysis', ()),
            'audit_study': Entry('audit_study.py', 'Study audit', ()),
        },
        offline_tests=('test_doctor.py',),
        behaviours=('fixed model gpt-6.1-sol / high', 'warmup-only LLM experience, <=10 lessons',
                    'gate: mean warmup score < 30% -> direct branch without EWAM', 'frozen experience hash'),
        needs=('PyRUA-Lean play loop', 'RoboDojo', 'EWAM service', 'SAM3', 'Responses API key')),
    'ewam_doctor_validation72': Cohort(
        name='ewam_doctor_validation72',
        role='72-episode paired validation with crossed GPU lanes and predeclared statistics.',
        config_template='pyrua-ewam-agent/configs/ewam_doctor_validation72.example.json',
        entries={
            'run_episode': Entry('run_episode.py', 'One API-planned episode',
                                 ('--config', '--task', '--eval-seed', '--layout', '--out')),
            'run_validation': Entry('run_validation.py', 'Run the declared 72 episodes', ('--retry-invalid',)),
            'analyze_validation': Entry('analyze_validation.py', 'Analysis', ()),
            'audit_study': Entry('audit_study.py', 'Audit', ()),
        },
        offline_tests=('test_doctor.py',),
        behaviours=('36 pairs, McNemar + layout-cluster bootstrap', 'frozen prior warmup guidance'),
        needs=('PyRUA-Lean play loop', 'RoboDojo', 'EWAM service', 'SAM3', 'Responses API key')),
    'ewam_doctor_efficiency48': Cohort(
        name='ewam_doctor_efficiency48',
        role='2x2 efficiency ablation: sparse supervision x server context compaction.',
        config_template='pyrua-ewam-agent/configs/ewam_doctor_efficiency48.example.json',
        entries={
            'run_episode': Entry('run_episode.py', 'One API-planned episode',
                                 ('--config', '--task', '--eval-seed', '--layout', '--out')),
            'run_validation': Entry('run_validation.py', 'Run the declared 48 episodes', ('--retry-invalid',)),
            'api_preflight': Entry('api_preflight.py', 'API preflight', ()),
            'analyze_efficiency': Entry('analyze_efficiency.py', 'Analysis', ()),
            'audit_study': Entry('audit_study.py', 'Audit', ()),
        },
        offline_tests=('test_doctor.py', 'test_efficiency.py', 'test_plan.py'),
        behaviours=('original arm payload equals the validation72 transport', 'sparse arm = host FK screen',
                    'compaction keeps the static system contract'),
        needs=('PyRUA-Lean play loop', 'RoboDojo', 'EWAM service', 'SAM3', 'Responses API key')),
}


def entry(cohort, name):
    try:
        return COHORTS[cohort].entries[name]
    except KeyError:
        raise KeyError(f'Unknown original entry {cohort}:{name}') from None


def inventory():
    return {name: {'role': c.role, 'config_template': c.config_template,
                   'entries': {k: dataclasses.asdict(v) for k, v in c.entries.items()},
                   'offline_tests': list(c.offline_tests), 'behaviours': list(c.behaviours),
                   'needs': list(c.needs)} for name, c in COHORTS.items()}
