"""One improvement round: P0 -> D1 -> (Memory candidates, practice) -> (data, train) ->
candidate P1 -> Memory revalidation -> paired evaluation -> release or keep P0 -> D2.

Each episode runs in its own process from an exact profile file. Memory and VLA
learning are separate switches read from the profile: Memory `learn` and/or
`learning.enabled`. With both off the round only collects D1 and D2 and proves
nothing was trained. Results carry the evidence labels of the blocks that made
them; stand-in rounds can never be reported as RoboDojo results.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from rsi_framework import REPO_ROOT
from rsi_framework.core.assembly import assemble
from rsi_framework.core.io import write_json
from rsi_framework.core.registry import REGISTRY
from rsi_framework.data import dataset as datasets
from rsi_framework.evaluation.candidates import compare
from rsi_framework.memory import manager as memory
from rsi_framework.releases.manifest import ReleaseStore, make_release


def case_name(case):
    return f"{case['task']}_s{case['eval_seed']}_l{case['layout']}"


class Round:
    def __init__(self, plan, out, python=None):
        self.plan = plan
        self.out = Path(out).resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.python = python or sys.executable
        self.store = ReleaseStore(plan['store'])
        self.services = []
        self.log = []

    # ---- episodes -------------------------------------------------------------------
    def episodes(self, profile, cases, label, release=None):
        profile_path = self.out / 'profiles' / f'{label}.json'
        profile_path.parent.mkdir(exist_ok=True)
        write_json(profile_path, profile)
        summaries = []
        for case in cases:
            out = self.out / 'episodes' / label / case_name(case)
            out.parent.mkdir(parents=True, exist_ok=True)
            argv = [self.python, '-B', '-m', 'rsi_framework', 'run', '--profile', str(profile_path),
                    '--task', case['task'], '--eval-seed', str(case['eval_seed']), '--layout', str(case['layout']),
                    '--out', str(out)]
            if release is not None:
                argv += ['--release', str(self.store.path(release['id']))]
            env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get('PYTHONPATH')])))
            with (out.parent / f'{case_name(case)}.log').open('w') as log:
                subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=self.plan.get('episode_timeout_s', 7200))
            path = out / 'episode_summary.json'
            summaries.append(json.loads(path.read_text()) if path.exists() else
                             {'valid': False, 'success': None, 'official_score': None, 'task': case['task'],
                              'eval_seed': case['eval_seed'], 'layout': case['layout'], 'stop_reason': 'no_summary'})
        self.log.append({'stage': label, 'episodes': [str(self.out / 'episodes' / label / case_name(c)) for c in cases]})
        return summaries

    def episode_dirs(self, label):
        return sorted(p for p in (self.out / 'episodes' / label).iterdir() if p.is_dir())

    # ---- services -----------------------------------------------------------------
    def serve(self, command, checkpoint, label):
        base = Path(self.plan.get('service_dir') or self.out)
        base.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix=f'svc_{label}_', dir=base)) / 's'
        argv = [part.format(root=str(root), checkpoint=str(checkpoint), python=self.python) for part in command]
        process = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + self.plan.get('service_start_timeout_s', 1800)
        while not (root / 'ewam.sock').exists():
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f'Candidate VLA service {label} did not start')
            time.sleep(0.1)
        self.services.append(process)
        return root

    def stop_services(self):
        for process in self.services:
            process.terminate()
            process.wait(timeout=30)

    # ---- the round ----------------------------------------------------------------
    def run(self):
        try:
            return self._run()
        finally:
            self.stop_services()

    def _run(self):
        plan, cases = self.plan, self.plan['cases']
        p0 = self.store.active()
        if p0 is None:
            from rsi_framework.core.compose import resolve
            from rsi_framework.core.config import load_profile
            profile = resolve(load_profile(plan['profile']))
            p0 = make_release(profile, assemble(profile), extras={'bootstrap': True})
            self.store.save(p0)
            self.store.activate(p0['id'], bootstrap=True)
        base = p0['profile']
        report = {'P0': p0['id'], 'evidence': set()}
        d1 = self.episodes(base, cases['collect'], 'D1', release=p0)
        report['D1'] = [{k: s.get(k) for k in ('task', 'eval_seed', 'layout', 'valid', 'success', 'official_score')} for s in d1]
        candidate = copy.deepcopy(base)
        extras, revalidation = {}, None

        mem = base.get('memory') or {'block': 'off'}
        if mem.get('block') == 'learn':
            report['memory'] = self.learn_memory(base, candidate)
            extras['memory'] = report['memory']
        learning = base.get('learning') or {}
        if learning.get('enabled'):
            report['vla'] = self.learn_vla(base, candidate, learning)
            extras['vla_learning'] = report['vla']
            if report['vla'].get('candidate_service') and (candidate.get('memory') or {}).get('block') in ('frozen', 'learn'):
                revalidation = self.revalidate_memory(candidate)
                report['memory_revalidation'] = revalidation
        else:
            trainer = learning.get('trainer') or {'block': 'none'}
            report['vla'] = {'enabled': False, 'trainer': trainer.get('block', 'none'), 'training_calls': 0}

        if candidate == base:
            report['decision'] = {'released': False, 'reason': 'no candidate change'}
        else:
            evaluation = compare(self.episodes(base, cases['evaluate'], 'eval_P0', release=p0),
                                 self.episodes(candidate, cases['evaluate'], 'eval_P1'),
                                 (learning.get('evaluation') or {}).get('rule') or plan.get('evaluation_rule'))
            report['evaluation'] = evaluation
            release = make_release(candidate, assemble(candidate), parent=p0['id'],
                                   extras=dict(extras, evaluation=evaluation, memory_revalidation=revalidation))
            self.store.save(release)
            if evaluation['passed'] and (revalidation is None or revalidation['passed']):
                self.store.activate(release['id'], evaluation=evaluation, memory_revalidation=revalidation)
                report['decision'] = {'released': True, 'release': release['id']}
            else:
                self.store.reject(release['id'], 'evaluation or Memory revalidation did not pass', evaluation)
                report['decision'] = {'released': False, 'release': release['id'], 'reason': 'gate not passed'}
        active = self.store.active()
        report['active_after'] = active['id']
        d2 = self.episodes(active['profile'], cases['collect'], 'D2', release=active)
        report['D2'] = [{k: s.get(k) for k in ('task', 'eval_seed', 'layout', 'valid', 'success', 'official_score')} for s in d2]
        for path in (self.out / 'episodes').rglob('episode_summary.json'):
            report['evidence'].add(json.loads(path.read_text()).get('evidence', 'unlabelled'))
        report['evidence'] = sorted(report['evidence'])
        report['stages'] = self.log
        write_json(self.out / 'round_report.json', report)
        return report

    def learn_memory(self, base, candidate):
        options = base['memory']['options']
        parent = json.loads(Path(options['snapshot']).read_text())
        lessons = [lesson for episode in self.episode_dirs('D1') for lesson in memory.candidates_from_episode(episode)]
        result = {'candidates': len(lessons), 'parent': parent['version']}
        if not lessons:
            return dict(result, promoted=False, reason='no candidates')
        promotion = dict({'min_cases': 2, 'min_success_delta': 0, 'min_score_delta': 0.0, 'no_lost_successes': True},
                         **options.get('promotion', {}))
        draft = memory.next_snapshot(parent, lessons, {'status': 'practice'}, options.get('max_lessons', 10))
        staging = self.out / 'memory_candidate.json'
        write_json(staging, draft)
        frozen = lambda snapshot: {'block': 'frozen', 'options': {'snapshot': str(snapshot),
                                                                   'retrieval': options.get('retrieval', 'tool+context')}}
        practice_base = dict(copy.deepcopy(base), memory=frozen(options['snapshot']))
        practice_cand = dict(copy.deepcopy(base), memory=frozen(staging))
        validation = compare(self.episodes(practice_base, self.plan['cases']['practice'], 'practice_M0'),
                             self.episodes(practice_cand, self.plan['cases']['practice'], 'practice_M1'), promotion)
        result['validation'] = validation
        if not validation['passed']:
            return dict(result, promoted=False, reason='practice did not pass')
        draft['validation'] = {k: validation[k] for k in ('cases', 'valid_pairs', 'won', 'lost', 'mean_score_delta', 'passed')}
        path, digest = memory.write_snapshot(options['store'], draft)
        candidate['memory'] = {'block': 'learn', 'options': dict(options, snapshot=str(path), sha256=digest)}
        return dict(result, promoted=True, version=draft['version'], snapshot=str(path), sha256=digest)

    def learn_vla(self, base, candidate, learning):
        dataset_spec = learning['dataset']
        data_dir = self.out / 'dataset'
        manifest = REGISTRY.create('dataset', dataset_spec.get('block', 'native_samples'),
                                   **dataset_spec.get('options', {})).build(self.episode_dirs('D1'), data_dir)
        trainer = REGISTRY.create('trainer', learning['trainer']['block'], **learning['trainer'].get('options', {}))
        base_checkpoint = (base['vla'].get('options') or {}).get('checkpoint_path')
        lineage = trainer.train(data_dir, base_checkpoint, self.out / 'training')
        result = {'enabled': True, 'dataset': {k: manifest[k] for k in ('version', 'samples', 'rejected', 'by_owner', 'samples_sha256')},
                  'trainer': trainer.describe(), 'lineage': lineage}
        if not lineage['ok']:
            return dict(result, candidate_service=None, reason='training failed or parameters unchanged')
        if base.get('vla') is None:
            return dict(result, candidate_service=None,
                        reason='llm_policy: checkpoint kept; serving it needs a separate agent_vla release')
        serve = learning['release'].get('serve')
        if not serve:
            return dict(result, candidate_service=None, reason='no serving command; checkpoint kept for review')
        root = self.serve(serve['command'], lineage['checkpoint'], 'P1')
        candidate['vla'] = {'block': base['vla']['block'], 'options': dict(base['vla'].get('options', {}),
                            service_root=str(root), checkpoint_ref=lineage['checkpoint_sha256'],
                            checkpoint_path=lineage['checkpoint'])}
        return dict(result, candidate_service=str(root))

    def revalidate_memory(self, candidate):
        options = candidate['memory']['options']
        snapshot = json.loads(Path(options['snapshot']).read_text())
        empty = memory.empty_snapshot(snapshot['task'], snapshot['compatibility'], version='m-revalidation-empty')
        empty_path = self.out / 'memory_empty.json'
        write_json(empty_path, empty)
        with_memory = dict(copy.deepcopy(candidate), memory={'block': 'frozen', 'options': {'snapshot': options['snapshot']}})
        without = dict(copy.deepcopy(candidate), memory={'block': 'frozen', 'options': {'snapshot': str(empty_path)}})
        rule = {'min_cases': 1, 'min_success_delta': 0, 'min_score_delta': -1.0, 'no_lost_successes': True}
        return compare(self.episodes(without, self.plan['cases']['practice'], 'revalidate_without'),
                       self.episodes(with_memory, self.plan['cases']['practice'], 'revalidate_with'), rule)


def run_round(plan_path, out, python=None):
    plan = json.loads(Path(plan_path).read_text())
    return Round(plan, out, python).run()
