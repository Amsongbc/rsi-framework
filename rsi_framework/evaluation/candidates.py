"""Candidate evaluation: paired episodes with everything but the compared part fixed.

A rule is fixed before the runs (min valid pairs, allowed lost successes, minimum
mean score change). The report keeps both sides' per-case outcomes so an
improvement claim can be checked against the raw summaries.
"""
from rsi_framework.core.registry import register
from rsi_framework.memory.manager import compare_practice

DEFAULT_RULE = {'min_cases': 2, 'min_success_delta': 0, 'min_score_delta': 0.0, 'no_lost_successes': True}


def compare(baseline, candidate, rule=None):
    rule = dict(DEFAULT_RULE, **(rule or {}))
    report = compare_practice(baseline, candidate, rule)
    report['per_case'] = [{'case': {k: b.get(k) for k in ('task', 'eval_seed', 'layout')},
                           'baseline': {k: b.get(k) for k in ('valid', 'success', 'official_score', 'evidence')},
                           'candidate': {k: c.get(k) for k in ('valid', 'success', 'official_score', 'evidence')}}
                          for b, c in zip(baseline, candidate)]
    labels = {s.get('evidence', 'unlabelled') for s in baseline + candidate}
    report['evidence'] = sorted(labels)
    return report


class PairedEvaluator:
    kind = 'evaluator'
    evidence = 'real'

    def __init__(self, *, rule=None):
        self.rule = dict(DEFAULT_RULE, **(rule or {}))

    def describe(self):
        return {'block': 'paired', 'rule': self.rule}

    def compare(self, baseline, candidate):
        return compare(baseline, candidate, self.rule)


@register('evaluator', 'paired', description='Paired cases, predeclared rule')
def make(**options):
    return PairedEvaluator(**options)
