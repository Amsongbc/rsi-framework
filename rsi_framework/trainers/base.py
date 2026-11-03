"""Trainers produce a candidate checkpoint outside any running episode.

Every trainer returns lineage: dataset manifest hash, base checkpoint hash, new
checkpoint hash, command and log. A candidate whose checkpoint hash equals the base
is reported as `parameters_changed: false` and cannot be released as trained.
"""
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from rsi_framework.core.io import read_jsonl, sha256_file, write_json
from rsi_framework.core.registry import register


class FrozenTrainer:
    """The frozen path: any training request is an error, and the count stays zero."""
    kind = 'trainer'
    evidence = 'real'
    calls = 0

    def describe(self):
        return {'block': 'none', 'calls': self.calls}

    def train(self, *args, **kwargs):
        raise RuntimeError('This profile is frozen: training is disabled')


class ExternalCommandTrainer:
    """Runs a training program you supply.

    `command` is an argv list; {dataset}, {samples}, {base_checkpoint}, {output_dir}
    are substituted. The program must write the checkpoint at `checkpoint` (relative
    to output_dir). Example: a fine-tuning script for the VLA you serve.
    """
    kind = 'trainer'
    evidence = 'real'

    def __init__(self, *, command, checkpoint, timeout_s=None, env=None):
        self.command = list(command)
        self.checkpoint = checkpoint
        self.timeout_s = timeout_s
        self.env = env

    def describe(self):
        return {'block': 'external_command', 'command': self.command, 'checkpoint': self.checkpoint}

    def train(self, dataset_dir, base_checkpoint, output_dir):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        values = {'dataset': str(dataset_dir), 'samples': str(Path(dataset_dir) / 'samples.jsonl'),
                  'base_checkpoint': str(base_checkpoint or ''), 'output_dir': str(output)}
        argv = [part.format(**values) for part in self.command]
        started = time.monotonic()
        with (output / 'train.log').open('w') as log:
            code = subprocess.call(argv, stdout=log, stderr=subprocess.STDOUT, timeout=self.timeout_s, env=self.env)
        checkpoint = output / self.checkpoint
        return lineage(dataset_dir, base_checkpoint, checkpoint if checkpoint.exists() else None, output,
                       dict(argv=argv, returncode=code, wall_s=time.monotonic() - started, trainer='external_command'))


class SyntheticTargetTrainer:
    """Offline loop exercise for the fake EWAM service: the 'checkpoint' is a target pose
    (mean final state of the selected episodes). Its evidence label keeps every
    result out of real conclusions."""
    kind = 'trainer'
    evidence = 'software_synthetic'

    def __init__(self, *, gain=0.06):
        self.gain = gain

    def describe(self):
        return {'block': 'synthetic_target', 'gain': self.gain}

    def train(self, dataset_dir, base_checkpoint, output_dir):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        samples = read_jsonl(Path(dataset_dir) / 'samples.jsonl')
        if not samples:
            return lineage(dataset_dir, base_checkpoint, None, output,
                           dict(trainer='synthetic_target', returncode=1, reason='no samples selected'))
        last = {}
        for sample in samples:
            last[sample['episode']] = sample['post_state']
        keys = sorted(set(samples[0]['action']) & set(samples[0]['post_state']))
        target = {key: np.mean([np.asarray(state[key], float).reshape(-1) for state in last.values()], axis=0).round(6).tolist()
                  for key in keys}
        checkpoint = output / 'checkpoint.json'
        write_json(checkpoint, {'format': 'fake-ewam-ckpt/1', 'target': target, 'gain': self.gain,
                                'episodes': sorted(last)})
        return lineage(dataset_dir, base_checkpoint, checkpoint, output, dict(trainer='synthetic_target', returncode=0))


def lineage(dataset_dir, base_checkpoint, checkpoint, output, run):
    manifest = Path(dataset_dir) / 'manifest.json'
    base = Path(base_checkpoint) if base_checkpoint else None
    record = {'dataset_manifest_sha256': sha256_file(manifest) if manifest.exists() else None,
              'base_checkpoint': str(base) if base else None,
              'base_checkpoint_sha256': sha256_file(base) if base and base.exists() else None,
              'checkpoint': str(checkpoint) if checkpoint else None,
              'checkpoint_sha256': sha256_file(checkpoint) if checkpoint else None, **run}
    record['parameters_changed'] = bool(record['checkpoint_sha256']
                                        and record['checkpoint_sha256'] != record['base_checkpoint_sha256'])
    record['ok'] = record.get('returncode') == 0 and record['parameters_changed']
    write_json(Path(output) / 'lineage.json', record)
    return record


@register('trainer', 'none', description='Frozen: no training')
def make_none(**options):
    return FrozenTrainer()


@register('trainer', 'external_command', description='Your training program for your VLA')
def make_external(**options):
    return ExternalCommandTrainer(**options)


@register('trainer', 'synthetic_target', evidence='software_synthetic', description='Offline loop exercise only')
def make_synthetic(**options):
    return SyntheticTargetTrainer(**options)
