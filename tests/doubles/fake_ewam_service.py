#!/usr/bin/env python3
"""Deterministic stand-in for projects/ewam_comparison/ewam_service.py (TEST DOUBLE, no model).

Speaks the same Unix-socket protocol (health / reset / release / infer, one episode
lease at a time, authkey file) so the unchanged EwamClient is exercised. `infer`
filters the observation with the original public_observation and returns 48 native
joint actions that drift toward an instruction-dependent pose, plus seeded noise, so
results depend on reset seeds and call order exactly like a stochastic policy would.
"""
import argparse
import hashlib
import json
from multiprocessing.connection import Listener
import os
from pathlib import Path
import random
import secrets
import sys
import traceback

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'projects/ewam_comparison'))
from ewam_service import public_observation  # noqa: E402

ARMS = ('left', 'right')


def infer(obs, checkpoint=None):
    digest = hashlib.sha256(obs['instruction'].encode()).digest()
    target = {arm: (np.frombuffer(digest[6 * i:6 * i + 6], np.uint8) / 255.0 - 0.5) * 0.8
              for i, arm in enumerate(ARMS)}
    if checkpoint:
        target = {arm: np.asarray(checkpoint['target'].get(arm + '_arm_joint_state', checkpoint['target'].get(arm)), float)
                  for arm in ARMS}
    q = {arm: np.asarray(obs['state'][arm + '_arm_joint_state'], float).copy() for arm in ARMS}
    grip = {arm: float(np.asarray(obs['state'][arm + '_ee_joint_state']).reshape(-1)[0]) for arm in ARMS}
    actions = []
    for _ in range(48):
        action = {}
        for arm in ARMS:
            q[arm] = q[arm] + (target[arm] - q[arm]) * 0.06 + np.random.normal(0, 1e-3, 6)
            action[arm + '_arm_joint_state'] = q[arm].copy()
            action[arm + '_ee_joint_state'] = np.array([grip[arm]])
        actions.append(action)
    return actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, help='fake-ewam-ckpt/1 JSON from the synthetic trainer')
    args = parser.parse_args()
    checkpoint = json.loads(args.checkpoint.read_text()) if args.checkpoint else None
    args.root.mkdir(parents=True, exist_ok=True, mode=0o700)
    socket = args.root / 'ewam.sock'
    if socket.exists():
        raise RuntimeError('Existing socket in fake EWAM root')
    metadata = {'status': 'ready', 'dummy': True, 'test_double': True, 'strict_checkpoint': False,
                'gpu': None, 'startup_s': 0.0, 'config': {'action_stride': 3}, 'pid': os.getpid(),
                'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None}
    (args.root / 'model.json').write_text(json.dumps(metadata, indent=2) + '\n')
    key = args.root / 'authkey'
    key.write_bytes(secrets.token_bytes(32))
    key.chmod(0o600)
    active = None
    with Listener(str(socket), family='AF_UNIX', authkey=key.read_bytes()) as listener:
        print('EWAM service ready', flush=True)
        while True:
            with listener.accept() as connection:
                try:
                    message = connection.recv()
                    op = message['op']
                    if op == 'health':
                        result = metadata
                    elif op == 'reset':
                        if active is not None and active != message['session_id']:
                            raise RuntimeError('Another experiment episode owns this EWAM service')
                        active = message['session_id']
                        seed = int(message['seed'])
                        random.seed(seed)
                        np.random.seed(seed)
                        result = {'seed': seed, 'session_id': active}
                    elif op == 'release':
                        if active != message['session_id']:
                            raise RuntimeError('Wrong EWAM episode lease')
                        active = None
                        result = {}
                    elif op == 'infer':
                        if active != message['session_id']:
                            raise RuntimeError('Wrong EWAM episode lease')
                        obs = public_observation(message['obs'], message.get('instruction'))
                        result = {'actions': infer(obs, checkpoint), 'inference_s': 0.0, 'instruction': obs['instruction']}
                    elif op == 'shutdown':
                        connection.send({'ok': True, 'result': {}})
                        return
                    else:
                        raise ValueError(f'Unknown EWAM operation: {op}')
                    connection.send({'ok': True, 'result': result})
                except Exception:
                    connection.send({'ok': False, 'error': traceback.format_exc()})


if __name__ == '__main__':
    main()
