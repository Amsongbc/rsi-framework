"""Deterministic toy forward kinematics with the upstream function names (TEST DOUBLE).

The link lengths are invented; only consistency matters, because the extension
checks FK against the measured link6 pose that this same module produced.
"""
import numpy as np

BASE = {'left': np.array([0.0, 0.3, 0.0]), 'right': np.array([0.0, -0.3, 0.0])}
LINKS = np.array([[0, 0, .12], [.25, 0, 0], [.22, 0, 0], [.06, 0, 0], [.05, 0, 0], [.04, 0, 0]])
AXES = 'zyyxyx'


def _rotation(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def fk(arm, q):
    q = np.asarray(q, float).reshape(6)
    transform = np.eye(4)
    transform[:3, 3] = BASE[arm]
    for axis, angle, link in zip(AXES, q, LINKS):
        step = np.eye(4)
        step[:3, :3] = _rotation(axis, angle)
        step[:3, 3] = step[:3, :3] @ link
        transform = transform @ step
    return transform


def tcp(pose, offset=0.1):
    pose = np.asarray(pose, float)
    return pose[:3, :3] @ np.array([offset, 0.0, 0.0]) + pose[:3, 3]


def quat_wxyz(rotation):
    m = np.asarray(rotation, float)
    trace = np.trace(m)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        return np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
    q = np.empty(4)
    q[0] = (m[k, j] - m[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (m[j, i] + m[i, j]) / s
    q[1 + k] = (m[k, i] + m[i, k]) / s
    return q
