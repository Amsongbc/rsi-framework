"""Generic tools for any RobotSpec and any environment adapter.

`move(key, target, steps)` ramps one native action key (a joint group, a gripper...)
to absolute values; `hold(steps)` repeats the current action; `state()` and `show()`
read. Every motion spends the shared native budget and is written to the primitive
ledger in the same format as PyRUA's, so data building and protocols work unchanged.
Use it to drive a new environment before writing robot-specific primitives.
"""
import dataclasses
import json
import time

import numpy as np

from rsi_framework.core.contracts import EpisodeEnded, ToolError
from rsi_framework.core.registry import register
from rsi_framework.policies import sandbox
from rsi_framework.tools.base import ToolSet


@dataclasses.dataclass
class CallRecord:
    name: str
    args: dict
    stateful: bool
    ok: bool = False
    error: str | None = None
    env_steps: int = 0
    elapsed_s: float = 0.0
    summary: object = None


class Ledger:
    def __init__(self):
        self.records = []

    def open(self, name, args, stateful=False):
        self.records.append(CallRecord(name, dict(args), stateful))
        return self.records[-1]

    def stateful_calls(self):
        return sum(r.stateful for r in self.records)

    def write_jsonl(self, path):
        with open(path, 'w') as handle:
            for record in self.records:
                handle.write(json.dumps(dataclasses.asdict(record), default=str) + '\n')


class JointRobot:
    """Robot built from its RobotSpec: absolute targets per native action key."""

    def __init__(self, backend, primitives=None):
        self._run = backend
        self._ledger = Ledger()
        self._last_state = None

    def _current(self):
        state = self._run.env.get_obs()['state']
        return {k: np.asarray(state[k], float).reshape(-1).copy() for k in self._run.robot.action_keys}

    def _stateful(self, tool, kwargs, build):
        run, env = self._run, self._run.env
        record = self._ledger.open(tool, kwargs, stateful=True)
        before, tick = env.issued_actions, time.monotonic()
        try:
            for action in list(build(kwargs)):
                if env.episode_over() or env.issued_actions >= env.initial_limit:
                    break
                env.step(action)
            record.ok = True
            return {'tool': tool, 'native_actions': env.issued_actions - before}
        except BaseException as exc:
            record.error = type(exc).__name__ + ': ' + str(exc)
            raise
        finally:
            record.env_steps = env.issued_actions - before
            record.elapsed_s = time.monotonic() - tick
            run.env_block.after_motion(run)
            self._last_state = None

    def state(self) -> dict:
        """Current public robot state (all state keys of the robot spec)."""
        if self._last_state is None:
            obs = self._run.env.get_obs()
            self._last_state = {k: np.asarray(obs['state'][k]).round(5).tolist() for k in self._run.robot.state_keys}
        return self._last_state

    def show(self, *cameras) -> dict:
        """Image shapes of the requested cameras (default: all); the images reach the policy with each observation."""
        obs = self._run.env.get_obs()
        return {c: list(np.asarray(obs['vision'][c]['color']).shape) for c in (cameras or self._run.robot.cameras)}

    def move(self, key: str, target: list, steps: int = 6) -> dict:
        """Ramp one native action key (see the robot spec's action_keys) to absolute `target` values in `steps` actions."""
        sizes = self._run.robot.action_keys
        if key not in sizes or len(target) != sizes[key] or not 1 <= steps <= 48:
            raise ValueError(f'move needs key in {sorted(sizes)}, {sizes.get(key)} target values and 1..48 steps')

        def build(kw):
            current = self._current()
            start, goal = current[key].copy(), np.asarray(target, float)
            for index in range(1, steps + 1):
                current[key] = start + (goal - start) * index / steps
                yield {k: v.copy() for k, v in current.items()}
        return self._stateful('move', {'key': key, 'target': list(target), 'steps': steps}, build)

    def hold(self, steps: int = 1) -> dict:
        """Repeat the current action for `steps` native actions."""
        return self._stateful('hold', {'steps': steps}, lambda kw: [self._current() for _ in range(steps)])


class JointTools(ToolSet):
    readonly = ('state', 'show')
    tool_error = ToolError
    primitives = None

    def __init__(self, *, expose=None):
        self.expose = tuple(expose) if expose else None

    def describe(self):
        return {'block': 'joint', 'expose': self.expose}

    def sandbox(self):
        return sandbox.make_builtins, sandbox.audit_source, sandbox.blocked_names, (EpisodeEnded,)

    def backend(self, run):
        return run.env_block.robot_backend(run)

    def base_robot_class(self):
        return JointRobot


@register('tools', 'joint', description='Generic move/hold/state for any RobotSpec and environment')
def make(**options):
    return JointTools(**options)
