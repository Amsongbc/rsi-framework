"""Bimanual RGB robot with the upstream class name and private hooks (TEST DOUBLE).

Primitives produce absolute joint native actions and spend them through the
toolkit's (counting) environment. `_stateful`, `_ledger` and `_last_state` keep the
shapes the extension relies on.
"""
import dataclasses
import json
import time

import numpy as np

ARMS = ('left', 'right')


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


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
        record = CallRecord(name, dict(args), stateful)
        self.records.append(record)
        return record

    def stateful_calls(self):
        return sum(record.stateful for record in self.records)

    def write_jsonl(self, path):
        with open(path, 'w') as handle:
            for record in self.records:
                handle.write(json.dumps(dataclasses.asdict(record), default=_plain) + '\n')


class RobodojoOfficialColourRobot:
    """Bimanual ARX X5 robot in RoboDojo seen through public RGB and joint state (TEST DOUBLE).
    Joint angles in radians, gripper opening 0 (closed) .. 1 (open)."""

    def __init__(self, backend, primitives='no-vla'):
        self._backend = backend
        self._primitives = primitives
        self._ledger = Ledger()
        self._last_state = None

    def _env(self):
        return self._backend.toolkit._primitives.env

    def _stateful(self, tool, kwargs, build):
        env = self._env()
        record = self._ledger.open(tool, kwargs, stateful=True)
        before, tick = env.issued_actions, time.monotonic()
        try:
            for action in list(build(kwargs)):
                if env.episode_over():
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
            self._backend.toolkit.state.record(env.get_obs(), env.get_status())
            self._last_state = None

    def _current(self):
        state = self._env().get_obs()['state']
        return ({arm: np.asarray(state[arm + '_arm_joint_state'], float).copy() for arm in ARMS},
                {arm: float(np.asarray(state[arm + '_ee_joint_state']).reshape(-1)[0]) for arm in ARMS})

    def _ramp(self, joints, grips, arm, joint_target=None, grip_target=None, steps=6):
        start_q, start_g = joints[arm].copy(), grips[arm]
        actions = []
        for index in range(1, steps + 1):
            fraction = index / steps
            if joint_target is not None:
                joints[arm] = start_q + (np.asarray(joint_target, float) - start_q) * fraction
            if grip_target is not None:
                grips[arm] = start_g + (grip_target - start_g) * fraction
            actions.append({f'{a}_{part}': (joints[a].copy() if part == 'arm_joint_state' else np.array([grips[a]]))
                            for a in ARMS for part in ('arm_joint_state', 'ee_joint_state')})
        return actions

    def state(self) -> dict:
        """Current joint angles, gripper openings and measured link6 poses of both arms."""
        if self._last_state is None:
            obs = self._env().get_obs()
            self._last_state = {key: np.asarray(value).round(5).tolist() for key, value in obs['state'].items()}
        return self._last_state

    def show(self, *cameras) -> dict:
        """Return current camera images to the model (cam_head, cam_left_wrist, cam_right_wrist)."""
        obs = self._env().get_obs()
        names = cameras or ('cam_head',)
        return {name: list(np.asarray(obs['vision'][name]['color']).shape) for name in names}

    def locate(self, prompt: str, camera: str = 'cam_head') -> dict:
        """Segment `prompt` in one camera image; read-only, no motion."""
        image = self._env().get_obs()['vision'][camera]['color']
        result = self._backend.toolkit._primitives.sam3_client.segment(image, text_prompt=prompt)
        return {'found': result.found, 'score': result.score, 'box': result.box}

    def move_to(self, arm: str, joints: list, steps: int = 6) -> dict:
        """Move one arm to absolute joint angles (6 values, radians) in `steps` native actions."""
        if arm not in ARMS or len(joints) != 6 or not 1 <= steps <= 24:
            raise ValueError('move_to needs arm left/right, 6 joints and 1..24 steps')
        return self._stateful('move_to', {'arm': arm, 'joints': list(joints), 'steps': steps},
                              lambda kw: self._ramp(*self._current(), arm, joint_target=joints, steps=steps))

    def set_gripper(self, arm: str, opening: float, steps: int = 3) -> dict:
        """Set one gripper opening, 0 closed .. 1 open."""
        if arm not in ARMS or not 0 <= opening <= 1:
            raise ValueError('set_gripper needs arm left/right and opening in [0, 1]')
        return self._stateful('set_gripper', {'arm': arm, 'opening': opening, 'steps': steps},
                              lambda kw: self._ramp(*self._current(), arm, grip_target=opening, steps=steps))

    def home(self, steps: int = 8) -> dict:
        """Move both arms to zero joint angles."""
        def build(kw):
            joints, grips = self._current()
            return (self._ramp(joints, grips, 'left', joint_target=np.zeros(6), steps=steps)
                    + self._ramp(joints, grips, 'right', joint_target=np.zeros(6), steps=steps))
        return self._stateful('home', {'steps': steps}, build)


@dataclasses.dataclass(frozen=True)
class RobotAdapter:
    name: str
    robot_cls: type
    services_novla: str


ADAPTER = RobotAdapter('robodojo_xpl_rgb', RobodojoOfficialColourRobot, 'a SAM3 segmentation service')
