"""The environment contract the runner, recorder, protocols and tools rely on.

EnvironmentAdapter (implement all, or inherit NativeEnvAdapter):
  boot(run)              set run.env: budget-counted, recorded env (see recording.py)
  finish(run) -> dict    conclude the episode, return the official verdict {success, score}
  outcome(run, verdict)  -> (raw results, details, valid) for the summary
  after_motion(run)      refresh caches after actions issued outside the tools (default: no-op)
  robot_backend(run)     handle the tool set builds its robot API on (default: the runner)
  processes_alive(run)   helper processes still running (default: True)
  stop(run)              stop helper processes (default: no-op)
  robot                  name of the RobotSpec whose observation/action format it publishes
"""
from rsi_framework.environments.recording import RecordedNativeEnv


class EnvironmentAdapter:
    kind = 'environment'
    robot = None
    evidence = 'real'

    def __init__(self, **options):
        self.options = options

    def describe(self):
        return {'block': getattr(self, 'name', type(self).__name__), 'robot': self.robot,
                'options': self.options, 'evidence': self.evidence}

    def prepare(self, run):
        pass

    def boot(self, run):
        raise NotImplementedError

    def booted(self, run):
        return hasattr(run, 'env')

    def context_entries(self, run):
        return {'env': run.env}

    def processes_alive(self, run):
        return True

    def finish(self, run):
        raise NotImplementedError

    def outcome(self, run, verdict):
        raise NotImplementedError

    def after_motion(self, run):
        pass

    def robot_backend(self, run):
        return run

    def stop(self, run):
        pass


class NativeEnvAdapter(EnvironmentAdapter):
    """Any environment object with get_obs, get_status ({step, step_limit}), step(action),
    episode_over(), conclude() and result() ({success, score}). Subclasses implement
    make_native(run) and set `robot`."""

    def make_native(self, run):
        raise NotImplementedError

    def boot(self, run):
        run.native_env = self.make_native(run)
        problems = run.robot.observation_problems(run.native_env.get_obs())
        if problems:
            raise ValueError(f'{type(self).__name__} observation does not match robot {run.robot.name}: {problems}')
        run.env = RecordedNativeEnv(run.native_env, run.out, run.robot.cameras)

    def finish(self, run):
        run.native_env.conclude()
        return run.native_env.result()

    def outcome(self, run, verdict):
        raw = [] if verdict is None else [{'source': type(run.native_env).__name__, 'value': verdict}]
        details = [] if verdict is None else [verdict]
        valid = bool(not run.errors and verdict is not None and 'success' in verdict)
        return raw, details, valid
