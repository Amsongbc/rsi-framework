"""Environment plugins: lifecycle, public observations, native actions, official result.

To add an environment, subclass environments.base.NativeEnvAdapter, set `robot` to a
registered RobotSpec name and implement `make_native(run)` returning an object with
get_obs / get_status / step / episode_over / conclude / result. Recording, the shared
native budget and the summary come from the base class. Environments with their own
process model (like the PyRUA host) subclass EnvironmentAdapter instead.
"""
