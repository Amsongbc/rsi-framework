"""S1: launch an original entry unchanged.

The script under the original projects/<cohort>/ runs with the caller's argv,
working directory and environment, exactly as the README commands run it (a
relative --out still resolves against the caller's directory). Nothing is written
into the run directory, because the original sessions require it not to exist.
"""
import os
import subprocess
import sys

from rsi_framework.compat import baseline, paths
from rsi_framework.compat.cohorts import entry


def command(cohort, name, args, python=None):
    script = paths.original_projects() / cohort / entry(cohort, name).script
    return [python or os.environ.get('EWAM_PYTHON') or sys.executable, str(script), *args]


def run(cohort, name, args, python=None, check_baseline=True, env=None, cwd=None):
    if check_baseline:
        baseline.require_unchanged(projects=paths.original_projects())
    return subprocess.call(command(cohort, name, args, python), cwd=cwd, env=env)


def exec_original(cohort, name, args, python=None):
    """Replace this process with the original entry (same pid, signals and exit code)."""
    baseline.require_unchanged(projects=paths.original_projects())
    argv = command(cohort, name, args, python)
    os.execv(argv[0], argv)
