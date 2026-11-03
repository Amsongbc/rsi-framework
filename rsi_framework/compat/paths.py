"""Where the original modules and upstream sources are found.

The original scripts compute PROJECTS from $PROJECTS_ROOT (default: their parent
directory) when imported, and their implementation snapshot requires the cohort
directory to sit inside PROJECTS. The framework therefore imports the original
modules from the same workspace: $RSI_ORIGINAL_PROJECTS if set, else $PROJECTS_ROOT
when it holds the cohorts, else projects/ of this repository. Whichever is used is
checked against the original manifest before compat runs.
"""
import os
from pathlib import Path
import sys

from rsi_framework import PROJECTS_DIR

WINDOW = 'ewam_window_parallel24'


def projects_root():
    return Path(os.environ.get('PROJECTS_ROOT', str(PROJECTS_DIR)))


def original_projects():
    if os.environ.get('RSI_ORIGINAL_PROJECTS'):
        return Path(os.environ['RSI_ORIGINAL_PROJECTS'])
    if (projects_root() / WINDOW / 'window_session.py').is_file():
        return projects_root()
    return PROJECTS_DIR


def runtime_root():
    return projects_root() / 'ewam_comparison'


def import_paths(cohort=WINDOW):
    root = projects_root()
    return [str(original_projects() / cohort), str(root / 'PyRUA-Lean/src'), str(root / 'ewam_comparison')]


def ensure_importable(cohort=WINDOW):
    """Put the original cohort, PyRUA-Lean and ewam_comparison on sys.path (idempotent)."""
    for path in reversed(import_paths(cohort)):
        if path not in sys.path:
            sys.path.insert(0, path)
