"""Composable RSI framework built on the pyrua-ewam extension in this repository.

Blocks (environment, VLA, LLM backend, policy, tools, execution protocol, Memory,
data, trainer, evaluation, release) are registered by name and assembled from a
profile. The original cohorts under projects/ stay byte-identical and runnable;
compat profiles either launch them unchanged or rebuild them from blocks that are
checked against them (see README.md in this package).
"""
from pathlib import Path

__version__ = '0.1.0'
REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECTS_DIR = REPO_ROOT / 'projects'
