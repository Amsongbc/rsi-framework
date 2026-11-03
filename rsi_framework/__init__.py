"""Composable RSI framework built on the pyrua-ewam extension in this repository.

Blocks (environment, VLA, LLM backend, policy, tools, execution protocol, Memory,
data, trainer, evaluation, release) are registered by name and assembled from a
profile. The original repository sits unchanged in pyrua-ewam-agent/; its cohorts
are either launched as they are or rebuilt from blocks that are checked against them.
"""
from pathlib import Path

__version__ = '0.1.0'
REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = REPO_ROOT / 'pyrua-ewam-agent'
PROJECTS_DIR = ORIGINAL_ROOT / 'projects'
