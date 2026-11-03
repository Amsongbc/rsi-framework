"""Trainer blocks. `none` is the frozen path; model-specific trainers are plugins.

This repository ships no EWAM training code (the download has none), so no
built-in trainer claims to train EWAM. `external_command` runs a training program
you provide; `synthetic_target` exists only to exercise the loop offline.
"""
