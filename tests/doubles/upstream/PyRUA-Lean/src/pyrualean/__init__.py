"""Offline TEST DOUBLE of the PyRUA-Lean interfaces used by the pyrua-ewam extension.

This is not PyRUA-Lean. It mirrors only the names and call signatures that the
extension and rsi_framework call, backed by a small deterministic simulator, so the
original code and the framework can be compared on identical inputs without Isaac
Sim, RoboDojo, EWAM or SAM3. Everything produced with it is software_synthetic
evidence and never a RoboDojo result.
"""

IS_TEST_DOUBLE = True
