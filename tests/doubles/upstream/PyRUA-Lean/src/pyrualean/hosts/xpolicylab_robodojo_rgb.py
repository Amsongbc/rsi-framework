"""RGB RoboDojo host (TEST DOUBLE); delegates to the base host so lay_out patches apply."""
from pyrualean.hosts import xpolicylab_robodojo as base

Cell = base.Cell


def boot(cell, out, **kwargs):
    return base.boot(cell, out, **kwargs)


def make_backend(toolkit):
    return base.make_backend(toolkit)
