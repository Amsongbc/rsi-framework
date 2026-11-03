"""Lifecycle helpers (TEST DOUBLE)."""


def stop_all(toolkit, daemons):
    for daemon in daemons or []:
        daemon.stop()
