"""Smoke tests for the project skeleton."""


def test_core_and_control_importable() -> None:
    """Both core and control packages must be importable from the repo root."""
    import control
    import core

    assert core is not None
    assert control is not None
