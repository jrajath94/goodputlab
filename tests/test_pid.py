"""Tests for control/pid.py — discrete PID + anti-windup."""

from __future__ import annotations

import pytest

from control.pid import PidController, PidGains


def test_pid_rejects_inverted_clamp_range() -> None:
    with pytest.raises(ValueError):
        PidController(PidGains(kp=1.0), output_min=10.0, output_max=0.0)


def test_pid_rejects_nonpositive_dt() -> None:
    p = PidController(PidGains(kp=1.0), output_min=-10.0, output_max=10.0)
    with pytest.raises(ValueError):
        p.update(1.0, dt=0.0)
    with pytest.raises(ValueError):
        p.update(1.0, dt=-1.0)


def test_pid_proportional_only_response() -> None:
    p = PidController(PidGains(kp=2.0, ki=0.0, kd=0.0), -10.0, 10.0)
    assert p.update(3.0, dt=1.0) == pytest.approx(6.0)
    assert p.update(-1.0, dt=1.0) == pytest.approx(-2.0)


def test_pid_integral_accumulates_over_time() -> None:
    p = PidController(PidGains(kp=0.0, ki=1.0, kd=0.0), -100.0, 100.0)
    p.update(2.0, dt=1.0)
    p.update(2.0, dt=1.0)
    p.update(2.0, dt=1.0)
    # Integral accumulates: 2 + 2 + 2 = 6
    assert p.integral == pytest.approx(6.0)


def test_pid_integral_eliminates_offset() -> None:
    """Proportional-only has steady-state error; integral cancels it."""
    p_only = PidController(PidGains(kp=1.0, ki=0.0, kd=0.0), -10.0, 10.0)
    pi_ctrl = PidController(PidGains(kp=1.0, ki=1.0, kd=0.0), -10.0, 10.0)
    # Constant error 1.0
    p_only.update(1.0, dt=1.0)
    pi_ctrl.update(1.0, dt=1.0)
    for _ in range(10):
        p_only.update(1.0, dt=1.0)
        pi_ctrl.update(1.0, dt=1.0)
    # PI should have output > 1.0 (integral pushed it up)
    assert pi_ctrl.update(1.0, dt=1.0) > p_only.update(1.0, dt=1.0)


def test_pid_derivative_dampens_oscillation() -> None:
    p = PidController(PidGains(kp=0.0, ki=0.0, kd=1.0), -10.0, 10.0)
    # First call: no previous error, derivative=0
    assert p.update(5.0, dt=1.0) == 0.0
    # Second call: derivative = (3 - 5) / 1 = -2
    assert p.update(3.0, dt=1.0) == pytest.approx(-2.0)


def test_pid_anti_windup_clamps_integrator() -> None:
    """Long-running positive error with low output clamp should not wind up."""
    p = PidController(PidGains(kp=0.0, ki=1.0, kd=0.0), -1.0, 1.0)
    # Push huge positive errors many times. Output clamps at 1.0; integrator must NOT grow.
    for _ in range(100):
        out = p.update(100.0, dt=1.0)
        assert out == 1.0
    # Integrator should be near 0 (held at 0 when saturating upward).
    # Actually with my impl, integrator doesn't accumulate on saturate at the top,
    # so it should remain 0.
    assert p.integral == pytest.approx(0.0)


def test_pid_anti_windup_clamps_integrator_negative() -> None:
    p = PidController(PidGains(kp=0.0, ki=1.0, kd=0.0), -1.0, 1.0)
    for _ in range(100):
        out = p.update(-100.0, dt=1.0)
        assert out == -1.0
    assert p.integral == pytest.approx(0.0)


def test_pid_reset_zero_state() -> None:
    p = PidController(PidGains(kp=1.0, ki=1.0, kd=1.0), -10.0, 10.0)
    p.update(5.0, dt=1.0)
    p.update(3.0, dt=1.0)
    assert p.integral != 0.0
    p.reset()
    assert p.integral == 0.0
    # After reset, integral and derivative state are zero; P term still applies.
    # kp*5 + ki*(0+5*1) + kd*0 = 5 + 5 + 0 = 10
    assert p.update(5.0, dt=1.0) == pytest.approx(10.0)


def test_pid_output_within_clamp_range() -> None:
    p = PidController(PidGains(kp=10.0), -5.0, 5.0)
    for err in [-100.0, -10.0, 0.0, 10.0, 100.0]:
        out = p.update(err)
        assert -5.0 <= out <= 5.0