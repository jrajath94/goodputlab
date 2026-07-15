"""Tests for the AUTO 0-drop + thrash counters (AUTOSCALER observability).

Per ROADMAP.md Phase 7 success criterion:

> "``flip_count_per_minute`` <0.5 sustained;
>  ``controller_thrash_detected`` alarm fires on 2 flips within 240s"
> "Zero dropped in-flight requests during role flips"

Two new counters required:

- ``goodputlab_controller_thrash_total`` — incremented when a flip
  fires within the 240s thrash window of the previous flip on the
  same pool (P6 mitigation; alarms when over 0/min steady-state).
- ``goodputlab_role_flip_inflight_dropped_total`` — incremented only
  when the controller forcibly tears down a pool with requests still
  in-flight (AUTO-05 zero-drop evidence). The drain protocol must
  keep this at 0 under correct usage.
"""

from __future__ import annotations

from control.autoscaler import PoolAutoscaler, PoolTopology
from control.pid import PidController, PidGains
from control.pool import Pool
from obs.registry import MetricsRegistry


class _Clock:
    def __init__(self, t0: float = 1000.0) -> None:
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _ctrl() -> PidController:
    return PidController(PidGains(kp=2.0, ki=0.0, kd=0.0), -10.0, 10.0)


def _autoscaler_with_metrics() -> tuple[PoolAutoscaler, MetricsRegistry, _Clock]:
    clock = _Clock(1000.0)
    metrics = MetricsRegistry()
    a = PoolAutoscaler(
        {Pool.PREFILL: _ctrl()},
        min_dwell_s=0.0,
        clock=clock,
        metrics=metrics,
    )
    return a, metrics, clock


def _count(metrics: MetricsRegistry, attr: str) -> float:
    """Return the current value of a Counter on the registry.

    ``samples[0].value`` is typed ``Any`` in ``prometheus_client``'s
    stubs; explicit ``float()`` keeps mypy strict-mode happy.
    """
    samples = list(getattr(metrics, attr).collect())[0].samples
    assert samples, f"{attr} counter has no samples"
    return float(samples[0].value)


def test_autoscaler_counters_exist() -> None:
    """The required thrash + zero-drop counters must be declared by the registry."""
    metrics = MetricsRegistry()
    assert hasattr(metrics, "controller_thrash")
    assert hasattr(metrics, "role_flip_inflight_dropped")


def test_autoscaler_two_flips_within_240s_emits_thrash() -> None:
    """Two flips within the thrash window increment the controller_thrash counter."""
    a, metrics, clock = _autoscaler_with_metrics()
    topo = {Pool.PREFILL: PoolTopology(pool=Pool.PREFILL, replicas=2, target_queue_depth=10)}

    # Flip 1: scale up (queue high) at t=0
    a.tick(topo, queue_depths={Pool.PREFILL: 30}, in_flight={Pool.PREFILL: 0})
    assert _count(metrics, "controller_thrash") == 0.0

    # Flip 2 at t=60s — within 240s window → thrash alarm fires
    clock.advance(60.0)
    a.tick(topo, queue_depths={Pool.PREFILL: 0}, in_flight={Pool.PREFILL: 0})

    assert _count(metrics, "controller_thrash") == 1.0


def test_autoscaler_flips_beyond_240s_no_thrash() -> None:
    """Flips separated by ≥240s do NOT trip the thrash alarm."""
    a, metrics, clock = _autoscaler_with_metrics()
    topo = {Pool.PREFILL: PoolTopology(pool=Pool.PREFILL, replicas=2, target_queue_depth=10)}

    # Flip 1
    a.tick(topo, queue_depths={Pool.PREFILL: 30}, in_flight={Pool.PREFILL: 0})
    # 240s gap, just past the window
    clock.advance(241.0)
    # Flip 2 (different direction)
    a.tick(topo, queue_depths={Pool.PREFILL: 0}, in_flight={Pool.PREFILL: 0})

    assert _count(metrics, "controller_thrash") == 0.0


def test_autoscaler_no_drop_when_drain_block_honored() -> None:
    """Drain protocol keeps inflight-dropped counter at 0 across many ticks."""
    a, metrics, _ = _autoscaler_with_metrics()
    topo = {Pool.PREFILL: PoolTopology(pool=Pool.PREFILL, replicas=4, target_queue_depth=10)}

    # 30 ticks with in_flight > 0 → drain_wait every time, no drops
    for n in (1, 2, 5, 8, 3, 7, 4):
        a.tick(topo, queue_depths={Pool.PREFILL: 0}, in_flight={Pool.PREFILL: n})

    assert _count(metrics, "role_flip_inflight_dropped") == 0.0


def test_autoscaler_metrics_none_back_compat() -> None:
    """Passing metrics=None must keep the autoscaler fully functional."""
    a = PoolAutoscaler(
        {Pool.PREFILL: _ctrl()},
        min_dwell_s=0.0,
    )
    topo = {Pool.PREFILL: PoolTopology(pool=Pool.PREFILL, replicas=2, target_queue_depth=10)}
    decisions = a.tick(topo, queue_depths={Pool.PREFILL: 30}, in_flight={Pool.PREFILL: 0})
    assert len(decisions) == 1
    assert decisions[0].delta != 0
