"""Tests for control/autoscaler.py — PID + drain protocol."""

from __future__ import annotations

import pytest

from control.autoscaler import AutoscalerDecision, PoolAutoscaler, PoolTopology
from control.pid import PidController, PidGains
from control.pool import Pool


def _ctrl(kp: float = 1.0) -> PidController:
    return PidController(PidGains(kp=kp, ki=0.0, kd=0.0), -10.0, 10.0)


def test_autoscaler_scales_up_when_queue_high() -> None:
    a = PoolAutoscaler({Pool.PREFILL: _ctrl()})
    topo = {Pool.PREFILL: PoolTopology(pool=Pool.PREFILL, replicas=2, target_queue_depth=10)}
    # Queue way above target → big positive error → scale up
    decisions = a.tick(topo, queue_depths={Pool.PREFILL: 30}, in_flight={Pool.PREFILL: 0})
    assert len(decisions) == 1
    d = decisions[0]
    assert d.pool == Pool.PREFILL
    assert d.delta >= 1
    assert d.reason == "queue_high"


def test_autoscaler_scales_down_when_queue_low() -> None:
    a = PoolAutoscaler({Pool.PREFILL: _ctrl()})
    topo = {Pool.PREFILL: PoolTopology(pool=Pool.PREFILL, replicas=4, target_queue_depth=10)}
    decisions = a.tick(topo, queue_depths={Pool.PREFILL: 0}, in_flight={Pool.PREFILL: 0})
    d = decisions[0]
    assert d.delta <= -1
    assert d.reason == "queue_low"
    assert d.drained is True


def test_autoscaler_drain_waits_when_in_flight_nonzero() -> None:
    a = PoolAutoscaler({Pool.PREFILL: _ctrl()})
    topo = {Pool.PREFILL: PoolTopology(pool=Pool.PREFILL, replicas=4, target_queue_depth=10)}
    # Queue low + in_flight > 0 → would scale down, but blocked by drain
    decisions = a.tick(topo, queue_depths={Pool.PREFILL: 0}, in_flight={Pool.PREFILL: 3})
    d = decisions[0]
    assert d.delta == 0
    assert d.reason == "drain_wait"


def test_autoscaler_drain_fires_when_in_flight_zero() -> None:
    a = PoolAutoscaler({Pool.PREFILL: _ctrl()})
    topo = {Pool.PREFILL: PoolTopology(pool=Pool.PREFILL, replicas=4, target_queue_depth=10)}
    decisions = a.tick(topo, queue_depths={Pool.PREFILL: 0}, in_flight={Pool.PREFILL: 0})
    d = decisions[0]
    assert d.delta < 0
    assert d.drained is True


def test_autoscaler_caps_at_max_replicas() -> None:
    a = PoolAutoscaler({Pool.PREFILL: _ctrl()}, max_replicas=3)
    topo = {Pool.PREFILL: PoolTopology(pool=Pool.PREFILL, replicas=3, target_queue_depth=10)}
    decisions = a.tick(topo, queue_depths={Pool.PREFILL: 100}, in_flight={Pool.PREFILL: 0})
    d = decisions[0]
    # Already at max → delta=0
    assert d.delta == 0


def test_autoscaler_floor_at_min_replicas() -> None:
    a = PoolAutoscaler({Pool.PREFILL: _ctrl()}, min_replicas=2)
    topo = {Pool.PREFILL: PoolTopology(pool=Pool.PREFILL, replicas=2, target_queue_depth=10)}
    decisions = a.tick(topo, queue_depths={Pool.PREFILL: 0}, in_flight={Pool.PREFILL: 0})
    d = decisions[0]
    # Already at min → no scale down
    assert d.delta == 0


def test_autoscaler_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        PoolAutoscaler({}, min_replicas=-1)
    with pytest.raises(ValueError):
        PoolAutoscaler({}, min_replicas=5, max_replicas=2)
    with pytest.raises(ValueError):
        PoolAutoscaler({}, step_size=0)


def test_autoscaler_handles_missing_controller() -> None:
    a = PoolAutoscaler({Pool.PREFILL: _ctrl()})
    topo = {Pool.DECODE: PoolTopology(pool=Pool.DECODE, replicas=2, target_queue_depth=10)}
    decisions = a.tick(topo, queue_depths={}, in_flight={})
    d = decisions[0]
    assert d.delta == 0
    assert d.reason == "no_controller"


def test_autoscaler_stable_when_at_target() -> None:
    a = PoolAutoscaler({Pool.PREFILL: _ctrl(kp=0.5)})
    topo = {Pool.PREFILL: PoolTopology(pool=Pool.PREFILL, replicas=2, target_queue_depth=10)}
    # Exactly at target → error = 0 → output = midpoint → normalized ≈ 0 → delta = 0
    decisions = a.tick(topo, queue_depths={Pool.PREFILL: 10}, in_flight={Pool.PREFILL: 0})
    d = decisions[0]
    assert d.delta == 0
    assert d.reason == "stable"


def test_autoscaler_emits_decision_per_pool() -> None:
    ctrl_p = _ctrl()
    ctrl_d = _ctrl()
    a = PoolAutoscaler({Pool.PREFILL: ctrl_p, Pool.DECODE: ctrl_d})
    topo = {
        Pool.PREFILL: PoolTopology(pool=Pool.PREFILL, replicas=2, target_queue_depth=10),
        Pool.DECODE: PoolTopology(pool=Pool.DECODE, replicas=2, target_queue_depth=10),
    }
    decisions = a.tick(
        topo,
        queue_depths={Pool.PREFILL: 30, Pool.DECODE: 5},
        in_flight={Pool.PREFILL: 0, Pool.DECODE: 0},
    )
    assert len(decisions) == 2
    pools = {d.pool for d in decisions}
    assert pools == {Pool.PREFILL, Pool.DECODE}


def test_autoscaler_decision_serializable() -> None:
    d = AutoscalerDecision(pool=Pool.PREFILL, delta=1, reason="queue_high", drained=False)
    j = d.model_dump_json()
    assert "queue_high" in j
    assert "delta" in j


def test_pool_topology_rejects_extra_fields() -> None:
    with pytest.raises(Exception):
        PoolTopology.model_validate(
            {"pool": "prefill", "replicas": 2, "target_queue_depth": 10, "wat": "no"}
        )


def test_autoscaler_decision_rejects_extra_fields() -> None:
    with pytest.raises(Exception):
        AutoscalerDecision.model_validate(
            {"pool": "prefill", "delta": 0, "reason": "stable", "wat": "no"}
        )


# ---------- Min-dwell tests (P3-1) ----------


class _Clock:
    """Tiny monotonic clock for min-dwell tests."""

    def __init__(self, t0: float = 1000.0) -> None:
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_min_dwell_blocks_rapid_flip_back() -> None:
    """Within min_dwell_s of a flip, any further delta must be 0 (reason=dwell_wait).

    Sequence: scale-up at t=0, then queue drops at t=30s (< 120s dwell).
    Scale-down must be blocked.
    """
    clock = _Clock(1000.0)
    a = PoolAutoscaler(
        {Pool.PREFILL: _ctrl(kp=1.0)},
        min_dwell_s=120.0,
        clock=clock,
    )
    topo = {Pool.PREFILL: PoolTopology(pool=Pool.PREFILL, replicas=2, target_queue_depth=10)}

    # t=0: queue high → scale up, fires
    decisions = a.tick(topo, queue_depths={Pool.PREFILL: 30}, in_flight={Pool.PREFILL: 0})
    d = decisions[0]
    assert d.delta > 0
    assert d.reason == "queue_high"

    # t=30s (< 120s): queue low, would scale down, blocked by dwell
    clock.advance(30.0)
    decisions = a.tick(topo, queue_depths={Pool.PREFILL: 0}, in_flight={Pool.PREFILL: 0})
    d = decisions[0]
    assert d.delta == 0
    assert d.reason == "dwell_wait"


def test_min_dwell_fires_after_window_elapses() -> None:
    """Once min_dwell_s has elapsed since last flip, scale-down proceeds."""
    clock = _Clock(1000.0)
    a = PoolAutoscaler(
        {Pool.PREFILL: _ctrl(kp=1.0)},
        min_dwell_s=120.0,
        clock=clock,
    )
    topo = {Pool.PREFILL: PoolTopology(pool=Pool.PREFILL, replicas=2, target_queue_depth=10)}

    # Flip at t=0
    a.tick(topo, queue_depths={Pool.PREFILL: 30}, in_flight={Pool.PREFILL: 0})

    # Just under dwell → still blocked
    clock.advance(119.0)
    decisions = a.tick(topo, queue_depths={Pool.PREFILL: 0}, in_flight={Pool.PREFILL: 0})
    assert decisions[0].delta == 0
    assert decisions[0].reason == "dwell_wait"

    # Cross the dwell boundary → fires
    clock.advance(2.0)
    decisions = a.tick(topo, queue_depths={Pool.PREFILL: 0}, in_flight={Pool.PREFILL: 0})
    d = decisions[0]
    assert d.delta < 0
    assert d.reason == "queue_low"


def test_min_dwell_no_flip_means_no_cooldown() -> None:
    """A tick that returns delta=0 does not start a dwell window."""
    clock = _Clock(1000.0)
    a = PoolAutoscaler(
        {Pool.PREFILL: _ctrl(kp=1.0)},
        min_dwell_s=120.0,
        clock=clock,
    )
    topo = {Pool.PREFILL: PoolTopology(pool=Pool.PREFILL, replicas=2, target_queue_depth=10)}

    # 5 stable ticks at target → delta=0 each, no dwell starts
    for _ in range(5):
        clock.advance(10.0)
        decisions = a.tick(topo, queue_depths={Pool.PREFILL: 10}, in_flight={Pool.PREFILL: 0})
        assert decisions[0].delta == 0
        assert decisions[0].reason == "stable"

    # Now a flip — should fire immediately, no dwell applies (last_flip_ts is None/0)
    clock.advance(10.0)
    decisions = a.tick(topo, queue_depths={Pool.PREFILL: 30}, in_flight={Pool.PREFILL: 0})
    assert decisions[0].delta > 0
    assert decisions[0].reason == "queue_high"


def test_min_dwell_zero_disables_feature() -> None:
    """min_dwell_s=0 disables the feature (default for tests)."""
    a = PoolAutoscaler({Pool.PREFILL: _ctrl(kp=1.0)})  # default min_dwell_s=0
    topo = {Pool.PREFILL: PoolTopology(pool=Pool.PREFILL, replicas=4, target_queue_depth=10)}

    # Flip up
    a.tick(topo, queue_depths={Pool.PREFILL: 30}, in_flight={Pool.PREFILL: 0})
    # Immediately flip down — no dwell blocking
    decisions = a.tick(topo, queue_depths={Pool.PREFILL: 0}, in_flight={Pool.PREFILL: 0})
    assert decisions[0].delta < 0
    assert decisions[0].reason == "queue_low"


def test_min_dwell_rejects_negative() -> None:
    with pytest.raises(ValueError):
        PoolAutoscaler({Pool.PREFILL: _ctrl()}, min_dwell_s=-1.0)


def test_min_dwell_property_alternating_queue() -> None:
    """Property test: alternating high/low queue cannot flip pool more than
    once per dwell window, even with continuous alternation.
    """
    clock = _Clock(1000.0)
    a = PoolAutoscaler(
        {Pool.PREFILL: _ctrl(kp=2.0)},
        min_dwell_s=120.0,
        clock=clock,
    )
    topo = {Pool.PREFILL: PoolTopology(pool=Pool.PREFILL, replicas=4, target_queue_depth=10)}

    flips_observed: list[tuple[float, int]] = []  # (time, delta)
    # Alternate queue every 10s for 600s. Without dwell: 60 flips. With 120s dwell: ≤5.
    for i in range(60):
        clock.advance(10.0)
        depth = 30 if i % 2 == 0 else 0
        decisions = a.tick(topo, queue_depths={Pool.PREFILL: depth}, in_flight={Pool.PREFILL: 0})
        d = decisions[0]
        if d.delta != 0:
            flips_observed.append((clock.t, d.delta))

    # With 120s dwell and 10s tick alternation, expect ≤5 effective flips
    # (initial flip + ~one per 120s = 1+5 = 6 max in 600s).
    assert len(flips_observed) <= 6, f"too many flips under alternating queue: {flips_observed}"


# ---------- Property tests — drain protocol (P3-3) ----------


def test_drain_blocks_scale_down_under_sustained_inflight() -> None:
    """Across many ticks with in_flight > 0, autoscaler never scales a pool
    down — even when the queue has been empty for the entire run. This is
    the load-bearing guarantee the drain protocol provides.
    """
    import random

    random.seed(1729)
    a = PoolAutoscaler({Pool.PREFILL: _ctrl(kp=2.0)})
    topo = {Pool.PREFILL: PoolTopology(pool=Pool.PREFILL, replicas=4, target_queue_depth=10)}
    in_flight_random = [random.randint(1, 8) for _ in range(50)]
    for n in in_flight_random:
        decisions = a.tick(topo, queue_depths={Pool.PREFILL: 0}, in_flight={Pool.PREFILL: n})
        d = decisions[0]
        assert d.delta >= 0, (
            f"drain violation: in_flight={n}, got delta={d.delta} (reason={d.reason})"
        )
        assert d.reason == "drain_wait", f"expected drain_wait, got {d.reason}"


def test_drain_fires_immediately_when_inflight_drops_to_zero() -> None:
    """The moment in_flight reaches 0, the queued scale-down fires."""
    a = PoolAutoscaler({Pool.PREFILL: _ctrl(kp=2.0)})
    topo = {Pool.PREFILL: PoolTopology(pool=Pool.PREFILL, replicas=4, target_queue_depth=10)}
    # 20 ticks with in_flight = 1 → drain_wait every time
    for _ in range(20):
        decisions = a.tick(topo, queue_depths={Pool.PREFILL: 0}, in_flight={Pool.PREFILL: 1})
        assert decisions[0].delta == 0
        assert decisions[0].reason == "drain_wait"
    # Single tick with in_flight = 0 → drain fires
    decisions = a.tick(topo, queue_depths={Pool.PREFILL: 0}, in_flight={Pool.PREFILL: 0})
    d = decisions[0]
    assert d.delta < 0
    assert d.drained is True
    assert d.reason == "queue_low"
