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