"""Tests for loadgen/arrival.py — open-loop arrival processes.

Covers LOAD-04 (Poisson + ON/OFF) and LOAD-07 (byte-identical replay).
"""

from __future__ import annotations

import pytest

from core.trace import (
    ArrivalConfig,
    RequestSpec,
    SloClass,
    Trace,
    WorkloadType,
)
from loadgen.arrival import (
    OnOffArrival,
    OpenLoopScheduler,
    PoissonArrival,
)


def _make_trace(n: int, arrival: ArrivalConfig, duration_s: float = 60.0) -> Trace:
    requests = [
        RequestSpec(
            request_id=f"r{i:04d}",
            slo_class=SloClass.INTERACTIVE,
            workload=WorkloadType.CHAT,
            prompt_tokens=512,
            output_tokens=128,
            prompt_text=f"prompt-{i}",
        )
        for i in range(n)
    ]
    return Trace(
        workload=WorkloadType.CHAT,
        seed=arrival.seed,
        duration_s=duration_s,
        arrival=arrival,
        requests=requests,
    )


def test_poisson_mean_within_2pct() -> None:
    """Empirical mean of expovariate draws is within 2% of 1/lambda."""
    rate = 25.0
    proc = PoissonArrival(rate_per_sec=rate, seed=7)
    offsets = proc.sample(100_000)
    # Mean inter-arrival ≈ 1/rate.
    diffs = [b - a for a, b in zip(offsets, offsets[1:])]
    mean = sum(diffs) / len(diffs)
    expected = 1.0 / rate
    assert mean == pytest.approx(expected, rel=0.02), (
        f"empirical mean {mean:.4f}s vs expected {expected:.4f}s (>2% drift)"
    )


def test_poisson_byte_identical() -> None:
    """LOAD-07: same (rate, seed, n) -> bit-identical float list."""
    a = PoissonArrival(rate_per_sec=10, seed=42).sample(1000)
    b = PoissonArrival(rate_per_sec=10, seed=42).sample(1000)
    assert a == b, "byte-identical replay contract broken"


def test_poisson_byte_identical_across_instances() -> None:
    """Independent instances with same seed must produce identical output."""
    a = PoissonArrival(rate_per_sec=3.5, seed=12345).sample(500)
    b = PoissonArrival(rate_per_sec=3.5, seed=12345).sample(500)
    assert a == b


def test_poisson_rejects_non_positive_rate() -> None:
    with pytest.raises(ValueError, match="rate_per_sec must be > 0"):
        PoissonArrival(rate_per_sec=0, seed=1)
    with pytest.raises(ValueError, match="rate_per_sec must be > 0"):
        PoissonArrival(rate_per_sec=-1, seed=1)


def test_poisson_sample_zero_returns_empty() -> None:
    assert PoissonArrival(rate_per_sec=10, seed=1).sample(0) == []


def test_on_off_emits_only_in_on_phase() -> None:
    """Arrivals fall strictly inside the ON-phase windows."""
    on_dur = 1.0
    off_dur = 1.0
    cycle = on_dur + off_dur
    proc = OnOffArrival(on_rate=50, on_dur=on_dur, off_dur=off_dur, seed=11)
    offsets = proc.sample(cycle * 4)  # 4 cycles
    # Every offset must be inside an ON window, never inside an OFF window.
    for t in offsets:
        phase = t % cycle
        assert phase < on_dur, f"arrival at t={t} fell in OFF phase (phase={phase})"


def test_on_off_byte_identical() -> None:
    """LOAD-07: same params + seed + duration -> bit-identical list."""
    a = OnOffArrival(on_rate=30, on_dur=2.0, off_dur=0.5, seed=99).sample(60.0)
    b = OnOffArrival(on_rate=30, on_dur=2.0, off_dur=0.5, seed=99).sample(60.0)
    assert a == b, "ON/OFF replay contract broken"


def test_on_off_rejects_invalid_params() -> None:
    with pytest.raises(ValueError, match="on_rate must be > 0"):
        OnOffArrival(on_rate=0, on_dur=1, off_dur=1, seed=1)
    with pytest.raises(ValueError, match="on_dur and off_dur must be > 0"):
        OnOffArrival(on_rate=1, on_dur=0, off_dur=1, seed=1)
    with pytest.raises(ValueError, match="on_dur and off_dur must be > 0"):
        OnOffArrival(on_rate=1, on_dur=1, off_dur=-0.5, seed=1)


def test_on_off_sample_zero_duration_returns_empty() -> None:
    proc = OnOffArrival(on_rate=10, on_dur=1, off_dur=1, seed=1)
    assert proc.sample(0) == []
    assert proc.sample(-5) == []


def test_scheduler_poisson_yields_in_time_order() -> None:
    arrival = ArrivalConfig(process="poisson", rate_per_sec=20, seed=5)
    trace = _make_trace(50, arrival, duration_s=10.0)
    sched = OpenLoopScheduler(trace)
    items = list(sched)
    assert len(items) == 50
    _, ts0 = items[0]
    for spec, ts in items[1:]:
        assert ts >= ts0, "scheduler must yield in non-decreasing time order"
        ts0 = ts


def test_scheduler_on_off_yields_in_time_order() -> None:
    arrival = ArrivalConfig(
        process="on_off",
        rate_per_sec=40,
        seed=5,
        on_duration_s=1.0,
        off_duration_s=0.5,
    )
    trace = _make_trace(80, arrival, duration_s=10.0)
    sched = OpenLoopScheduler(trace)
    items = list(sched)
    assert len(items) > 0
    _, ts0 = items[0]
    for spec, ts in items:
        assert ts >= ts0
        ts0 = ts


def test_scheduler_pairs_specs_in_order() -> None:
    arrival = ArrivalConfig(process="poisson", rate_per_sec=10, seed=1)
    trace = _make_trace(20, arrival)
    sched = OpenLoopScheduler(trace)
    items = list(sched)
    for i, (spec, _ts) in enumerate(items):
        assert spec.request_id == f"r{i:04d}"
