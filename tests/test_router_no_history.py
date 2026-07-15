"""Tests for the cache_aware_router_looked_up_no_history counter (RTR cold-cache distinguisher).

Per ROADMAP.md Phase 4 success criterion #3:

> "`cache_aware_router_looked_up_no_history` counter distinguishes
> cold-cache from cache-miss"

The Router may hit the LRU prefix cache (warm regime) or fall through
to load-balance (cold regime).  The counter must increment only on the
cold path.  This guards P7 (cold-cache false confidence) by giving the
observability layer a way to bin requests by regime.
"""

from __future__ import annotations

from control.pool import Pool, PoolState
from control.router import Router
from core.trace import RequestSpec, SloClass, WorkloadType
from obs.registry import MetricsRegistry


def _spec(rid: str, prompt: str) -> RequestSpec:
    return RequestSpec(
        request_id=rid,
        slo_class=SloClass.INTERACTIVE,
        workload=WorkloadType.CHAT,
        prompt_tokens=max(1, len(prompt.split())),
        output_tokens=10,
        prompt_text=prompt,
    )


def _router_with_metrics() -> tuple[Router, MetricsRegistry]:
    metrics = MetricsRegistry()
    r = Router(metrics=metrics)
    r.register_pool(PoolState(pool=Pool.PREFILL, queue_depth=5, capacity=64))
    r.register_pool(PoolState(pool=Pool.DECODE, queue_depth=3, capacity=64))
    return r, metrics


def _counter_value(metrics: MetricsRegistry) -> float:
    """Read the current value of no_history, irrespective of label structure."""
    samples = list(metrics.no_history.collect())[0].samples
    assert samples, "no_history counter has no samples"
    return samples[0].value


def test_no_history_increments_on_first_seen_prefix() -> None:
    """First route() for an unseen prompt must bump the cold-regime counter."""
    r, metrics = _router_with_metrics()
    before = _counter_value(metrics)
    r.route(_spec(rid="r0", prompt="fresh prompt alpha"))
    after = _counter_value(metrics)
    assert after == before + 1


def test_no_history_does_not_increment_on_cache_hit() -> None:
    """Second route() for the SAME prompt is a cache hit — counter stays flat."""
    r, metrics = _router_with_metrics()
    r.route(_spec(rid="r0", prompt="warm regime beta"))  # first → +1
    before = _counter_value(metrics)
    r.route(_spec(rid="r1", prompt="warm regime beta"))  # cache hit → no-op
    after = _counter_value(metrics)
    assert after == before


def test_no_history_increments_per_unique_new_prefix() -> None:
    """Distinct prompts each increment exactly once on first sight."""
    r, metrics = _router_with_metrics()
    r.route(_spec(rid="r0", prompt="unique one"))
    r.route(_spec(rid="r1", prompt="unique two"))
    r.route(_spec(rid="r2", prompt="unique three"))
    # Three first-sight routes → counter advanced by 3
    r.route(_spec(rid="r3", prompt="unique one"))  # cache hit
    r.route(_spec(rid="r4", prompt="unique two"))  # cache hit
    final = _counter_value(metrics)
    assert final == 3.0


def test_no_history_skipped_when_no_registry_injected() -> None:
    """Back-compat: Router with metrics=None must still route, just without telemetry."""
    r = Router()  # no metrics
    r.register_pool(PoolState(pool=Pool.PREFILL, queue_depth=1, capacity=64))
    d = r.route(_spec(rid="r0", prompt="no registry"))
    assert d.admitted is True  # routed cleanly without raising
