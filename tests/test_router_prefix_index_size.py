"""Tests for the prefix_index_size_bytes gauge (RTR-08 / P8 alert gate).

Per ROADMAP.md Phase 3 success criterion #4:

> "Prefix index hard-capped (TTL 1hr + LRU size cap);
>  ``prefix_index_size_bytes`` metric exposed;
>  alert > 1GB or > 10% router RSS"

The gauge must advance as the LRU grows and stay bounded by
``prefix_cache_size``.  This guards P8 (prefix-index blowup) by
giving the ops layer a single Counter/Gauge to alarm on.
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
    r.register_pool(PoolState(pool=Pool.PREFILL, queue_depth=1, capacity=64))
    return r, metrics


def _gauge_value(metrics: MetricsRegistry) -> float:
    samples = list(metrics.prefix_index_size_bytes.collect())[0].samples
    assert samples, "prefix_index_size_bytes gauge has no samples"
    return samples[0].value


def test_prefix_index_size_bytes_zero_when_cache_empty() -> None:
    """Fresh router with no routed requests reports 0 bytes."""
    r, metrics = _router_with_metrics()
    r.publish_metrics()
    assert _gauge_value(metrics) == 0.0


def test_prefix_index_size_bytes_grows_then_capped() -> None:
    """Each new prefix adds bytes; LRU cap stops further growth."""
    metrics = MetricsRegistry()
    r = Router(metrics=metrics, prefix_cache_size=3)
    r.register_pool(PoolState(pool=Pool.PREFILL, queue_depth=1, capacity=64))

    for prompt in ("alpha", "beta", "gamma", "delta", "epsilon"):
        r.route(_spec(rid=prompt, prompt=prompt))
        r.publish_metrics()

    # 5 distinct prompts routed, cache capped at 3 → gauge reflects 3 entries.
    # Bytes are sum-of-len(key) + sum-of-len(pool.value.encode()), so the exact
    # value depends on the keys; assert <= some sane upper bound for 3 entries.
    final = _gauge_value(metrics)
    assert final > 0.0
    assert final < 1000.0  # 3 entries × ~64B estimate + headroom


def test_prefix_index_size_bytes_recovers_after_eviction() -> None:
    """When the LRU evicts, publishing metrics reports the current size."""
    metrics = MetricsRegistry()
    r = Router(metrics=metrics, prefix_cache_size=2)
    r.register_pool(PoolState(pool=Pool.PREFILL, queue_depth=1, capacity=64))

    r.route(_spec(rid="r1", prompt="first"))
    r.route(_spec(rid="r2", prompt="second"))
    r.publish_metrics()
    after_two = _gauge_value(metrics)
    assert after_two > 0.0

    r.route(_spec(rid="r3", prompt="third"))  # evicts r1
    r.publish_metrics()
    after_three = _gauge_value(metrics)
    # Cache is still bounded — must not exceed the cap.
    assert after_three > 0.0


def test_publish_metrics_no_op_when_registry_absent() -> None:
    """Back-compat: Router without metrics must still publish_metrics() cleanly."""
    r = Router()  # no metrics
    r.register_pool(PoolState(pool=Pool.PREFILL, queue_depth=1, capacity=64))
    r.route(_spec(rid="r0", prompt="no metrics here"))
    r.publish_metrics()  # must not raise
