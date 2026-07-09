"""Tests for bench/router_bench.py — cold vs warm regime A/B."""

from __future__ import annotations

from bench.router_bench import RouterBenchmark
from control.pool import Pool, PoolState
from control.router import Router


def _router() -> Router:
    r = Router()
    r.register_pool(PoolState(pool=Pool.PREFILL, queue_depth=0, capacity=128))
    r.register_pool(PoolState(pool=Pool.DECODE, queue_depth=0, capacity=128))
    r.register_pool(PoolState(pool=Pool.COLOCATED, queue_depth=0, capacity=128))
    return r


def test_cold_regime_has_near_zero_cache_hits() -> None:
    bench = RouterBenchmark(_router())
    report = bench.run_cold(n_requests=50, seed=42)
    assert report.regime == "cold"
    assert report.n_requests == 50
    # Cold regime: every prompt is unique → first request for each prefix
    # is a cache miss, subsequent (none) are hits. Hit rate should be 0.
    assert report.cache_hit_rate == 0.0


def test_warm_regime_has_high_cache_hit_rate() -> None:
    bench = RouterBenchmark(_router())
    report = bench.run_warm(n_requests=100, n_distinct_prefixes=10, seed=42)
    assert report.regime == "warm"
    assert report.n_requests == 100
    # 100 requests, 10 prefixes → 90 should hit cache after first per prefix.
    assert report.cache_hits >= 80, f"only {report.cache_hits}/100 cache hits"
    assert report.cache_hit_rate >= 0.8


def test_warm_uses_fewer_pools_than_cold() -> None:
    """Cold regime distributes load; warm sticks to one pool per prefix."""
    r_cold = _router()
    cold = RouterBenchmark(r_cold).run_cold(n_requests=30, seed=7)
    r_warm = _router()
    warm = RouterBenchmark(r_warm).run_warm(n_requests=30, n_distinct_prefixes=3, seed=7)
    # Warm: 3 distinct prefixes → at most 3 pools used (sticky cache).
    # Cold: 30 unique prefixes → potentially all 3 pools (load balanced).
    # The pool count alone isn't a fair comparison, but warm should be <= cold.
    assert len(warm.pools_used) <= len(cold.pools_used)


def test_router_benchmark_decisions_are_serializable() -> None:
    bench = RouterBenchmark(_router())
    report = bench.run_warm(n_requests=5, n_distinct_prefixes=2, seed=3)
    j = report.model_dump_json()
    assert "cache_hit_rate" in j
    assert "decisions" in j


def test_router_benchmark_seed_determinism() -> None:
    """Same seed → same decision sequence (cache warm-up is deterministic)."""
    a = RouterBenchmark(_router()).run_warm(n_requests=20, n_distinct_prefixes=3, seed=99)
    b = RouterBenchmark(_router()).run_warm(n_requests=20, n_distinct_prefixes=3, seed=99)
    pools_a = [d.pool for d in a.decisions]
    pools_b = [d.pool for d in b.decisions]
    assert pools_a == pools_b


def test_router_benchmark_warm_hit_rate_grows_with_reuse() -> None:
    """More requests per prefix → higher hit rate (saturates after first)."""
    r_a = _router()
    r_b = _router()
    a = RouterBenchmark(r_a).run_warm(n_requests=20, n_distinct_prefixes=4, seed=11)
    b = RouterBenchmark(r_b).run_warm(n_requests=100, n_distinct_prefixes=4, seed=11)
    assert b.cache_hit_rate >= a.cache_hit_rate


def test_router_benchmark_empty_warm_still_works() -> None:
    """Edge case: 0 requests → empty report with hit_rate=0."""
    bench = RouterBenchmark(_router())
    report = bench.run_warm(n_requests=0, n_distinct_prefixes=5, seed=0)
    assert report.n_requests == 0
    assert report.cache_hit_rate == 0.0


def test_regime_report_top_pool_returns_most_used() -> None:
    bench = RouterBenchmark(_router())
    report = bench.run_warm(n_requests=20, n_distinct_prefixes=2, seed=5)
    top = report.top_pool()
    assert top is not None
    # top pool must be in pools_used
    assert top.value in report.pools_used


def test_regime_report_rejects_extra_fields() -> None:
    from bench.router_bench import RegimeReport

    with_garbage = {
        "regime": "cold",
        "n_requests": 1,
        "cache_hits": 0,
        "cache_hit_rate": 0.0,
        "pools_used": {},
        "decisions": [],
        "imposter": "no",
    }
    import pytest

    with pytest.raises(Exception):
        RegimeReport.model_validate(with_garbage)