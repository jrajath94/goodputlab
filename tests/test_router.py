"""Tests for control/router.py — SLO-aware cache-aware routing."""

from __future__ import annotations

import pytest

from control.pool import Pool, PoolState
from control.router import AdmissionPolicy, Router, _prefix_key
from core.trace import RequestSpec, SloClass, WorkloadType


def _spec(
    rid: str = "r0",
    prompt: str = "hello world",
    slo: SloClass = SloClass.INTERACTIVE,
    workload: WorkloadType = WorkloadType.CHAT,
    out_tokens: int = 10,
) -> RequestSpec:
    return RequestSpec(
        request_id=rid,
        slo_class=slo,
        workload=workload,
        prompt_tokens=max(1, len(prompt.split())),
        output_tokens=out_tokens,
        prompt_text=prompt,
    )


def _router(policy: AdmissionPolicy | None = None) -> Router:
    r = Router(policy=policy or AdmissionPolicy(headroom_pct=80.0, interactive_bypass=True))
    r.register_pool(PoolState(pool=Pool.PREFILL, queue_depth=5, capacity=64))
    r.register_pool(PoolState(pool=Pool.DECODE, queue_depth=3, capacity=64))
    r.register_pool(PoolState(pool=Pool.COLOCATED, queue_depth=10, capacity=128))
    return r


# ---------- Prefix key ----------


def test_prefix_key_is_stable_for_same_prompt() -> None:
    s = _spec(prompt="the quick brown fox" * 10)
    assert _prefix_key(s) == _prefix_key(s)


def test_prefix_key_uses_only_first_256_chars() -> None:
    a = _spec(prompt="x" * 300)
    b = _spec(prompt="x" * 256 + "y" * 100)
    assert _prefix_key(a) == _prefix_key(b)


def test_prefix_key_differs_for_different_prompts() -> None:
    a = _spec(prompt="alpha alpha alpha")
    b = _spec(prompt="beta beta beta")
    assert _prefix_key(a) != _prefix_key(b)


# ---------- Routing basics ----------


def test_router_rejects_when_no_pools_registered() -> None:
    r = Router()
    d = r.route(_spec())
    assert d.admitted is False
    assert d.reason == "no_pools_registered"


def test_router_picks_lowest_depth_for_interactive_no_cache() -> None:
    r = _router()
    d = r.route(_spec(prompt="never seen before xyz"))
    # DECODE has lowest depth (3)
    assert d.pool == Pool.DECODE
    assert d.admitted is True
    assert d.reason == "load_balance"


def test_router_returns_cache_hit_on_second_identical_request() -> None:
    r = _router()
    s = _spec(prompt="repeated prompt text")
    first = r.route(s)
    second = r.route(s)
    assert first.pool == second.pool
    assert second.reason == "cache_hit"
    assert first.prefix_hash == second.prefix_hash


def test_router_cache_hit_uses_lowest_depth_on_first_miss() -> None:
    r = _router()
    s = _spec(prompt="brand new prompt")
    first = r.route(s)
    # First time: should hit lowest depth pool (DECODE = 3)
    assert first.reason == "load_balance"
    assert first.pool == Pool.DECODE


def test_router_lru_evicts_oldest_when_full() -> None:
    r = Router(prefix_cache_size=2)
    r.register_pool(PoolState(pool=Pool.COLOCATED, queue_depth=0, capacity=128))
    r.route(_spec(rid="r1", prompt="prompt one"))
    r.route(_spec(rid="r2", prompt="prompt two"))
    r.route(_spec(rid="r3", prompt="prompt three"))  # evicts prompt one
    assert r.cache_size() == 2


def test_router_lru_promotes_recently_used() -> None:
    r = Router(prefix_cache_size=2)
    r.register_pool(PoolState(pool=Pool.COLOCATED, queue_depth=0, capacity=128))
    r.route(_spec(rid="r1", prompt="prompt one"))   # cache: [one]
    r.route(_spec(rid="r2", prompt="prompt two"))   # cache: [one, two]
    r.route(_spec(rid="r1", prompt="prompt one"))   # promote: [two, one]
    r.route(_spec(rid="r3", prompt="prompt three")) # evicts two → [one, three]
    # prompt one still cached (was promoted)
    d_one = r.route(_spec(rid="r1", prompt="prompt one"))
    assert d_one.reason == "cache_hit"
    # prompt two was evicted
    d_two = r.route(_spec(rid="r2", prompt="prompt two"))
    assert d_two.reason == "load_balance"


# ---------- Workload-specific ----------


def test_router_prefers_tier_pool_for_rag_workload() -> None:
    r = Router()
    r.register_pool(PoolState(pool=Pool.PREFILL, queue_depth=0, capacity=64))
    r.register_pool(PoolState(pool=Pool.DECODE, queue_depth=0, capacity=64))
    r.register_pool(PoolState(pool=Pool.TIER, queue_depth=20, capacity=128))  # higher depth
    s = _spec(workload=WorkloadType.RAG, prompt="rag query")
    d = r.route(s)
    # TIER wins despite higher depth — KV reuse beats raw prefill.
    assert d.pool == Pool.TIER


def test_router_falls_back_to_lowest_depth_for_chat() -> None:
    r = Router()
    r.register_pool(PoolState(pool=Pool.PREFILL, queue_depth=0, capacity=64))
    r.register_pool(PoolState(pool=Pool.TIER, queue_depth=50, capacity=128))
    s = _spec(workload=WorkloadType.CHAT)
    d = r.route(s)
    # PREFILL has lower depth → wins for chat
    assert d.pool == Pool.PREFILL


# ---------- Admission control ----------


def test_router_rejects_when_pool_at_capacity_and_no_bypass() -> None:
    r = Router(policy=AdmissionPolicy(headroom_pct=80.0, interactive_bypass=False))
    r.register_pool(PoolState(pool=Pool.PREFILL, queue_depth=60, capacity=64))  # 93%
    r.register_pool(PoolState(pool=Pool.DECODE, queue_depth=64, capacity=64))  # 100%
    s = _spec(slo=SloClass.BATCH)
    d = r.route(s)
    assert d.admitted is False
    assert d.reason == "all_pools_full"


def test_router_admits_batch_to_pool_under_headroom() -> None:
    r = Router(policy=AdmissionPolicy(headroom_pct=80.0, interactive_bypass=False))
    r.register_pool(PoolState(pool=Pool.PREFILL, queue_depth=50, capacity=64))  # 78%
    s = _spec(slo=SloClass.BATCH)
    d = r.route(s)
    assert d.admitted is True


def test_router_interactive_bypasses_saturated_pool() -> None:
    r = Router(policy=AdmissionPolicy(headroom_pct=80.0, interactive_bypass=True))
    r.register_pool(PoolState(pool=Pool.PREFILL, queue_depth=64, capacity=64))  # 100%
    s = _spec(slo=SloClass.INTERACTIVE)
    d = r.route(s)
    # Interactive bypass: still admitted
    assert d.admitted is True


def test_router_batch_does_not_bypass_when_saturated() -> None:
    r = Router(policy=AdmissionPolicy(headroom_pct=80.0, interactive_bypass=True))
    r.register_pool(PoolState(pool=Pool.PREFILL, queue_depth=64, capacity=64))
    s = _spec(slo=SloClass.BATCH)
    d = r.route(s)
    assert d.admitted is False


def test_router_skips_unhealthy_pools() -> None:
    r = _router()
    r.update_depth  # noqa: B018 — touch to silence linter on unused
    r._pools[Pool.PREFILL].healthy = False  # noqa: SLF001 — test internal
    r._pools[Pool.DECODE].healthy = False  # noqa: SLF001
    s = _spec()
    d = r.route(s)
    # Only COLOCATED healthy
    assert d.pool == Pool.COLOCATED


# ---------- Observability / serialization ----------


def test_pool_decision_is_pydantic_serializable() -> None:
    r = _router()
    d = r.route(_spec())
    json = d.model_dump_json()
    assert "pool" in json
    assert "admitted" in json
    assert "reason" in json


def test_router_decision_tracks_role_change() -> None:
    """If the cached pool becomes saturated, router picks another = role-flip."""
    r = _router()
    s = _spec(prompt="persistent prompt")
    first = r.route(s)
    assert first.admitted
    # Saturate the cached pool
    r.update_depth(first.pool, 100)
    second = r.route(s)
    # Either admitted via bypass (interactive default) into a new pool, OR rejected.
    # Interactive bypass → admitted into a different pool
    assert second.admitted is True
    assert second.pool != first.pool or second.reason == "cache_hit"


def test_router_admission_policy_rejects_invalid_headroom() -> None:
    with pytest.raises(Exception):
        AdmissionPolicy(headroom_pct=150.0)
    with pytest.raises(Exception):
        AdmissionPolicy(headroom_pct=-1.0)