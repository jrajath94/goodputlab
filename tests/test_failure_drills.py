"""Failure-drill tests — synthetic-fault injection (P5).

These tests prove the control plane reacts correctly to fault scenarios
that mirror real production failure modes, without needing live GPU
hardware.  Each test injects a fault at the *boundary* of one
component and asserts the next layer behaves as documented.

Coverage:

- ``test_node_failure_routes_around_unhealthy_pool`` — Router
- ``test_all_pools_down_returns_admitted_false`` — Router
- ``test_kv_stall_tier_admission_rejects_with_full_reason`` — KV tier
- ``test_kv_cold_tier_admission_rejects_with_cold_reason`` — KV tier
- ``test_spec_auto_disables_under_pathological_low_acceptance`` — Spec
- ``test_spec_topology_gate_disables_for_disagg_on_init`` — Spec
- ``test_router_falls_back_after_cache_hit_pool_dies`` — Router

These run as part of the regular pytest suite (no marker needed).  Live
end-to-end drills are described in ``docs/FAILURE_DRILLS.md`` and run on
GPU pods separately.
"""

from __future__ import annotations

from control.pool import Pool, PoolState
from control.router import AdmissionPolicy, Router
from core.trace import RequestSpec, SloClass, WorkloadType
from kv.lmcache_client import KvStats, MockLmcacheClient
from kv.tier_policy import TierPolicy
from spec.eagle import SpecDecoder, SpecPolicy

# ---------- Shared fixtures ----------


def _spec(
    rid: str = "r0",
    prompt: str = "hello world",
    slo: SloClass = SloClass.INTERACTIVE,
    workload: WorkloadType = WorkloadType.CHAT,
) -> RequestSpec:
    return RequestSpec(
        request_id=rid,
        slo_class=slo,
        workload=workload,
        prompt_tokens=max(1, len(prompt.split())),
        output_tokens=10,
        prompt_text=prompt,
    )


# ---------- Router drills ----------


def test_node_failure_routes_around_unhealthy_pool() -> None:
    """Drill: PREFILL pod dies mid-flight. Router must fall back to a healthy pool.

    Scenario: chat workload, PREFILL unhealthy, DECODE + COLOCATED healthy.
    Expectation: route() picks one of the healthy pools (not PREFILL).
    """
    r = Router(policy=AdmissionPolicy(headroom_pct=80.0, interactive_bypass=True))
    r.register_pool(PoolState(pool=Pool.PREFILL, queue_depth=5, capacity=64, healthy=False))
    r.register_pool(PoolState(pool=Pool.DECODE, queue_depth=3, capacity=64, healthy=True))
    r.register_pool(PoolState(pool=Pool.COLOCATED, queue_depth=10, capacity=128, healthy=True))

    d = r.route(_spec(workload=WorkloadType.CHAT))

    assert d.admitted, f"request should be admitted via fallback, got {d}"
    assert d.pool != Pool.PREFILL, (
        f"router chose unhealthy PREFILL after node failure: {d.pool}"
    )
    assert d.pool in {Pool.DECODE, Pool.COLOCATED}


def test_router_falls_back_after_cache_hit_pool_dies() -> None:
    """Drill: cached pool dies after a warm-up request.

    Sequence:
    1. First request with prefix X → cached to PREFILL.
    2. Mark PREFILL unhealthy (simulated pod failure).
    3. Second request with same prefix X → must NOT return cache_hit on PREFILL;
       must fall through to load-balance onto a healthy pool.
    """
    r = Router()
    r.register_pool(PoolState(pool=Pool.PREFILL, queue_depth=0, capacity=64, healthy=True))
    r.register_pool(PoolState(pool=Pool.DECODE, queue_depth=3, capacity=64, healthy=True))
    r.register_pool(PoolState(pool=Pool.COLOCATED, queue_depth=10, capacity=128, healthy=True))

    first = r.route(_spec(rid="r1", prompt="warm up the cache please"))
    assert first.admitted
    cached_pool = first.pool
    assert r.cache_size() == 1

    # Pod failure on the cached pool
    r.register_pool(PoolState(pool=cached_pool, queue_depth=0, capacity=64, healthy=False))

    second = r.route(_spec(rid="r2", prompt="warm up the cache please"))
    assert second.admitted, f"second request rejected: {second}"
    assert second.pool != cached_pool, (
        f"cache_hit returned on dead pool {cached_pool}: {second}"
    )
    assert second.reason in ("load_balance",), (
        f"expected fallthrough reason, got {second.reason}"
    )


def test_all_pools_down_returns_admitted_false() -> None:
    """Drill: every pool marked unhealthy (data center partition)."""
    r = Router()
    r.register_pool(PoolState(pool=Pool.PREFILL, queue_depth=0, capacity=64, healthy=False))
    r.register_pool(PoolState(pool=Pool.DECODE, queue_depth=0, capacity=64, healthy=False))
    r.register_pool(PoolState(pool=Pool.COLOCATED, queue_depth=0, capacity=128, healthy=False))

    d = r.route(_spec())
    assert not d.admitted
    assert d.reason == "no_pools_registered" or d.reason == "all_pools_full"


def test_rag_routes_to_tier_when_healthy() -> None:
    """Drill: RAG workload + healthy TIER → routes to TIER (not lowest-depth)."""
    r = Router()
    r.register_pool(PoolState(pool=Pool.PREFILL, queue_depth=1, capacity=64, healthy=True))
    r.register_pool(PoolState(pool=Pool.DECODE, queue_depth=2, capacity=64, healthy=True))
    r.register_pool(PoolState(pool=Pool.TIER, queue_depth=20, capacity=128, healthy=True))

    d = r.route(_spec(workload=WorkloadType.RAG))
    assert d.admitted
    assert d.pool == Pool.TIER, f"RAG should prefer TIER even at higher depth: {d}"


def test_rag_skips_tier_when_tier_dead() -> None:
    """Drill: RAG workload + TIER unhealthy → fall back to PREFILL/DECODE."""
    r = Router()
    r.register_pool(PoolState(pool=Pool.PREFILL, queue_depth=1, capacity=64, healthy=True))
    r.register_pool(PoolState(pool=Pool.DECODE, queue_depth=2, capacity=64, healthy=True))
    r.register_pool(PoolState(pool=Pool.TIER, queue_depth=0, capacity=128, healthy=False))

    d = r.route(_spec(workload=WorkloadType.RAG))
    assert d.admitted
    assert d.pool != Pool.TIER


# ---------- KV tier drills ----------


def test_kv_stall_tier_admission_rejects_with_full_reason() -> None:
    """Drill: tier is full (capacity_free_pct < threshold) → reject.

    Synthesizes the failure mode where LMCache's eviction loop falls behind
    and the tier reports < 10% free space.  The tier admission policy must
    return False with reason='tier_full', forcing the router to fall back
    to PREFILL.
    """
    policy = TierPolicy(min_hit_rate=0.5, min_capacity_free_pct=10.0)
    # Tier: full (free = 2%), warm (hit_rate = 0.9)
    stats = KvStats(
        tier="mock-lmcache",
        hit_rate=0.9,
        hits=9,
        misses=1,
        capacity_bytes=1000,
        used_bytes=980,
        eviction_count=0,
    )
    assert not policy.should_use_tier(stats, expected_hit_prob=0.8)
    assert policy.reject_reason(stats, expected_hit_prob=0.8) == "tier_full"


def test_kv_cold_tier_admission_rejects_with_cold_reason() -> None:
    """Drill: tier is cold (hit_rate below threshold) → reject with reason='tier_cold'."""
    policy = TierPolicy(min_hit_rate=0.5, min_capacity_free_pct=10.0)
    # Tier: warm capacity, cold hits (hit_rate = 0.1)
    stats = KvStats(
        tier="mock-lmcache",
        hit_rate=0.1,
        hits=1,
        misses=9,
        capacity_bytes=1000,
        used_bytes=200,
        eviction_count=0,
    )
    assert not policy.should_use_tier(stats, expected_hit_prob=0.8)
    assert policy.reject_reason(stats, expected_hit_prob=0.8) == "tier_cold"


def test_mock_client_saturates_capacity_then_evicts() -> None:
    """Drill: tier overflow path. Store beyond capacity → oldest evicted,
    used_bytes stays bounded, eviction_count increases.
    """
    c = MockLmcacheClient(capacity=3, hit_probability=1.0)
    c.store("a", size_bytes=100)
    c.store("b", size_bytes=100)
    c.store("c", size_bytes=100)
    assert c.stats().used_bytes == 300
    assert c.stats().eviction_count == 0

    c.store("d", size_bytes=100)  # should evict "a"
    assert c.stats().used_bytes == 300
    assert c.stats().eviction_count == 1
    assert not c.lookup("a")  # evicted
    assert c.lookup("d")  # present


# ---------- Spec decoder drills ----------


def test_spec_auto_disables_under_pathological_low_acceptance() -> None:
    """Drill: pathological prompt distribution tanks acceptance rate →
    spec decoder auto-disables after sliding-window mean drops below threshold.

    Real-world cause: prompts far out-of-distribution from training mix
    (e.g., code-switching, exotic Unicode) → draft head loses calibration.

    We use ``n_draft=20`` (closer to real EAGLE-3 operating point) so the
    per-round accept-rate converges close to ``acceptance_rate`` rather
    than being inflated by small-batch variance.
    """
    # 15% acceptance, well below the 40% default threshold.
    sd = SpecDecoder(
        policy=SpecPolicy(min_acceptance_rate=0.4, min_window=20),
        acceptance_rate=0.15,
        seed=7,
    )
    # Run rounds until window fills and threshold should trigger.
    for _ in range(40):
        sd.propose_and_verify(n_draft=20)
    assert sd.is_enabled is False, (
        f"spec decoder failed to auto-disable under 15% acceptance: "
        f"observed={sd.observed_acceptance_rate}"
    )


def test_spec_stays_enabled_at_healthy_acceptance() -> None:
    """Counter-drill: 80% acceptance → spec stays enabled indefinitely."""
    sd = SpecDecoder(
        policy=SpecPolicy(min_acceptance_rate=0.4, min_window=20),
        acceptance_rate=0.8,
        seed=11,
    )
    for _ in range(40):
        sd.propose_and_verify(n_draft=5)
    assert sd.is_enabled is True
    assert sd.observed_acceptance_rate >= 0.4


def test_spec_topology_gate_disables_for_disagg_on_init() -> None:
    """Drill: spec decoder refuses to operate on pure disagg topologies
    (P3 addendum — speculative decoding loses its advantage when prefill
    is already cheap and KV transfer dominates).
    """
    sd = SpecDecoder(
        policy=SpecPolicy(topology="disagg", min_acceptance_rate=0.4),
        acceptance_rate=0.9,
        seed=1,
    )
    assert sd.is_enabled is False, (
        "spec decoder must auto-disable on init for 'disagg' topology"
    )
    # Even on a high-acceptance round, stays disabled (no auto-recovery).
    outcome = sd.propose_and_verify(n_draft=5)
    assert outcome is None
    assert sd.is_enabled is False


def test_spec_topology_gate_disables_for_disagg_tier() -> None:
    """Same as above for 'disagg_tier' — KV transfer overhead dominates."""
    sd = SpecDecoder(
        policy=SpecPolicy(topology="disagg_tier", min_acceptance_rate=0.4),
        acceptance_rate=0.9,
        seed=1,
    )
    assert sd.is_enabled is False


def test_spec_topology_gate_allows_colocated_and_chunked() -> None:
    """Counter-drill: 'colocated' and 'chunked' are spec-compatible."""
    for topo in ("colocated", "chunked"):
        sd = SpecDecoder(
            policy=SpecPolicy(topology=topo, min_acceptance_rate=0.4),
            acceptance_rate=0.8,
            seed=1,
        )
        assert sd.is_enabled is True, f"topology {topo!r} should allow spec"
