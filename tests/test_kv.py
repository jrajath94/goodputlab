"""Tests for kv/lmcache_client.py + kv/tier_policy.py."""

from __future__ import annotations

import pytest

from kv.lmcache_client import KvStats, MockLmcacheClient
from kv.tier_policy import TierPolicy

# ---------- MockLmcacheClient ----------


def test_mock_lmcache_lookup_miss_then_hit() -> None:
    """First lookup of a new prefix is a miss; after store() it's a hit."""
    cache = MockLmcacheClient(capacity=10, hit_probability=1.0)
    assert cache.lookup("p1") is False  # miss
    cache.store("p1")
    assert cache.lookup("p1") is True  # hit


def test_mock_lmcache_lookup_with_probabilistic_hit() -> None:
    """hit_probability=0 means every lookup misses, even if stored."""
    cache = MockLmcacheClient(capacity=10, hit_probability=0.0)
    cache.store("p1")
    assert cache.lookup("p1") is False


def test_mock_lmcache_lru_evicts_oldest_when_full() -> None:
    cache = MockLmcacheClient(capacity=2, hit_probability=1.0)
    cache.store("p1")
    cache.store("p2")
    cache.store("p3")  # evicts p1
    s = cache.stats()
    assert s.eviction_count == 1
    assert cache.lookup("p1") is False  # evicted
    assert cache.lookup("p2") is True
    assert cache.lookup("p3") is True


def test_mock_lmcache_lookup_promotes_recently_used() -> None:
    cache = MockLmcacheClient(capacity=2, hit_probability=1.0)
    cache.store("p1")
    cache.store("p2")
    cache.lookup("p1")  # promote p1 → order: [p2, p1]
    cache.store("p3")  # evicts p2 (now oldest)
    assert cache.lookup("p2") is False
    assert cache.lookup("p1") is True
    assert cache.lookup("p3") is True


def test_mock_lmcache_stats_compute_hit_rate() -> None:
    cache = MockLmcacheClient(capacity=10, hit_probability=1.0)
    cache.store("p1")
    cache.lookup("p1")  # hit
    cache.lookup("p1")  # hit
    cache.lookup("p2")  # miss (never stored)
    s = cache.stats()
    assert s.hits == 2
    assert s.misses == 1
    assert s.hit_rate == pytest.approx(2 / 3)


def test_mock_lmcache_stats_initial_zero_rate() -> None:
    cache = MockLmcacheClient()
    s = cache.stats()
    assert s.hits == 0
    assert s.misses == 0
    assert s.hit_rate == 0.0
    assert s.capacity_free_pct() == 100.0


def test_mock_lmcache_used_bytes_grow_with_store() -> None:
    cache = MockLmcacheClient(capacity=3, hit_probability=1.0)
    cache.store("p1", size_bytes=1000)
    cache.store("p2", size_bytes=2000)
    s = cache.stats()
    assert s.used_bytes == 3000


def test_mock_lmcache_used_bytes_shrink_on_evict() -> None:
    cache = MockLmcacheClient(capacity=2, hit_probability=1.0)
    cache.store("p1", size_bytes=500)
    cache.store("p2", size_bytes=500)
    cache.evict("p1")
    s = cache.stats()
    assert s.used_bytes == 500


# ---------- TierPolicy ----------


def test_tier_policy_admits_warm_rag() -> None:
    """RAG with high expected hit prob + warm tier → admit."""
    policy = TierPolicy(min_hit_rate=0.5, min_capacity_free_pct=10.0)
    stats = KvStats(
        tier="mock",
        hit_rate=0.8,
        hits=80,
        misses=20,
        capacity_bytes=1000,
        used_bytes=200,  # 80% free
        eviction_count=0,
    )
    assert policy.should_use_tier(stats, expected_hit_prob=0.85)


def test_tier_policy_rejects_when_capacity_full() -> None:
    policy = TierPolicy(min_hit_rate=0.5, min_capacity_free_pct=10.0)
    stats = KvStats(
        tier="mock",
        hit_rate=0.9,
        hits=90,
        misses=10,
        capacity_bytes=1000,
        used_bytes=950,  # 5% free < 10%
        eviction_count=0,
    )
    assert not policy.should_use_tier(stats, expected_hit_prob=0.85)
    assert policy.reject_reason(stats, 0.85) == "tier_full"


def test_tier_policy_rejects_when_tier_cold() -> None:
    policy = TierPolicy(min_hit_rate=0.5)
    stats = KvStats(
        tier="mock",
        hit_rate=0.2,  # below threshold
        hits=2,
        misses=8,
        capacity_bytes=1000,
        used_bytes=100,
        eviction_count=0,
    )
    assert not policy.should_use_tier(stats, expected_hit_prob=0.85)
    assert policy.reject_reason(stats, 0.85) == "tier_cold"


def test_tier_policy_rejects_chat_with_low_expected_hit() -> None:
    """Chat workload shouldn't burn tier capacity even if tier is warm."""
    policy = TierPolicy(min_hit_rate=0.5)
    stats = KvStats(
        tier="mock",
        hit_rate=0.9,
        hits=90,
        misses=10,
        capacity_bytes=1000,
        used_bytes=200,
        eviction_count=0,
    )
    # Chat expected hit prob is 0.3 → below threshold
    assert not policy.should_use_tier(stats, expected_hit_prob=0.3)
    assert policy.reject_reason(stats, 0.3) == "workload_low_hit_prob"


def test_tier_policy_rejects_invalid_thresholds() -> None:
    with pytest.raises(Exception):
        TierPolicy(min_hit_rate=1.5)
    with pytest.raises(Exception):
        TierPolicy(min_capacity_free_pct=-1.0)


def test_tier_policy_reject_reason_returns_none_when_admitted() -> None:
    policy = TierPolicy()
    stats = KvStats(
        tier="mock",
        hit_rate=0.9,
        hits=9,
        misses=1,
        capacity_bytes=1000,
        used_bytes=100,
        eviction_count=0,
    )
    assert policy.reject_reason(stats, expected_hit_prob=0.9) is None


def test_kv_stats_rejects_extra_fields() -> None:
    with pytest.raises(Exception):
        KvStats.model_validate(
            {
                "tier": "x",
                "hit_rate": 0.0,
                "hits": 0,
                "misses": 0,
                "capacity_bytes": 0,
                "used_bytes": 0,
                "eviction_count": 0,
                "backdoor": "yes",
            }
        )