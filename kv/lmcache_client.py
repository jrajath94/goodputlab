"""LMCache KV tier client abstraction.

The real LMCache runs as a sidecar (gRPC/HTTP); we don't bind to its
specific wire format here.  Instead we define a small ``LmcacheClient``
protocol + a deterministic ``MockLmcacheClient`` that lets the rest of
the codebase (router, autoscaler, bench) exercise tier-aware behavior
without GPU hardware.

Swap ``MockLmcacheClient`` for a real implementation behind the same
``lookup`` / ``stats`` / ``evict`` surface when LMCache is deployed.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class KvStats(BaseModel):
    """Snapshot of one KV tier's health."""

    model_config = ConfigDict(extra="forbid")

    tier: str
    hit_rate: float = Field(ge=0.0, le=1.0)
    hits: int = Field(ge=0)
    misses: int = Field(ge=0)
    capacity_bytes: int = Field(ge=0)
    used_bytes: int = Field(ge=0)
    eviction_count: int = Field(ge=0)

    def capacity_free_pct(self) -> float:
        """Return percentage of capacity still free [0, 100]."""
        if self.capacity_bytes <= 0:
            return 0.0
        return max(0.0, 100.0 * (1.0 - self.used_bytes / self.capacity_bytes))


class LmcacheClient(Protocol):
    """Surface every tier-aware caller relies on."""

    def lookup(self, prefix_hash: str) -> bool: ...
    def stats(self) -> KvStats: ...
    def evict(self, prefix_hash: str) -> None: ...
    def store(self, prefix_hash: str, size_bytes: int = 4096) -> None: ...


class MockLmcacheClient:
    """In-memory LRU tier with a fixed probability of returning hits.

    - ``capacity``: max number of entries (LRU eviction past this)
    - ``hit_probability``: when an entry is present, probability the
      caller sees a hit (for modeling stale entries, partial hits, etc.)
    """

    def __init__(self, capacity: int = 100, hit_probability: float = 0.9) -> None:
        self._capacity = capacity
        self._hit_prob = hit_probability
        self._store: OrderedDict[str, int] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._used_bytes = 0
        self._tier_name = "mock-lmcache"

    # ---------- Public surface ----------

    def lookup(self, prefix_hash: str) -> bool:
        if prefix_hash in self._store:
            self._store.move_to_end(prefix_hash)
            # Probabilistic hit (model stale entries / partial reuse).
            import random as _r

            if _r.random() < self._hit_prob:
                self._hits += 1
                return True
            self._misses += 1
            return False
        self._misses += 1
        return False

    def stats(self) -> KvStats:
        total = self._hits + self._misses
        rate = (self._hits / total) if total > 0 else 0.0
        return KvStats(
            tier=self._tier_name,
            hit_rate=rate,
            hits=self._hits,
            misses=self._misses,
            capacity_bytes=self._capacity * 4096,
            used_bytes=self._used_bytes,
            eviction_count=self._evictions,
        )

    def evict(self, prefix_hash: str) -> None:
        if prefix_hash in self._store:
            self._used_bytes -= self._store.pop(prefix_hash)

    def store(self, prefix_hash: str, size_bytes: int = 4096) -> None:
        if prefix_hash in self._store:
            self._store.move_to_end(prefix_hash)
            return
        while len(self._store) >= self._capacity:
            _, old_size = self._store.popitem(last=False)
            self._used_bytes -= old_size
            self._evictions += 1
        self._store[prefix_hash] = size_bytes
        self._used_bytes += size_bytes

    # ---------- Test helpers ----------

    @property
    def tier_name(self) -> str:
        return self._tier_name

    def set_tier_name(self, name: str) -> None:
        self._tier_name = name


__all__ = ["KvStats", "LmcacheClient", "MockLmcacheClient"]