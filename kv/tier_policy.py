"""Tier admission policy — should this request go through LMCache?

Three rejection rules (all must pass to admit):
1. Tier has at least ``min_capacity_free_pct`` free space.
2. Observed ``hit_rate`` ≥ ``min_hit_rate`` (tier is warm enough to be useful).
3. Workload-prior ``expected_hit_prob`` ≥ ``min_hit_rate`` (don't route
   workloads that historically don't benefit — chat vs RAG).

The router calls ``should_use_tier(stats, expected_hit_prob)`` before
falling through to PREFILL.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from kv.lmcache_client import KvStats


class TierPolicy(BaseModel):
    """Admission policy for routing to a KV tier pool."""

    model_config = ConfigDict(extra="forbid")

    min_hit_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    min_capacity_free_pct: float = Field(default=10.0, ge=0.0, le=100.0)

    def should_use_tier(self, stats: KvStats, expected_hit_prob: float) -> bool:
        """Return True iff all three admission checks pass."""
        if stats.capacity_free_pct() < self.min_capacity_free_pct:
            return False
        if stats.hit_rate < self.min_hit_rate:
            return False
        return not expected_hit_prob < self.min_hit_rate

    def reject_reason(
        self, stats: KvStats, expected_hit_prob: float
    ) -> str | None:
        """Return the first rejection reason, or None if admitted."""
        if stats.capacity_free_pct() < self.min_capacity_free_pct:
            return "tier_full"
        if stats.hit_rate < self.min_hit_rate:
            return "tier_cold"
        if expected_hit_prob < self.min_hit_rate:
            return "workload_low_hit_prob"
        return None


__all__ = ["TierPolicy"]