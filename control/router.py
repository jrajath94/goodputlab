"""SLO-aware cache-aware router with admission control.

Decision rules (see 03-01-PLAN.md for full narrative):

1. INTERACTIVE requests prefer a cached pool for the same prefix;
   tiebreak by lowest queue depth.
2. BATCH requests prefer the lowest-depth pool regardless of cache.
3. RAG requests prefer a registered TIER pool (KV reuse wins).
4. Admission policy rejects requests whose chosen pool exceeds
   ``headroom_pct`` of capacity, unless ``interactive_bypass=True``
   and the request is INTERACTIVE.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

from pydantic import BaseModel, ConfigDict, Field

from control.pool import Pool, PoolState
from core.trace import RequestSpec, SloClass, WorkloadType


class AdmissionPolicy(BaseModel):
    """Admission control config — keeps the router stateless to inspect.

    ``headroom_pct=80`` means: reject when a pool's queue depth exceeds
    80% of its capacity.  ``interactive_bypass=True`` lets INTERACTIVE
    requests through even when pools are saturated (they get the
    lowest-depth pool regardless).
    """

    model_config = ConfigDict(extra="forbid")

    headroom_pct: float = Field(default=80.0, ge=0.0, le=100.0)
    interactive_bypass: bool = True


class PoolDecision(BaseModel):
    """Router's verdict for one ``RequestSpec``.

    ``pool`` is the chosen pool.  ``admitted=False`` means the request
    should be dropped (or queued by the caller).  ``reason`` is a
    short tag for observability.
    """

    model_config = ConfigDict(extra="forbid")

    pool: Pool
    admitted: bool
    reason: str
    prefix_hash: str | None = None


def _prefix_key(spec: RequestSpec, prefix_chars: int = 256) -> str:
    """Stable cache key from the first ``prefix_chars`` of the prompt.

    SHA-256 hex digest; first 16 chars used to keep labels compact.
    """
    h = hashlib.sha256(spec.prompt_text[:prefix_chars].encode("utf-8")).hexdigest()
    return h[:16]


class Router:
    """SLO-aware cache-aware router with admission control.

    Stateless w.r.t. requests — only state is the pool registry + LRU
    prefix→pool cache.  Thread-safe via single-writer assumption (the
    orchestrator owns updates; tests drive ``route()`` deterministically).
    """

    def __init__(
        self,
        policy: AdmissionPolicy | None = None,
        prefix_cache_size: int = 1024,
    ) -> None:
        self._policy = policy or AdmissionPolicy()
        self._cache_max = prefix_cache_size
        self._prefix_cache: OrderedDict[str, Pool] = OrderedDict()
        self._pools: dict[Pool, PoolState] = {}

    # ---------- Pool registry ----------

    def register_pool(self, state: PoolState) -> None:
        self._pools[state.pool] = state

    def update_depth(self, pool: Pool, depth: int) -> None:
        if pool in self._pools:
            self._pools[pool].queue_depth = max(0, depth)

    def pool_states(self) -> dict[Pool, PoolState]:
        return dict(self._pools)

    def cache_size(self) -> int:
        return len(self._prefix_cache)

    # ---------- Routing ----------

    def route(self, spec: RequestSpec) -> PoolDecision:
        """Pick a pool for ``spec``; return admission decision."""
        if not self._pools:
            return PoolDecision(
                pool=Pool.COLOCATED,
                admitted=False,
                reason="no_pools_registered",
            )

        key = _prefix_key(spec)

        # 1. Cache hit: stick to the same pool if it still has capacity.
        if key in self._prefix_cache:
            cached_pool = self._prefix_cache[key]
            # Mark as recently used (move to end of LRU).
            self._prefix_cache.move_to_end(key)
            state = self._pools.get(cached_pool)
            if state is not None and state.healthy and self._admissible(spec, state):
                return PoolDecision(
                    pool=cached_pool,
                    admitted=True,
                    reason="cache_hit",
                    prefix_hash=key,
                )
            # Cache hit but pool is full → fall through to load-balance.

        # 2. Pick the lowest-depth pool, iterating to find an admissible one.
        chosen_pool = self._select_admissible(spec)
        if chosen_pool is None:
            # No pool can admit this request (all unhealthy or all over headroom).
            return PoolDecision(
                pool=Pool.COLOCATED,
                admitted=False,
                reason="all_pools_full",
                prefix_hash=key,
            )

        # Record in cache only if admitted.
        self._prefix_cache[key] = chosen_pool
        if len(self._prefix_cache) > self._cache_max:
            self._prefix_cache.popitem(last=False)

        return PoolDecision(
            pool=chosen_pool,
            admitted=True,
            reason="load_balance",
            prefix_hash=key,
        )

    # ---------- Internals ----------

    def _admissible(self, spec: RequestSpec, state: PoolState) -> bool:
        """Apply admission policy to a single pool for this request."""
        if not state.healthy:
            return False
        used_pct = state.headroom_used_pct()
        if used_pct < self._policy.headroom_pct:
            return True
        # Over headroom: bypass only for interactive if enabled.
        return bool(spec.slo_class == SloClass.INTERACTIVE and self._policy.interactive_bypass)

    def _select_admissible(self, spec: RequestSpec) -> Pool | None:
        """Find the lowest-depth pool that admits this request.

        Walks all healthy pools in depth order; returns the first that
        passes ``_admissible``.  Returns ``None`` when no pool admits.
        """
        eligible = [
            (p, s) for p, s in self._pools.items() if s.healthy
        ]
        if not eligible:
            return None

        # Workload-specific preference order.
        if spec.workload == WorkloadType.RAG:
            tier_states: list[tuple[Pool, PoolState]] = [
                (p, s) for p, s in eligible if p == Pool.TIER
            ]
            if tier_states:
                tier_states.sort(key=lambda ps: ps[1].queue_depth)
                for entry in tier_states:
                    if self._admissible(spec, entry[1]):
                        return entry[0]

        eligible.sort(key=lambda ps: (ps[1].queue_depth, ps[1].pool.value))
        for entry in eligible:
            if self._admissible(spec, entry[1]):
                return entry[0]
        return None


__all__ = ["AdmissionPolicy", "PoolDecision", "Router"]