"""SLO-aware cache-aware router with admission control.

Decision rules (see 03-01-PLAN.md for full narrative):

1. INTERACTIVE requests prefer a cached pool for the same prefix;
   tiebreak by lowest queue depth.
2. BATCH requests prefer the lowest-depth pool regardless of cache.
3. RAG requests prefer a registered TIER pool (KV reuse wins).
4. Admission policy rejects requests whose chosen pool exceeds
   ``headroom_pct`` of capacity, unless ``interactive_bypass=True``
   and the request is INTERACTIVE.

P2 mitigation (CVE-2025-25183): prefix hashes are salted per-pool so an
attacker who learns one prefix digest cannot pre-compute collisions
against a different pool.  ``salt_for_pool`` defaults to a constant
empty byte string (legacy behavior); production wiring passes a callable
that returns per-pool bytes.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from control.pool import Pool, PoolState
from core.trace import RequestSpec, SloClass, WorkloadType
from obs.registry import MetricsRegistry


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


def _prefix_key(
    spec: RequestSpec,
    salt: bytes = b"",
    prefix_chars: int = 256,
) -> str:
    """Stable cache key from the first ``prefix_chars`` of the prompt + salt.

    SHA-256 over ``salt || prompt[0:prefix_chars]``; first 16 hex chars
    used to keep labels compact.  Per-pool salt makes the cache namespace
    pool-local (P2 mitigation against CVE-2025-25183).
    """
    h = hashlib.sha256(salt + spec.prompt_text[:prefix_chars].encode("utf-8")).hexdigest()
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
        salt_for_pool: Callable[[Pool], bytes] | None = None,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self._policy = policy or AdmissionPolicy()
        self._cache_max = prefix_cache_size
        # Default to empty salt (legacy behavior); production callers pass
        # a Callable[[Pool], bytes]. See control/router.py docstring + PITFALLS P2.
        self._salt_for_pool: Callable[[Pool], bytes] = salt_for_pool or (lambda _p: b"")
        self._prefix_cache: OrderedDict[str, Pool] = OrderedDict()
        self._pools: dict[Pool, PoolState] = {}
        # Optional metrics handle.  When provided, the router increments
        # ``goodputlab_cache_no_history_total`` for each cold-cache lookup
        # (P7 / RTR-04: dual-regime reporting).  When ``None`` the router
        # routes without emitting any telemetry.
        self._metrics = metrics

    # ---------- Pool registry ----------

    def register_pool(self, state: PoolState) -> None:
        self._pools[state.pool] = state

    def update_depth(self, pool: Pool, depth: int) -> None:
        if pool in self._pools:
            self._pools[pool].queue_depth = max(0, depth)

    def publish_metrics(self) -> None:
        """Snapshot LRU size + per-pool depth into the metrics registry.

        Call this from a periodic poller (e.g. once per second) when
        telemetry is wired; it is a no-op when ``metrics`` was ``None``
        on construction (back-compat with earlier tests/orchestrators).
        """
        if self._metrics is None:
            return
        # Estimate byte footprint: each LRU entry holds a str key + Pool
        # enum value.  Round up conservatively so the alert threshold
        # in P8 (>1GB or >10% router RSS) trips before OOM.
        total_bytes = 0
        for key, pool in self._prefix_cache.items():
            total_bytes += len(key.encode("utf-8")) + len(pool.value.encode("utf-8")) + 16
        self._metrics.set_prefix_index_size_bytes(float(total_bytes))
        for pool, state in self._pools.items():
            self._metrics.set_queue_depth(pool.value, state.queue_depth)

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

        # 1. Cache hit: stick to the same pool if it still has capacity.
        # Look up under each candidate pool's salt in turn, preferring the
        # lowest-depth healthy pool's hash first (cheap best-case; in the
        # worst case we inspect ~O(num_pools) entries per request).
        chosen_pool = self._select_admissible(spec)
        if chosen_pool is not None:
            salt = self._salt_for_pool(chosen_pool)
            key = _prefix_key(spec, salt)
        else:
            # Provisional key (used only in the all_pools_full response).
            key = _prefix_key(spec, self._salt_for_pool(Pool.COLOCATED))

        # Cache lookup is pool-aware: only entries under THIS pool's salt
        # match. This is the P2 mitigation — pools share no key namespace.
        if key in self._prefix_cache:
            cached_pool = self._prefix_cache[key]
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

        # Cold-cache regime: the prefix key has no prior history.  Record
        # this in telemetry so callers can bin cold vs warm lookups (P7
        # / RTR-04 dual-regime reporting).  Falls through to load-balance
        # below when no cached entry accepted the request.
        if self._metrics is not None:
            self._metrics.inc_no_history()

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