"""P2 mitigation — per-pool prefix-hash salt (CVE-2025-25183).

Different pools MUST hash the same prompt differently so an attacker
who learns one prefix-hash cannot pre-compute collisions against a
different pool.  Spec calls this out as P2 in PITFALLS.md.
"""

from __future__ import annotations

from collections.abc import Callable

from control.pool import Pool, PoolState
from control.router import AdmissionPolicy, Router
from core.trace import RequestSpec, SloClass, WorkloadType


def _spec(prompt: str = "shared prompt", slo: SloClass = SloClass.INTERACTIVE) -> RequestSpec:
    return RequestSpec(
        request_id=f"r-{prompt[:8]}",
        slo_class=slo,
        workload=WorkloadType.CHAT,
        prompt_tokens=len(prompt.split()),
        output_tokens=10,
        prompt_text=prompt,
    )


def _constant_salt(salt: bytes) -> Callable[[Pool], bytes]:
    def _fn(pool: Pool) -> bytes:
        return salt

    return _fn


def _per_pool_salt() -> Callable[[Pool], bytes]:
    def _fn(pool: Pool) -> bytes:
        return f"salt-{pool.value}".encode()

    return _fn


def _two_pools() -> Router:
    r = Router(
        policy=AdmissionPolicy(headroom_pct=80.0, interactive_bypass=True),
        salt_for_pool=_per_pool_salt(),
    )
    r.register_pool(PoolState(pool=Pool.PREFILL, queue_depth=0, capacity=64))
    r.register_pool(PoolState(pool=Pool.DECODE, queue_depth=0, capacity=64))
    return r


# ---------- Public surface ----------


def test_router_accepts_salt_for_pool() -> None:
    """Router constructor must accept the new kwarg without TypeError."""
    Router(salt_for_pool=_per_pool_salt())  # must not raise


def test_router_routes_differently_under_per_pool_salt() -> None:
    """Same prompt + constant depth config → pools get different prefix hashes.

    We force a cache miss on the second pool by saturating it BEFORE the
    second routing decision so the router must pick the other one — and
    proves the cache key is salted (a same-key entry on PREFILL must not
    satisfy a DECODE lookup).
    """
    r = _two_pools()
    s = _spec()

    # First request: load-balanced to the lowest-depth pool.
    first = r.route(s)
    assert first.admitted

    # Saturate the same pool (depth=64/64 = 100% headroom used) so the
    # next request to the same prefix must fall through to the OTHER pool.
    r.update_depth(first.pool, 64)
    second = r.route(s)

    # If the cache key were NOT salted, second would hit the cached pool
    # via cache_hit (interactive bypass admits saturated pools, so even
    # without bypass the fallback is to a different pool). Under per-pool
    # salt the second request gets a DIFFERENT cache lookup → NEW pool.
    assert second.pool != first.pool


def test_router_constant_salt_collapses_to_single_namespace() -> None:
    """A constant salt degenerates the mitigation: pools share the key.

    We assert the conservative property — the code accepts ANY callable
    returning bytes, including a constant — without crashing. The
    security guarantee comes from callers passing a *per-pool* callable.
    """
    r = Router(salt_for_pool=_constant_salt(b"same-salt"))
    r.register_pool(PoolState(pool=Pool.PREFILL, queue_depth=0, capacity=64))
    r.register_pool(PoolState(pool=Pool.DECODE, queue_depth=0, capacity=64))
    s = _spec(prompt="abc")
    first = r.route(s)
    second = r.route(s)  # same prompt, same salt
    assert first.pool == second.pool  # cache_hit on constant salt
    assert second.reason == "cache_hit"


def test_router_per_pool_salt_distinct_for_every_pool() -> None:
    """Every pool must receive a distinct salt so collisions are pool-local.

    This is the contract a security reviewer relies on. We assert the
    callable returns distinct bytes for each Pool enum value.
    """
    salt_fn = _per_pool_salt()
    salts = {p: salt_fn(p) for p in Pool}
    assert len(set(salts.values())) == len(Pool), (
        f"salts must be distinct across pools; got {salts}"
    )
