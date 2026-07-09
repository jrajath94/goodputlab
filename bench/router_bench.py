"""Router A/B benchmark — cold vs warm regime cache hit rate.

Generates request streams with controlled prefix overlap, routes each
through the ``Router``, and reports cache hit rate + pool usage.  This
is the closed-loop proof that the LRU prefix cache in ``control/router.py``
is doing useful work (RTR-04).

Cold regime: every request's prompt is a fresh paragraph → cache hits ≈ 0
Warm regime: requests share ``n_distinct_prefixes`` shared prompts → high reuse
"""

from __future__ import annotations

import random

from pydantic import BaseModel, ConfigDict, Field

from control.pool import Pool
from control.router import PoolDecision, Router
from core.trace import RequestSpec, SloClass, WorkloadType


class RegimeReport(BaseModel):
    """Outcome of one regime run (cold or warm)."""

    model_config = ConfigDict(extra="forbid")

    regime: str
    n_requests: int = Field(ge=0)
    cache_hits: int = Field(ge=0)
    cache_hit_rate: float = Field(ge=0.0, le=1.0)
    pools_used: dict[str, int]
    decisions: list[PoolDecision]

    def top_pool(self) -> Pool | None:
        if not self.pools_used:
            return None
        top_name = max(self.pools_used, key=lambda k: self.pools_used[k])
        return Pool(top_name)


def _make_prefix(rng: random.Random, idx: int, target_tokens: int = 60) -> str:
    """Generate a stable prefix indexed by ``idx`` (deterministic)."""
    words = [
        "the", "of", "and", "to", "in", "that", "with", "for", "on", "at",
        "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "should",
        "could", "may", "might", "must", "shall", "can", "this", "these",
        "those", "an", "a", "or", "but", "if", "then", "else", "when",
        "where", "why", "how", "what", "which", "who", "whom", "whose",
        "data", "system", "model", "inference", "training", "learning",
        "neural", "network", "transformer", "attention", "cache", "memory",
        "prefill", "decode", "latency", "throughput", "token", "embedding",
        "weight", "layer", "node", "edge", "graph", "tensor", "kernel",
        "compute", "scheduler", "queue", "batch", "prompt", "response",
        "context", "kv", "attention", "flash", "page", "block", "sequence",
        "request", "service", "control", "plane", "load", "balancer",
        "metric", "p99", "p95", "p50", "ttft", "itl", "tpot", "goodput",
    ]
    base = f"prefix-{idx}-" + " ".join(
        rng.choice(words) for _ in range(target_tokens)
    )
    return base


def _spec_from_prompt(rid: str, prompt: str) -> RequestSpec:
    return RequestSpec(
        request_id=rid,
        slo_class=SloClass.INTERACTIVE,
        workload=WorkloadType.CHAT,
        prompt_tokens=max(1, len(prompt.split())),
        output_tokens=32,
        prompt_text=prompt,
    )


class RouterBenchmark:
    """Drive a ``Router`` with cold/warm request streams; report metrics."""

    def __init__(self, router: Router) -> None:
        self._router = router

    def run_cold(
        self,
        n_requests: int = 100,
        prompt_token_target: int = 60,
        seed: int = 1,
    ) -> RegimeReport:
        """Every request has a unique prompt → cache hit rate ≈ 0."""
        rng = random.Random(seed)
        decisions: list[PoolDecision] = []
        for i in range(n_requests):
            prompt = _make_prefix(rng, idx=i, target_tokens=prompt_token_target)
            spec = _spec_from_prompt(f"cold-{i:04d}", prompt)
            decisions.append(self._router.route(spec))
        return self._summarize("cold", decisions)

    def run_warm(
        self,
        n_requests: int = 100,
        n_distinct_prefixes: int = 10,
        prompt_token_target: int = 60,
        seed: int = 1,
    ) -> RegimeReport:
        """Requests share ``n_distinct_prefixes`` prefixes → high cache reuse."""
        rng = random.Random(seed)
        prefixes = [
            _make_prefix(rng, idx=i, target_tokens=prompt_token_target)
            for i in range(n_distinct_prefixes)
        ]
        decisions: list[PoolDecision] = []
        for i in range(n_requests):
            base = rng.choice(prefixes)
            suffix = f" query-{i}"
            spec = _spec_from_prompt(f"warm-{i:04d}", base + suffix)
            decisions.append(self._router.route(spec))
        return self._summarize("warm", decisions)

    def _summarize(self, regime: str, decisions: list[PoolDecision]) -> RegimeReport:
        cache_hits = sum(1 for d in decisions if d.reason == "cache_hit")
        n = len(decisions)
        hit_rate = cache_hits / n if n > 0 else 0.0
        pools_used: dict[str, int] = {}
        for d in decisions:
            if d.admitted:
                pools_used[d.pool.value] = pools_used.get(d.pool.value, 0) + 1
        return RegimeReport(
            regime=regime,
            n_requests=n,
            cache_hits=cache_hits,
            cache_hit_rate=hit_rate,
            pools_used=pools_used,
            decisions=decisions,
        )


__all__ = ["RegimeReport", "RouterBenchmark"]