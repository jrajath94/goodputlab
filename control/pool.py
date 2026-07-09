"""Pool topology model for the GoodputLab control plane."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Pool(StrEnum):
    """Inference pool role in a (possibly disaggregated) topology.

    PREFILL: prompt-only worker (KV prefill + NIXL transfer to DECODE)
    DECODE: token-by-token worker (consumes KV cache from PREFILL)
    COLOCATED: combined P+D on the same node (vLLM default)
    TIER: KV-tier backed pool (LMCache external store → fast prefix reuse)
    """

    PREFILL = "prefill"
    DECODE = "decode"
    COLOCATED = "colocated"
    TIER = "tier"


class PoolState(BaseModel):
    """Live snapshot of a pool's queue depth + capacity.

    The router keeps one of these per registered Pool.  The orchestrator
    updates ``queue_depth`` from scrapes of vLLM's
    ``vllm:num_requests_running`` metric (via core/metrics).
    """

    model_config = ConfigDict(extra="forbid")

    pool: Pool
    queue_depth: int = Field(default=0, ge=0)
    capacity: int = Field(default=64, gt=0)
    healthy: bool = Field(default=True)

    def headroom_used_pct(self) -> float:
        """Return queue_depth / capacity as a percentage [0, 100+]."""
        if self.capacity <= 0:
            return 100.0
        return (self.queue_depth / self.capacity) * 100.0


__all__ = ["Pool", "PoolState"]