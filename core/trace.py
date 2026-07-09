"""Canonical trace + per-request telemetry schema for GoodputLab.

Single source of truth for LOAD-05 (per-request log schema), LOAD-07
(byte-identical replay contract), and the reconciliation input that
``core/metrics.reconcile`` consumes.  Every other module — load
generators, router, autoscaler, replay engine — references these
types rather than re-defining them.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class WorkloadType(StrEnum):
    """Workload shape for the request batch."""

    CHAT = "chat"
    RAG = "rag"
    AGENTIC = "agentic"


class SloClass(StrEnum):
    """Service-level objective bucket the router uses to pick a pool."""

    INTERACTIVE = "interactive"  # TTFT p95 < 800ms required
    BATCH = "batch"  # throughput-first; degrades gracefully under load


class RequestSpec(BaseModel):
    """Static description of a request — what to send, not when."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    slo_class: SloClass
    workload: WorkloadType
    prompt_tokens: int = Field(ge=1, le=200_000)
    output_tokens: int = Field(ge=1, le=16_000)
    prompt_text: str

    @field_validator("request_id")
    @classmethod
    def _id_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("request_id must be non-empty")
        return v


class RequestTelemetry(BaseModel):
    """Per-request client-side telemetry — LOAD-05 contract.

    Timestamps are recorded with ``time.perf_counter_ns`` so they are
    monotonic on a single node.  Cross-node reconciliation requires a
    one-shot ``clock_skew_ms`` correction (see ``Trace.metadata``) and
    is enforced by ``core/metrics.reconcile``.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str
    enqueue_ts_ns: int = Field(ge=0)
    ttft_ms: float | None = None  # None when request failed before first token
    per_token_ts_ns: list[int] = Field(default_factory=list)
    completion_ts_ns: int | None = None
    status_code: int = Field(ge=100, le=599)
    error: str | None = None
    routed_pool: str | None = None  # "prefill" | "decode" | "colocated" | "tier"

    @field_validator("per_token_ts_ns")
    @classmethod
    def _per_token_monotonic(cls, v: list[int]) -> list[int]:
        prev: int | None = None
        for curr in v:
            if prev is not None and curr < prev:
                raise ValueError("per_token_ts_ns must be monotonically non-decreasing")
            prev = curr
        return v


class ArrivalConfig(BaseModel):
    """Open-loop arrival process descriptor (LOAD-04)."""

    model_config = ConfigDict(frozen=True)

    process: Literal["poisson", "on_off"]
    rate_per_sec: float = Field(gt=0)
    seed: int
    on_duration_s: float | None = Field(default=None, gt=0)
    off_duration_s: float | None = Field(default=None, gt=0)

    @field_validator("on_duration_s", "off_duration_s")
    @classmethod
    def _on_off_pair_required(cls, v: float | None, info) -> float | None:  # type: ignore[no-untyped-def]
        # Pydantic v2 passes ValidationInfo; cross-field check is below in model_validator.
        return v

    def model_post_init(self, __context: object) -> None:
        if (
            self.process == "on_off"
            and (self.on_duration_s is None or self.off_duration_s is None)
        ):
            raise ValueError("on_off process requires on_duration_s and off_duration_s")


class Trace(BaseModel):
    """A complete workload trace — specs + arrival config + provenance."""

    model_config = ConfigDict(extra="forbid")

    workload: WorkloadType
    seed: int
    duration_s: float = Field(gt=0)
    arrival: ArrivalConfig
    requests: list[RequestSpec]
    metadata: dict[str, str] = Field(default_factory=dict)

    def n_requests(self) -> int:
        return len(self.requests)

    def expected_arrivals(self) -> float:
        """Return the expected number of arrivals over ``duration_s``."""
        if self.arrival.process == "poisson":
            return self.arrival.rate_per_sec * self.duration_s
        on_frac = self.arrival.on_duration_s / (  # type: ignore[operator]
            self.arrival.on_duration_s + self.arrival.off_duration_s  # type: ignore[operator]
        )
        return self.arrival.rate_per_sec * on_frac * self.duration_s


__all__ = [
    "ArrivalConfig",
    "RequestSpec",
    "RequestTelemetry",
    "SloClass",
    "Trace",
    "WorkloadType",
]
