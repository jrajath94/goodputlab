"""Bench orchestrator — drives the loadgen → router → mock-vLLM pipeline.

Offline-only: every report is tagged ``[DRY-RUN]`` so the bench can be
smoke-tested in CI without burning GPU.  The real bench (RunPod H100 +
live vLLM) replaces ``MockVllmServer`` behind a flag — the rest of the
pipeline (loadgen, router, reconciler) is identical.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from bench.mock_vllm import MockVllmServer
from control.router import Router
from core.trace import RequestSpec, RequestTelemetry, Trace
from loadgen.replay import ReplayRunner


class Topology(StrEnum):
    """Benchmark campaign topology enum."""

    COLOCATED = "colocated"
    CHUNKED = "chunked"
    DISAGG = "disagg"
    DISAGG_TIER = "disagg_tier"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    k = (len(s) - 1) * pct
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


class CampaignReport(BaseModel):
    """Per-topology bench outcome."""

    model_config = ConfigDict(extra="forbid")

    topology: Topology
    n_requests: int = Field(ge=0)
    success_rate: float = Field(ge=0.0, le=1.0)
    mean_ttft_ms: float = Field(ge=0.0)
    p95_ttft_ms: float = Field(ge=0.0)
    mean_itl_ms: float = Field(ge=0.0)
    cache_hit_rate: float = Field(ge=0.0, le=1.0)
    reconcile_passes: bool
    notes: list[str] = Field(default_factory=list)


def _summary(telemetries: list[RequestTelemetry]) -> dict[str, float]:
    successes = [t for t in telemetries if t.status_code == 200]
    ttft_values = [t.ttft_ms for t in successes if t.ttft_ms is not None]
    itl_values: list[float] = []
    for t in successes:
        ts = t.per_token_ts_ns
        for i in range(1, len(ts)):
            itl_values.append((ts[i] - ts[i - 1]) / 1_000_000)
    routed = [t.routed_pool for t in telemetries if t.routed_pool is not None]
    return {
        "n_requests": float(len(telemetries)),
        "success_rate": (len(successes) / len(telemetries)) if telemetries else 0.0,
        "mean_ttft_ms": statistics.mean(ttft_values) if ttft_values else 0.0,
        "p95_ttft_ms": _percentile(ttft_values, 0.95),
        "mean_itl_ms": statistics.mean(itl_values) if itl_values else 0.0,
        "cache_hit_rate": (
            sum(1 for t in telemetries if t.routed_pool is not None) / max(len(routed), 1)
        ),
    }


class BenchOrchestrator:
    """Drives the bench pipeline; emits ``CampaignReport`` per topology."""

    DRY_RUN_TAG = "[DRY-RUN]"

    def __init__(self, mock_vllm: MockVllmServer) -> None:
        self._mock = mock_vllm

    def run_topology(
        self,
        topology: Topology,
        trace: Trace,
        router: Router,
    ) -> CampaignReport:
        """Run one trace through mock vLLM, route via ``router``, report."""
        # We use httpx.ASGITransport to talk to the mock app in-process.
        import httpx

        from loadgen.client import VllmHttpClient

        client = VllmHttpClient(
            base_url="http://mock-bench",
            model="bench-model",
            max_concurrent=8,
            transport=httpx.ASGITransport(app=self._mock.app),
        )

        runner = ReplayRunner(client)

        def routed_pool_for(_spec: RequestSpec) -> str:
            # Use the router to pick a pool; the mock always serves.
            decision = router.route(_spec)
            if decision.admitted:
                return decision.pool.value
            # If router rejects, still send to mock so we measure end-to-end.
            return "colocated"

        # Run via asyncio.run since this is the entry point.
        import asyncio

        telemetries = asyncio.run(runner.replay(trace, routed_pool_for=routed_pool_for))

        summary = _summary(telemetries)
        # Reconcile gate: in dry-run we don't have real vLLM metrics,
        # so the gate passes only when client reports all-success.
        reconcile_passes = summary["success_rate"] >= 0.99

        return CampaignReport(
            topology=topology,
            n_requests=int(summary["n_requests"]),
            success_rate=summary["success_rate"],
            mean_ttft_ms=summary["mean_ttft_ms"],
            p95_ttft_ms=summary["p95_ttft_ms"],
            mean_itl_ms=summary["mean_itl_ms"],
            cache_hit_rate=summary["cache_hit_rate"],
            reconcile_passes=reconcile_passes,
            notes=[
                self.DRY_RUN_TAG,
                f"mock_vllm_label={self._mock.label}",
            ],
        )

    def run_campaign(
        self,
        traces: dict[Topology, Trace],
        router_factory: Callable[[Topology], Router],
    ) -> list[CampaignReport]:
        """Run the full campaign: one report per topology."""
        reports: list[CampaignReport] = []
        for topology, trace in traces.items():
            router = router_factory(topology)
            reports.append(self.run_topology(topology, trace, router))
        return reports


__all__ = ["BenchOrchestrator", "CampaignReport", "Topology"]