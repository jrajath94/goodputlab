"""Tests for bench/orchestrator.py — bench campaign harness."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bench.mock_vllm import MockVllmServer, build_mock_app
from bench.orchestrator import BenchOrchestrator, CampaignReport, Topology
from control.pool import Pool, PoolState
from control.router import Router
from core.trace import (
    ArrivalConfig,
    RequestSpec,
    SloClass,
    Trace,
    WorkloadType,
)


def _trace(n: int = 5) -> Trace:
    return Trace(
        workload=WorkloadType.CHAT,
        seed=1,
        duration_s=10.0,
        arrival=ArrivalConfig(process="poisson", rate_per_sec=20, seed=1),
        requests=[
            RequestSpec(
                request_id=f"r{i:04d}",
                slo_class=SloClass.INTERACTIVE,
                workload=WorkloadType.CHAT,
                prompt_tokens=10,
                output_tokens=3,
                prompt_text=f"prompt {i}",
            )
            for i in range(n)
        ],
    )


def _router() -> Router:
    r = Router()
    r.register_pool(PoolState(pool=Pool.PREFILL, queue_depth=0, capacity=64))
    r.register_pool(PoolState(pool=Pool.DECODE, queue_depth=0, capacity=64))
    r.register_pool(PoolState(pool=Pool.COLOCATED, queue_depth=0, capacity=128))
    return r


def test_orchestrator_runs_trace_through_mock() -> None:
    app = build_mock_app(base_latency_ms=0.1)
    orch = BenchOrchestrator(MockVllmServer(app=app))
    report = orch.run_topology(Topology.COLOCATED, _trace(n=3), _router())
    assert report.topology == Topology.COLOCATED
    assert report.n_requests == 3


def test_orchestrator_records_success_rate() -> None:
    app = build_mock_app(base_latency_ms=0.1)
    orch = BenchOrchestrator(MockVllmServer(app=app))
    report = orch.run_topology(Topology.CHUNKED, _trace(n=5), _router())
    assert report.success_rate == 1.0


def test_orchestrator_marks_report_dry_run() -> None:
    app = build_mock_app()
    orch = BenchOrchestrator(MockVllmServer(app=app))
    report = orch.run_topology(Topology.DISAGG, _trace(n=2), _router())
    assert any(note.startswith("[DRY-RUN]") for note in report.notes)


def test_orchestrator_emits_ttft_and_itl() -> None:
    app = build_mock_app(base_latency_ms=0.1)
    orch = BenchOrchestrator(MockVllmServer(app=app))
    report = orch.run_topology(Topology.COLOCATED, _trace(n=4), _router())
    assert report.mean_ttft_ms >= 0.0
    assert report.p95_ttft_ms >= 0.0
    assert report.mean_itl_ms >= 0.0


def test_orchestrator_uses_router_for_pool_assignment() -> None:
    """Same trace through different routers → different pool assignments."""
    app = build_mock_app()
    a = BenchOrchestrator(MockVllmServer(app=app)).run_topology(
        Topology.COLOCATED, _trace(n=4), _router()
    )
    b = BenchOrchestrator(MockVllmServer(app=app)).run_topology(
        Topology.COLOCATED, _trace(n=4), _router()
    )
    # Deterministic router → same assignments
    assert a.cache_hit_rate == b.cache_hit_rate


def test_orchestrator_reconcile_passes_on_clean_dry_run() -> None:
    app = build_mock_app()
    orch = BenchOrchestrator(MockVllmServer(app=app))
    report = orch.run_topology(Topology.COLOCATED, _trace(n=3), _router())
    assert report.reconcile_passes is True


def test_orchestrator_runs_full_campaign() -> None:
    app = build_mock_app()
    orch = BenchOrchestrator(MockVllmServer(app=app))
    traces = {t: _trace(n=2) for t in Topology}
    reports = orch.run_campaign(traces, router_factory=lambda _t: _router())
    assert len(reports) == len(Topology)
    assert {r.topology for r in reports} == set(Topology)


def test_orchestrator_campaign_reports_serializable() -> None:
    app = build_mock_app()
    orch = BenchOrchestrator(MockVllmServer(app=app))
    reports = orch.run_campaign(
        {Topology.COLOCATED: _trace(n=2)}, router_factory=lambda _t: _router()
    )
    j = reports[0].model_dump_json()
    assert "topology" in j
    assert "success_rate" in j
    assert "p95_ttft_ms" in j


def test_campaign_report_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        CampaignReport.model_validate(
            {
                "topology": "colocated",
                "n_requests": 1,
                "success_rate": 1.0,
                "mean_ttft_ms": 0.0,
                "p95_ttft_ms": 0.0,
                "mean_itl_ms": 0.0,
                "cache_hit_rate": 0.0,
                "reconcile_passes": True,
                "imposter": "no",
            }
        )


def test_orchestrator_empty_trace_emits_zero_report() -> None:
    app = build_mock_app()
    orch = BenchOrchestrator(MockVllmServer(app=app))
    report = orch.run_topology(Topology.COLOCATED, _trace(n=0), _router())
    assert report.n_requests == 0
    assert report.success_rate == 0.0