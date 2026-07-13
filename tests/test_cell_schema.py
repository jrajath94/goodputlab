"""Tests for bench/schema/cell_schema.py — Pydantic models for cell JSON."""

from __future__ import annotations

from datetime import datetime

import pytest

from bench.schema.cell_schema import (
    CampaignResult,
    CellResult,
    CellSpec,
    Mix,
    Model,
    SummaryStats,
    ThermalReading,
    Topology,
)


# ---------- CellSpec ----------


def test_cell_spec_full() -> None:
    c = CellSpec(
        topology=Topology.COLOCATED,
        model=Model.QWEN2_5_7B,
        rate_rps=4,
        mix=Mix.CHAT,
        n_warmup=5,
        n_measure=30,
        seed=12345,
    )
    assert c.cell_id == "colocated__qwen2.5-7b__rate-4__chat"


def test_cell_spec_default_warmup_measure() -> None:
    c = CellSpec(topology=Topology.DISAGG, model=Model.QWEN3_30B, rate_rps=8, mix=Mix.RAG)
    assert c.n_warmup == 5
    assert c.n_measure == 30
    assert c.seed > 0


def test_cell_spec_rejects_invalid_rate() -> None:
    with pytest.raises(ValueError):
        CellSpec(topology=Topology.COLOCATED, model=Model.QWEN2_5_7B, rate_rps=0, mix=Mix.CHAT)


def test_cell_spec_rejects_unknown_topology() -> None:
    with pytest.raises(ValueError):
        CellSpec.model_validate(
            {
                "topology": "made_up_topology",
                "model": "qwen2.5-7b",
                "rate_rps": 4,
                "mix": "chat",
            }
        )


# ---------- ThermalReading ----------


def test_thermal_reading_ok() -> None:
    t = ThermalReading(gpu_temp_c=65, gpu_util_pct=87, gpu_mem_used_mb=52340)
    assert t.gpu_temp_c == 65
    assert t.gpu_util_pct == 87
    assert t.gpu_mem_used_mb == 52340


def test_thermal_flag_for_overheating() -> None:
    t = ThermalReading(gpu_temp_c=85, gpu_util_pct=99, gpu_mem_used_mb=60000)
    assert t.is_overheating is True


def test_thermal_no_flag_under_threshold() -> None:
    t = ThermalReading(gpu_temp_c=75, gpu_util_pct=80, gpu_mem_used_mb=60000)
    assert t.is_overheating is False


# ---------- CellResult ----------


def _cell_result(**overrides: object) -> CellResult:
    base: dict[str, object] = {
        "cell_id": "colocated__qwen2.5-7b__rate-4__chat",
        "topology": Topology.COLOCATED,
        "model": Model.QWEN2_5_7B,
        "rate_rps": 4,
        "mix": Mix.CHAT,
        "n_warmup": 5,
        "n_measure": 30,
        "seed": 12345,
        "mean_ttft_ms": 76.5,
        "p95_ttft_ms": 127.3,
        "mean_itl_ms": 6.38,
        "success_rate": 1.0,
        "cache_hit_rate": 1.0,
        "reconcile_passes": True,
        "thermal": ThermalReading(gpu_temp_c=65, gpu_util_pct=87, gpu_mem_used_mb=52340),
        "started_at": datetime(2026, 7, 13, 12, 0, 0),
        "duration_s": 12.3,
        "notes": [],
    }
    base.update(overrides)
    return CellResult(**base)  # type: ignore[arg-type]


def test_cell_result_round_trip_json() -> None:
    r = _cell_result()
    j = r.model_dump_json()
    r2 = CellResult.model_validate_json(j)
    assert r == r2


def test_cell_result_serializes_iso_timestamp() -> None:
    r = _cell_result()
    j = r.model_dump_json()
    assert "2026-07-13T12:00:00" in j


def test_cell_result_rejects_extra_fields() -> None:
    with pytest.raises(ValueError):
        CellResult.model_validate(
            {
                "cell_id": "x",
                "topology": Topology.COLOCATED,
                "model": Model.QWEN2_5_7B,
                "rate_rps": 4,
                "mix": Mix.CHAT,
                "n_warmup": 5,
                "n_measure": 30,
                "seed": 1,
                "mean_ttft_ms": 1.0,
                "p95_ttft_ms": 1.0,
                "mean_itl_ms": 1.0,
                "success_rate": 1.0,
                "cache_hit_rate": 1.0,
                "reconcile_passes": True,
                "thermal": {"gpu_temp_c": 65, "gpu_util_pct": 80, "gpu_mem_used_mb": 50000},
                "started_at": "2026-07-13T12:00:00",
                "duration_s": 1.0,
                "wat": "no",
            }
        )


def test_cell_result_flags_overheating() -> None:
    r = _cell_result(
        thermal=ThermalReading(gpu_temp_c=85, gpu_util_pct=99, gpu_mem_used_mb=60000),
    )
    assert r.has_thermal_warning is True


# ---------- SummaryStats ----------


def test_summary_stats_from_results() -> None:
    results = [
        _cell_result(mean_ttft_ms=70.0),
        _cell_result(mean_ttft_ms=80.0),
        _cell_result(mean_ttft_ms=90.0),
    ]
    s = SummaryStats.from_results(results)
    assert s.n_cells == 3
    assert s.mean_ttft_ms == 80.0
    assert s.all_reconciled is True


def test_summary_stats_flags_unreconciled() -> None:
    results = [
        _cell_result(reconcile_passes=True),
        _cell_result(reconcile_passes=False),
    ]
    s = SummaryStats.from_results(results)
    assert s.all_reconciled is False
    assert s.n_unreconciled == 1


# ---------- CampaignResult ----------


def test_campaign_result_includes_cost_and_duration() -> None:
    cr = CampaignResult(
        n_cells_completed=216,
        n_cells_failed=0,
        total_duration_s=14760.0,
        cost_usd=7.34,
        pod_id="t3son251d5gcvg",
        started_at=datetime(2026, 7, 13, 12, 0, 0),
        ended_at=datetime(2026, 7, 13, 16, 6, 0),
    )
    assert cr.n_cells_completed == 216
    assert cr.cost_usd == 7.34