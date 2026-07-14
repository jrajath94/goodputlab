"""Tests for bench/matrix_aggregator.py — collects CellResult JSONs into summary.

Pure read-only. Given a directory of valid CellResult JSONs, produce:
- SummaryStats (overall)
- per-topology breakdown (topology → list[CellResult])
- CampaignResult (n_cells_completed/failed, total_duration_s, cost_usd)
- a list[CellResult] sorted by cell_id

Tests use on-disk fixture JSONs (no mocks).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from bench.matrix_aggregator import (
    CampaignReport,
    aggregate,
    per_topology_breakdown,
    write_summary,
)
from bench.schema.cell_schema import (
    CellResult,
    Mix,
    Model,
    SummaryStats,
    ThermalReading,
    Topology,
)


def _cell(
    cell_id: str,
    topology: Topology = Topology.COLOCATED,
    model: Model = Model.QWEN2_5_7B,
    rate_rps: int = 4,
    mix: Mix = Mix.CHAT,
    mean_ttft_ms: float = 80.0,
    mean_itl_ms: float = 6.0,
    success_rate: float = 1.0,
    reconcile_passes: bool = True,
    duration_s: float = 12.3,
) -> CellResult:
    return CellResult(
        cell_id=cell_id,
        topology=topology,
        model=model,
        rate_rps=rate_rps,
        mix=mix,
        n_warmup=5,
        n_measure=30,
        seed=12345,
        mean_ttft_ms=mean_ttft_ms,
        p95_ttft_ms=mean_ttft_ms * 1.5,
        mean_itl_ms=mean_itl_ms,
        success_rate=success_rate,
        cache_hit_rate=0.0,
        reconcile_passes=reconcile_passes,
        thermal=ThermalReading(gpu_temp_c=65, gpu_util_pct=80, gpu_mem_used_mb=50000),
        started_at=datetime(2026, 7, 13, 12, 0, 0),
        duration_s=duration_s,
        notes=[],
    )


def _write_cell(path: Path, result: CellResult) -> None:
    path.write_text(result.model_dump_json(indent=2))


def test_aggregate_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no CellResult JSONs"):
        aggregate(tmp_path)


def test_aggregate_skips_non_json_files(tmp_path: Path) -> None:
    cid = "a__b__rate-4__chat"
    _write_cell(tmp_path / f"{cid}.json", _cell(cid))
    (tmp_path / "README.md").write_text("# junk")
    # Partial / corrupt JSON: silently skipped (cell_id check never runs).
    (tmp_path / "partial.json").write_text("{not valid")
    results = aggregate(tmp_path)
    assert len(results) == 1  # only the one valid CellResult
    assert results[0].cell_id == cid


def test_aggregate_returns_sorted_results(tmp_path: Path) -> None:
    # Write out of order; aggregator must return by cell_id.
    _write_cell(tmp_path / "zeta.json", _cell("zeta"))
    _write_cell(tmp_path / "alpha.json", _cell("alpha"))
    _write_cell(tmp_path / "mid.json", _cell("mid"))
    results = aggregate(tmp_path)
    assert [r.cell_id for r in results] == ["alpha", "mid", "zeta"]


def test_aggregate_validates_cell_id_matches_filename(tmp_path: Path) -> None:
    """Filename must equal cell_id — guards against file rename mistakes."""
    _write_cell(tmp_path / "wrong_name.json", _cell("right_id"))
    with pytest.raises(ValueError, match="cell_id mismatch"):
        aggregate(tmp_path)


def test_aggregate_summary_stats(tmp_path: Path) -> None:
    _write_cell(tmp_path / "a.json", _cell("a", mean_ttft_ms=70.0))
    _write_cell(tmp_path / "b.json", _cell("b", mean_ttft_ms=80.0))
    _write_cell(
        tmp_path / "c.json",
        _cell("c", mean_ttft_ms=90.0, reconcile_passes=False),
    )
    results = aggregate(tmp_path)
    summary = SummaryStats.from_results(results)
    assert summary.n_cells == 3
    assert summary.n_unreconciled == 1
    assert summary.all_reconciled is False
    assert summary.mean_ttft_ms == pytest.approx(80.0)


def test_per_topology_breakdown_groups_correctly(tmp_path: Path) -> None:
    _write_cell(tmp_path / "a.json", _cell("a", topology=Topology.COLOCATED))
    _write_cell(tmp_path / "b.json", _cell("b", topology=Topology.DISAGG))
    _write_cell(tmp_path / "c.json", _cell("c", topology=Topology.COLOCATED))
    results = aggregate(tmp_path)
    grouped = per_topology_breakdown(results)
    assert sorted(grouped.keys()) == [Topology.COLOCATED, Topology.DISAGG]
    assert len(grouped[Topology.COLOCATED]) == 2
    assert len(grouped[Topology.DISAGG]) == 1


def test_write_summary_writes_summary_json(tmp_path: Path) -> None:
    _write_cell(tmp_path / "a.json", _cell("a", duration_s=10.0))
    _write_cell(tmp_path / "b.json", _cell("b", duration_s=20.0))
    report = CampaignReport(
        n_cells_completed=2,
        n_cells_failed=0,
        total_duration_s=30.0,
        cost_usd=0.05,
        pod_id="local-test",
        started_at=datetime(2026, 7, 13, 12, 0, 0),
        ended_at=datetime(2026, 7, 13, 12, 0, 30),
    )
    path = write_summary(tmp_path, report, cost_per_hour_usd=1.79)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["campaign"]["n_cells_completed"] == 2
    assert data["summary"]["n_cells"] == 2
    assert data["cost"]["per_hour_usd"] == 1.79
    assert data["cost"]["n_cells"] == 2


def test_write_summary_includes_per_topology_table(tmp_path: Path) -> None:
    _write_cell(
        tmp_path / "a.json",
        _cell("a", topology=Topology.COLOCATED, mean_ttft_ms=70.0, mean_itl_ms=5.0),
    )
    _write_cell(
        tmp_path / "b.json",
        _cell("b", topology=Topology.DISAGG, mean_ttft_ms=80.0, mean_itl_ms=6.0),
    )
    report = CampaignReport(
        n_cells_completed=2,
        n_cells_failed=0,
        total_duration_s=100.0,
        cost_usd=0.05,
        pod_id="x",
        started_at=datetime(2026, 7, 13, 12, 0, 0),
        ended_at=datetime(2026, 7, 13, 12, 1, 40),
    )
    path = write_summary(tmp_path, report, cost_per_hour_usd=1.79)
    data = json.loads(path.read_text())
    by_topo = data["per_topology"]
    assert Topology.COLOCATED.value in by_topo
    assert Topology.DISAGG.value in by_topo
    assert by_topo[Topology.COLOCATED.value]["n_cells"] == 1
    assert by_topo[Topology.COLOCATED.value]["mean_ttft_ms"] == 70.0


def test_aggregate_handles_216_cell_count_simulated(tmp_path: Path) -> None:
    """Sanity check that the aggregator handles the full 216-cell sweep size."""
    for i in range(216):
        cid = f"topo__{i:03d}__rate-4__chat"
        _write_cell(tmp_path / f"{cid}.json", _cell(cid, duration_s=10.0 + i * 0.01))
    results = aggregate(tmp_path)
    assert len(results) == 216
    summary = SummaryStats.from_results(results)
    assert summary.n_cells == 216
    assert summary.all_reconciled is True