"""Tests for bench.matrix_report — sweep completion diagnostic.

Given cells_dir + MatrixSpec, report (a) expected total, (b) on-disk
count, (c) missing cells by topo/model/rate/mix, (d) corrupt or
cell_id-mismatched files. Pure read-only — does not touch the runner
or write any files.

Used after a sweep to self-diagnose: "did we run everything we said
we would?" — answers the question without manual JSON tallying.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from bench.matrix_report import SweepReport, sweep_completion_report
from bench.runpod_matrix import MatrixSpec
from bench.schema.cell_schema import (
    CellResult,
    Mix,
    Model,
    ThermalReading,
    Topology,
)


def _cell(
    cell_id: str,
    topology: Topology = Topology.COLOCATED,
    model: Model = Model.QWEN2_5_7B,
    rate_rps: int = 4,
    mix: Mix = Mix.CHAT,
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
        mean_ttft_ms=80.0,
        p95_ttft_ms=120.0,
        mean_itl_ms=6.0,
        success_rate=1.0,
        cache_hit_rate=0.0,
        reconcile_passes=True,
        thermal=ThermalReading(gpu_temp_c=65, gpu_util_pct=80, gpu_mem_used_mb=50000),
        started_at=datetime(2026, 7, 14, 12, 0, 0),
        duration_s=12.3,
        notes=[],
    )


def _write_cell(path: Path, result: CellResult) -> None:
    path.write_text(result.model_dump_json(indent=2))


def _spec_72() -> MatrixSpec:
    """1 model x 4 topos x 6 rates x 3 mixes = 72 cells (current YAML)."""
    return MatrixSpec(
        topologies=list(Topology),
        models=[Model.QWEN2_5_7B],
        rates_rps=[1, 2, 4, 8, 16, 32],
        mixes=list(Mix),
    )


def _write_cell_for_spec(tmp_path: Path, spec: MatrixSpec) -> None:
    """Write one CellResult JSON per cell in the spec."""
    for c in spec.cells():
        result = _cell(c.cell_id, c.topology, c.model, c.rate_rps, c.mix)
        _write_cell(tmp_path / f"{c.cell_id}.json", result)


def test_report_full_completion(tmp_path: Path) -> None:
    """All expected cells on disk: 100% complete, no missing, no corrupt."""
    spec = _spec_72()
    _write_cell_for_spec(tmp_path, spec)
    report = sweep_completion_report(tmp_path, spec)
    assert report.expected_total == 72
    assert report.on_disk == 72
    assert report.missing_count == 0
    assert report.missing_cell_ids == []
    assert report.missing_by_topology == {t: 0 for t in Topology}
    assert report.corrupt_or_mismatched == []
    assert report.completion_pct == pytest.approx(100.0)


def test_report_counts_overlap_with_spec_scope(tmp_path: Path) -> None:
    """Disk has 72 cells (3 models x 2 topos); spec is qwen2.5-7b only.

    Overlap is 18 (qwen2.5-7b in colocated + chunked). Reports
    completion against spec scope, not raw disk count.
    """
    spec = _spec_72()  # qwen2.5-7b only
    # colocated x 3 models (54)
    for model in Model:
        for rate in [1, 2, 4, 8, 16, 32]:
            for mix in Mix:
                cid = f"colocated__{model.value}__rate-{rate}__{mix.value}"
                _write_cell(
                    tmp_path / f"{cid}.json",
                    _cell(cid, Topology.COLOCATED, model, rate, mix),
                )
    # chunked x qwen3-1.7b (18)
    for rate in [1, 2, 4, 8, 16, 32]:
        for mix in Mix:
            cid = f"chunked__qwen3-1.7b__rate-{rate}__{mix.value}"
            _write_cell(
                tmp_path / f"{cid}.json",
                _cell(cid, Topology.CHUNKED, Model.QWEN3_1_7B, rate, mix),
            )

    report = sweep_completion_report(tmp_path, spec)
    # Spec scope: qwen2.5-7b x 4 topos x 6 x 3 = 72 expected
    assert report.expected_total == 72
    # Disk has 72 valid cell-named files; only 18 match spec scope (qwen2.5-7b colocated)
    assert report.on_disk == 72
    # Spec-scope overlap: 18 / 72 = 25%
    assert report.missing_count == 54
    assert report.completion_pct == pytest.approx(25.0)


def test_report_missing_cells_by_topology(tmp_path: Path) -> None:
    """Run only colocated, leave chunked/disagg/disagg_tier empty.

    spec = 72 (1 model x 4 topos x 6 x 3).
    disk = 18 (colocated only).
    missing: 54 — chunked=18, disagg=18, disagg_tier=18.
    """
    spec = _spec_72()
    for rate in [1, 2, 4, 8, 16, 32]:
        for mix in Mix:
            cid = f"colocated__qwen2.5-7b__rate-{rate}__{mix.value}"
            _write_cell(
                tmp_path / f"{cid}.json",
                _cell(cid, Topology.COLOCATED, Model.QWEN2_5_7B, rate, mix),
            )
    report = sweep_completion_report(tmp_path, spec)
    assert report.expected_total == 72
    assert report.on_disk == 18
    assert report.missing_count == 54
    assert report.missing_by_topology[Topology.COLOCATED] == 0
    assert report.missing_by_topology[Topology.CHUNKED] == 18
    assert report.missing_by_topology[Topology.DISAGG] == 18
    assert report.missing_by_topology[Topology.DISAGG_TIER] == 18
    assert report.completion_pct == pytest.approx(25.0)


def test_report_handles_corrupt_json(tmp_path: Path) -> None:
    """Corrupt JSONs (cell-named) are reported, not silently dropped."""
    spec = _spec_72()
    cid = "colocated__qwen2.5-7b__rate-4__chat"
    _write_cell(tmp_path / f"{cid}.json", _cell(cid))
    corrupt_name = "chunked__qwen2.5-7b__rate-4__chat.json"
    (tmp_path / corrupt_name).write_text("{not valid json")
    report = sweep_completion_report(tmp_path, spec)
    assert report.on_disk == 1
    assert report.corrupt_or_mismatched == [corrupt_name]
    assert report.missing_count == 71


def test_report_handles_cell_id_mismatch(tmp_path: Path) -> None:
    """JSON with cell_id != filename is reported as corrupt.

    Filename must match cell naming convention (contain __rate-); otherwise
    it's ignored as a non-cell artifact.
    """
    spec = _spec_72()
    _write_cell(
        tmp_path / "colocated__qwen2.5-7b__rate-4__WRONG.json",
        _cell("right_id"),
    )
    report = sweep_completion_report(tmp_path, spec)
    assert report.on_disk == 0
    assert report.corrupt_or_mismatched == [
        "colocated__qwen2.5-7b__rate-4__WRONG.json"
    ]


def test_report_empty_directory(tmp_path: Path) -> None:
    """No cells on disk: 0% complete, all expected cells missing."""
    spec = _spec_72()
    report = sweep_completion_report(tmp_path, spec)
    assert report.expected_total == 72
    assert report.on_disk == 0
    assert report.missing_count == 72
    assert report.completion_pct == 0.0


def test_report_ignores_non_json_files(tmp_path: Path) -> None:
    """README.md, .gitkeep, summary.json don't count as cells or corrupt."""
    spec = _spec_72()
    (tmp_path / "README.md").write_text("# junk")
    (tmp_path / ".gitkeep").write_text("")
    # summary.json is non-cell JSON; should be ignored, not flagged corrupt
    (tmp_path / "summary.json").write_text('{"campaign": {}, "summary": {}}')
    report = sweep_completion_report(tmp_path, spec)
    assert report.on_disk == 0
    assert report.corrupt_or_mismatched == []


def test_sweep_report_dataclass_is_frozen() -> None:
    """SweepReport is immutable — callers can't mutate the diagnostic."""
    report = SweepReport(
        expected_total=72,
        on_disk=18,
        missing_count=54,
        missing_by_topology={t: 0 for t in Topology},
        missing_cell_ids=["a", "b"],
        corrupt_or_mismatched=[],
        completion_pct=25.0,
    )
    with pytest.raises((AttributeError, Exception)):
        report.on_disk = 999  # type: ignore[misc]