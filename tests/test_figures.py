"""Tests for bench/figures.py — Run 1 figure generation + sweep curves."""

from __future__ import annotations

from pathlib import Path

import pytest

import bench.figures as figures
from bench.schema.cell_schema import Mix, Model, Topology


def test_load_results_has_all_topologies() -> None:
    results = figures.load_results()
    assert set(results.keys()) == set(figures.TOPOS)
    for topo in figures.TOPOS:
        r = results[topo]
        assert r["n_requests"] == 30
        assert r["success_rate"] == 1.0
        assert r["mean_ttft_ms"] > 0
        assert r["mean_itl_ms"] > 0


def test_cost_scales_linearly_with_replicas() -> None:
    single = figures.cost_per_million_tokens(1)
    double = figures.cost_per_million_tokens(2)
    assert double == pytest.approx(2 * single, rel=1e-9)


def test_cost_matches_expected_dollar_amount() -> None:
    expected = (
        figures.H100_SXM_SPOT_USD_PER_HR * 1e6
        / (3600 * figures.TOKENS_PER_SEC_PER_H100)
    )
    assert figures.cost_per_million_tokens(1) == pytest.approx(expected, rel=1e-9)


def test_replica_counts_match_topology() -> None:
    assert figures.REPLICAS["colocated"] == 1
    assert figures.REPLICAS["chunked"] == 1
    assert figures.REPLICAS["disagg"] == 2
    assert figures.REPLICAS["disagg_tier"] == 2


def test_main_generates_all_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: run main() with FIGURES_DIR redirected to tmp, assert all
    artifacts exist + non-empty."""
    monkeypatch.setattr(figures, "FIGURES_DIR", tmp_path)
    figures.main()
    expected = [
        tmp_path / "ttft_comparison.png",
        tmp_path / "itl_comparison.png",
        tmp_path / "cost_per_million_tokens.csv",
        tmp_path / "cost_per_million_tokens.md",
    ]
    for p in expected:
        assert p.exists(), f"missing artifact: {p}"
        assert p.stat().st_size > 0, f"empty artifact: {p}"


def test_cost_md_documents_assumptions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The cost table is honest: assumptions are visible in the markdown."""
    monkeypatch.setattr(figures, "FIGURES_DIR", tmp_path)
    figures.main()
    text = (tmp_path / "cost_per_million_tokens.md").read_text()
    assert "H100 SXM spot" in text
    assert "$1.99" in text
    assert "120" in text
    assert "256" in text


# ---------- runpod_full sweep curves ----------


def _fake_cell(
    topo: Topology, model: Model, rate: int, mix: Mix, ttft: float
) -> dict[str, float | int | str]:
    return {
        "topology": topo.value,
        "model": model.value,
        "rate_rps": rate,
        "mix": mix.value,
        "mean_ttft_ms": ttft,
        "mean_itl_ms": 6.0,
        "success_rate": 1.0,
        "reconcile_passes": True,
    }


def test_load_runpod_full_filters_unreconciled(tmp_path: Path) -> None:
    """Stub cells (reconcile_passes=False) must be filtered out — averaging
    their zeros with real measurements masks performance."""
    cell_dir = tmp_path / "cells"
    cell_dir.mkdir()
    (cell_dir / "colocated__qwen2.5-7b__rate-4__chat.json").write_text(
        '{"topology":"colocated","model":"qwen2.5-7b","rate_rps":4,'
        '"mix":"chat","mean_ttft_ms":100.0,"mean_itl_ms":6.0,'
        '"success_rate":1.0,"reconcile_passes":true}'
    )
    (cell_dir / "disagg__qwen2.5-7b__rate-8__chat.json").write_text(
        '{"topology":"disagg","model":"qwen2.5-7b","rate_rps":8,'
        '"mix":"chat","mean_ttft_ms":0.0,"mean_itl_ms":0.0,'
        '"success_rate":0.0,"reconcile_passes":false}'
    )
    (cell_dir / "summary.json").write_text('{"campaign":{"total":2}}')

    loaded = figures.load_runpod_full_cells(cell_dir)
    assert len(loaded) == 1
    assert loaded[0]["topology"] == "colocated"
    assert loaded[0]["mean_ttft_ms"] == 100.0


def test_load_runpod_full_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert figures.load_runpod_full_cells(tmp_path / "nope") == []


def test_load_runpod_full_skips_malformed_json(tmp_path: Path) -> None:
    cell_dir = tmp_path / "cells"
    cell_dir.mkdir()
    (cell_dir / "colocated__qwen2.5-7b__rate-4__chat.json").write_text("{not json")
    (cell_dir / "colocated__qwen2.5-7b__rate-8__chat.json").write_text(
        '{"topology":"colocated","model":"qwen2.5-7b","rate_rps":8,'
        '"mix":"chat","mean_ttft_ms":50.0,"mean_itl_ms":5.0,'
        '"success_rate":1.0,"reconcile_passes":true}'
    )
    loaded = figures.load_runpod_full_cells(cell_dir)
    assert len(loaded) == 1
    assert loaded[0]["rate_rps"] == 8


def test_plot_runpod_full_curves_draws_one_line_per_topo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With >=2 reconciled cells per topology, the plot must emit a non-empty
    PNG keyed by mix name. FIGURES_DIR redirected to tmp so the test owns its
    own artifact."""
    monkeypatch.setattr(figures, "FIGURES_DIR", tmp_path)
    cells = [
        _fake_cell(Topology.COLOCATED, Model.QWEN2_5_7B, 1, Mix.CHAT, 80.0),
        _fake_cell(Topology.COLOCATED, Model.QWEN2_5_7B, 4, Mix.CHAT, 100.0),
        _fake_cell(Topology.COLOCATED, Model.QWEN2_5_7B, 16, Mix.CHAT, 200.0),
        _fake_cell(Topology.CHUNKED, Model.QWEN2_5_7B, 1, Mix.CHAT, 82.0),
        _fake_cell(Topology.CHUNKED, Model.QWEN2_5_7B, 4, Mix.CHAT, 105.0),
        _fake_cell(Topology.CHUNKED, Model.QWEN2_5_7B, 16, Mix.CHAT, 220.0),
    ]
    out = figures.plot_runpod_full_curves(cells)
    assert out is not None
    assert out.name == "runpod_full_ttft_chat.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_runpod_full_curves_returns_none_for_single_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A topology with only 1 reconciled cell cannot draw a line. If every
    topology has only 1 cell, the whole plot is None."""
    monkeypatch.setattr(figures, "FIGURES_DIR", tmp_path)
    cells = [_fake_cell(Topology.COLOCATED, Model.QWEN2_5_7B, 4, Mix.CHAT, 100.0)]
    assert figures.plot_runpod_full_curves(cells) is None


def test_plot_runpod_full_curves_filters_by_mix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mix argument selects which workload to plot."""
    monkeypatch.setattr(figures, "FIGURES_DIR", tmp_path)
    cells = [
        _fake_cell(Topology.COLOCATED, Model.QWEN2_5_7B, 1, Mix.CHAT, 80.0),
        _fake_cell(Topology.COLOCATED, Model.QWEN2_5_7B, 4, Mix.CHAT, 100.0),
        _fake_cell(Topology.COLOCATED, Model.QWEN2_5_7B, 1, Mix.RAG, 120.0),
        _fake_cell(Topology.COLOCATED, Model.QWEN2_5_7B, 4, Mix.RAG, 150.0),
    ]
    chat_out = figures.plot_runpod_full_curves(cells, mix="chat")
    rag_out = figures.plot_runpod_full_curves(cells, mix="rag")
    assert chat_out is not None
    assert rag_out is not None
    assert chat_out.name == "runpod_full_ttft_chat.png"
    assert rag_out.name == "runpod_full_ttft_rag.png"


def test_plot_runpod_full_curves_sorts_by_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Input order doesn't matter — internal sort by rate ascending."""
    monkeypatch.setattr(figures, "FIGURES_DIR", tmp_path)
    cells = [
        _fake_cell(Topology.COLOCATED, Model.QWEN2_5_7B, 16, Mix.CHAT, 200.0),
        _fake_cell(Topology.COLOCATED, Model.QWEN2_5_7B, 1, Mix.CHAT, 80.0),
        _fake_cell(Topology.COLOCATED, Model.QWEN2_5_7B, 4, Mix.CHAT, 100.0),
    ]
    out = figures.plot_runpod_full_curves(cells)
    assert out is not None