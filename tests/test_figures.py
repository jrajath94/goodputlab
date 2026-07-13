"""Tests for bench/figures.py — Run 1 figure generation."""

from __future__ import annotations

from pathlib import Path

import pytest

import bench.figures as figures


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
    # Hand-derived: 1 GPU at $1.99/hr, 120 tok/s → $/1M = 1.99/3600*1e6/120 ≈ 4.608
    expected = figures.H100_SXM_SPOT_USD_PER_HR * 1e6 / (3600 * figures.TOKENS_PER_SEC_PER_H100)
    assert figures.cost_per_million_tokens(1) == pytest.approx(expected, rel=1e-9)


def test_replica_counts_match_topology() -> None:
    assert figures.REPLICAS["colocated"] == 1
    assert figures.REPLICAS["chunked"] == 1
    assert figures.REPLICAS["disagg"] == 2
    assert figures.REPLICAS["disagg_tier"] == 2


def test_main_generates_all_artifacts(tmp_path: Path) -> None:
    """End-to-end: run main() against the canonical figures dir, then assert
    the four artifacts exist and are non-empty.
    """
    figures.main()
    expected = [
        figures.FIGURES_DIR / "ttft_comparison.png",
        figures.FIGURES_DIR / "itl_comparison.png",
        figures.FIGURES_DIR / "cost_per_million_tokens.csv",
        figures.FIGURES_DIR / "cost_per_million_tokens.md",
    ]
    for p in expected:
        assert p.exists(), f"missing artifact: {p}"
        assert p.stat().st_size > 0, f"empty artifact: {p}"


def test_cost_md_documents_assumptions() -> None:
    """The cost table is honest: assumptions are visible in the markdown."""
    figures.main()  # ensure file exists
    text = (figures.FIGURES_DIR / "cost_per_million_tokens.md").read_text()
    assert "H100 SXM spot" in text
    assert "$1.99" in text
    assert "120" in text  # tok/s per GPU
    assert "256" in text  # output tokens assumption
