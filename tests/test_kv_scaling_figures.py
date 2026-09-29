"""Tests for bench/kv_scaling/figures.py — KV-cache scaling study plots.

Builds a SYNTHETIC campaign fixture in tmp_path (theory curve + seeded
noise, two batch sizes, six sequence lengths) and runs all three plot
functions plus the markdown summary generator against it.

Conventions mirror tests/test_figures.py: FIGURES output dir is redirected
to tmp so tests own their artifacts; PNGs must be real renders (>10KB),
titles must carry the evidence context, and no DATA_URI/placeholder
strings may appear anywhere in the outputs (repo incident 2026-09-28).
Everything runs on a CPU-only box (matplotlib Agg).
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import pytest

import bench.kv_scaling.figures as kvf

MODEL_ID = "synth-test/Qwen3-0.6B"
MODEL_REVISION = "qwen3-main-2026-09-28"
GPU = "H100 SXM"
VLLM = "0.9.1"
METHOD = "gpu-alloc-delta"
THEORY_BPT = 27648.0  # synthetic campaign theory value (from data, not a constant)
BATCHES = [1, 4]
SEQ_LENS = [512, 1024, 2048, 4096, 8192, 16384]
MIN_PNG_BYTES = 10 * 1024


def _write_synthetic_campaign(tmp_path: Path) -> tuple[Path, Path]:
    """Theory curve + seeded noise. Returns (cells.csv, summary.json)."""
    rng = random.Random(7)
    rows: list[dict[str, str]] = []
    for b in BATCHES:
        for s in SEQ_LENS:
            total_tokens = b * s
            kv_bytes = THEORY_BPT * total_tokens
            baseline = 5.0e8
            noise = 1.0 + rng.uniform(-0.015, 0.015)
            p50 = baseline + kv_bytes * noise
            p95 = p50 * 1.02
            measured_bpt = (p50 - baseline) / total_tokens
            rows.append(
                {
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "gpu": GPU,
                    "vllm_version": VLLM,
                    "seq_len": str(s),
                    "batch_size": str(b),
                    "n_samples": "30",
                    "p50_alloc_bytes": f"{p50:.1f}",
                    "p95_alloc_bytes": f"{p95:.1f}",
                    "baseline_bytes": f"{baseline:.1f}",
                    "total_tokens": str(total_tokens),
                    "theory_bytes_per_token": f"{THEORY_BPT:.1f}",
                    "measured_bytes_per_token": f"{measured_bpt:.3f}",
                    "method": METHOD,
                }
            )
    # One defensive row: per-cell incremental measurement absent, as the
    # real campaign may leave measured_bytes_per_token empty.
    rows.append(
        {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "gpu": GPU,
            "vllm_version": VLLM,
            "seq_len": "32768",
            "batch_size": "4",
            "n_samples": "30",
            "p50_alloc_bytes": "5200000000",
            "p95_alloc_bytes": "5300000000",
            "baseline_bytes": "500000000",
            "total_tokens": "131072",
            "theory_bytes_per_token": f"{THEORY_BPT:.1f}",
            "measured_bytes_per_token": "",
            "method": METHOD,
        }
    )
    cells_csv = tmp_path / "cells.csv"
    with cells_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_json = tmp_path / "summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "campaign_id": "synth-kv-001",
                "measured_date": "2026-09-28",
                "model_revision": MODEL_REVISION,
                "theory_bytes_per_token": THEORY_BPT,
                "slope_bytes_per_token": THEORY_BPT * 0.997,
                "intercept_bytes": 0.0,
                "r2": 0.998,
                "knee_total_tokens_by_batch": {"1": 200000, "4": 120000},
                "notes": "synthetic fixture for figure tests",
            }
        )
    )
    return cells_csv, summary_json


def _write_sibling_cells_json(tmp_path: Path) -> Path:
    """cells.json in the sibling implementer's real format (schema 1.0),
    with a model_revision and a passing theory_check."""
    cells: list[dict[str, Any]] = []
    for b in BATCHES:
        for s in SEQ_LENS:
            total_tokens = b * s
            kv = THEORY_BPT * total_tokens
            cells.append(
                {
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "gpu": GPU,
                    "vllm_version": VLLM,
                    "seq_len": s,
                    "batch_size": b,
                    "n_samples": 30,
                    "mem_allocated_bytes_samples": [int(5.0e8 + kv)] * 30,
                    "total_tokens": total_tokens,
                    "baseline_bytes": int(5.0e8),
                    "p50_bytes": 5.0e8 + kv,
                    "p95_bytes": (5.0e8 + kv) * 1.02,
                    "method": METHOD,
                }
            )
    payload = {
        "schema_version": "1.0",
        "campaign": {
            "campaign_id": "sibling-kv-002",
            "model_revision": MODEL_REVISION,
            "cells": cells,
            "theory_bytes_per_token": THEORY_BPT,
            "regression": {
                "slope_bpt": THEORY_BPT * 0.997,
                "intercept_bytes": 0.0,
                "r2": 0.998,
                "residual_se": 1000.0,
            },
            "knee": {"total_tokens": 120000, "kv_share": 0.52, "batch_size": 4},
            "theory_check": {
                "measured": THEORY_BPT * 0.997,
                "theory": THEORY_BPT,
                "within_tol": True,
            },
        },
    }
    out = tmp_path / "cells.json"
    out.write_text(json.dumps(payload))
    return out


@pytest.fixture()
def campaign(tmp_path: Path) -> kvf.KVCampaign:
    cells_csv, summary_json = _write_synthetic_campaign(tmp_path)
    return kvf.load_campaign(cells_csv, summary_json)


# ---------- Loader ----------


def test_load_campaign_parses_all_cells(tmp_path: Path) -> None:
    cells_csv, summary_json = _write_synthetic_campaign(tmp_path)
    camp = kvf.load_campaign(cells_csv, summary_json)
    assert len(camp.cells) == 13
    for cell in camp.cells:
        assert cell.model_id == MODEL_ID
        assert cell.model_revision == MODEL_REVISION
        assert cell.theory_bytes_per_token == pytest.approx(THEORY_BPT)


def test_load_campaign_empty_measured_becomes_none(tmp_path: Path) -> None:
    """measured_bytes_per_token may be empty per cell; loader must not crash."""
    cells_csv, summary_json = _write_synthetic_campaign(tmp_path)
    camp = kvf.load_campaign(cells_csv, summary_json)
    empties = [c for c in camp.cells if c.measured_bytes_per_token is None]
    assert len(empties) == 1
    assert empties[0].seq_len == 32768


def test_load_campaign_summary_fields(tmp_path: Path) -> None:
    cells_csv, summary_json = _write_synthetic_campaign(tmp_path)
    camp = kvf.load_campaign(cells_csv, summary_json)
    assert camp.summary.campaign_id == "synth-kv-001"
    assert camp.summary.r2 == pytest.approx(0.998)
    assert camp.summary.knee_total_tokens_by_batch == {"1": 200000, "4": 120000}


def test_load_campaign_from_json_sibling_format(tmp_path: Path) -> None:
    """The sibling's cells.json (schema 1.0) loads: theory, revision,
    regression, single knee, and the theory check all land correctly."""
    cells_json = _write_sibling_cells_json(tmp_path)
    camp, check = kvf.load_campaign_from_json(cells_json)
    assert len(camp.cells) == 12
    for cell in camp.cells:
        assert cell.model_revision == MODEL_REVISION
        assert cell.theory_bytes_per_token == pytest.approx(THEORY_BPT)
        assert cell.p50_alloc_bytes > 0
    assert camp.summary.campaign_id == "sibling-kv-002"
    assert camp.summary.slope_bytes_per_token == pytest.approx(THEORY_BPT * 0.997)
    assert camp.summary.knee_total_tokens_by_batch == {"4": 120000}
    assert check is not None
    assert check.within_tol is True


def test_title_context_carries_evidence(campaign: kvf.KVCampaign) -> None:
    """Every plot title must state model, GPU, vLLM version, n, method
    (README evidence policy)."""
    ctx = kvf.title_context(campaign.cells, campaign.summary)
    assert "Qwen3-0.6B" in ctx
    assert "H100 SXM" in ctx
    assert "0.9.1" in ctx
    assert "n=" in ctx
    assert "gpu-alloc-delta" in ctx


def test_title_context_includes_revision_when_present(
    campaign: kvf.KVCampaign,
) -> None:
    """Raj's update: revision renders in the title when the campaign has one."""
    ctx = kvf.title_context(campaign.cells, campaign.summary)
    assert MODEL_REVISION in ctx
    assert "(rev " in ctx


def test_title_context_without_revision_has_no_cruft(tmp_path: Path) -> None:
    """Revision is optional: titles stay clean when the campaign omits it."""
    cells_csv, summary_json = _write_synthetic_campaign(tmp_path)
    camp = kvf.load_campaign(cells_csv, summary_json)
    for cell in camp.cells:
        cell.model_revision = ""
    ctx = kvf.title_context(camp.cells, camp.summary)
    assert "rev" not in ctx
    assert "None" not in ctx
    assert MODEL_ID in ctx


def test_no_hardcoded_model_names_in_module() -> None:
    """Model identity must come from campaign data, never from a constant."""
    src = Path(kvf.__file__).read_text()
    for token in ("Qwen2.5", "Qwen3", "Llama", "24576", "32768"):
        assert token not in src, f"hardcoded model token in module: {token}"


def test_campaign_theory_comes_from_data(campaign: kvf.KVCampaign) -> None:
    """The theory line value is the campaign's field, not a module constant."""
    assert kvf._campaign_theory(campaign.cells) == pytest.approx(THEORY_BPT)
    # A campaign with a different theory moves the line with it.
    for cell in campaign.cells:
        cell.theory_bytes_per_token = 10000.0
    assert kvf._campaign_theory(campaign.cells) == pytest.approx(10000.0)


# ---------- Theory-check gate ----------


def test_theory_check_caption_format() -> None:
    check = kvf.TheoryCheck(measured=10000.0, theory=10200.0, within_tol=True)
    caption = kvf.theory_check_caption(check)
    assert caption == "theory check: PASS (measured 10,000 vs predicted 10,200, 2.0% diff)"


def test_theory_check_pass_renders_parity_plot(campaign: kvf.KVCampaign, tmp_path: Path) -> None:
    """A passed check plots fine (dict form, as the campaign JSON carries it)."""
    out = kvf.plot_theory_vs_measured(
        [
            kvf.ParityPoint(
                THEORY_BPT,
                campaign.summary.slope_bytes_per_token or 0.0,
                campaign.summary.campaign_id,
            )
        ],
        tmp_path,
        context=kvf.title_context(campaign.cells, campaign.summary),
        theory_check={"measured": THEORY_BPT * 0.997, "theory": THEORY_BPT, "within_tol": True},
    )
    assert out.exists()
    assert out.stat().st_size > MIN_PNG_BYTES


def test_theory_check_fail_raises_and_plots_nothing(
    campaign: kvf.KVCampaign, tmp_path: Path
) -> None:
    """A failed check must raise TheoryCheckFailed and write no PNG."""
    with pytest.raises(kvf.TheoryCheckFailed):
        kvf.plot_theory_vs_measured(
            [
                kvf.ParityPoint(
                    THEORY_BPT,
                    campaign.summary.slope_bytes_per_token or 0.0,
                    campaign.summary.campaign_id,
                )
            ],
            tmp_path,
            context=kvf.title_context(campaign.cells, campaign.summary),
            theory_check=kvf.TheoryCheck(measured=15000.0, theory=THEORY_BPT, within_tol=False),
        )
    assert not (tmp_path / "theory_vs_measured.png").exists()


def test_theory_check_none_still_plots(campaign: kvf.KVCampaign, tmp_path: Path) -> None:
    """No check supplied (older campaigns): plot as before, no gate."""
    out = kvf.plot_theory_vs_measured(
        [kvf.ParityPoint(THEORY_BPT, THEORY_BPT * 0.997, "old-campaign")],
        tmp_path,
        context="old campaign",
    )
    assert out.exists()
    assert out.stat().st_size > MIN_PNG_BYTES


# ---------- Plots ----------


def _run_all(campaign: kvf.KVCampaign, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        kvf.plot_bytes_per_token_vs_seqlen(campaign.cells, campaign.summary, out_dir),
        kvf.plot_kv_fraction_of_gpu_ram(campaign.cells, campaign.summary, out_dir),
        kvf.plot_theory_vs_measured(
            [
                kvf.ParityPoint(
                    THEORY_BPT,
                    campaign.summary.slope_bytes_per_token or 0.0,
                    campaign.summary.campaign_id,
                )
            ],
            out_dir,
            context=kvf.title_context(campaign.cells, campaign.summary),
            theory_check={"measured": THEORY_BPT * 0.997, "theory": THEORY_BPT, "within_tol": True},
        ),
        kvf.write_kv_scaling_summary(campaign, out_dir),
    ]
    return paths


def test_all_artifacts_are_real_pngs(campaign: kvf.KVCampaign, tmp_path: Path) -> None:
    """No DATA_URI placeholders: every png must exist and be a real render."""
    paths = _run_all(campaign, tmp_path)
    pngs = [p for p in paths if p.suffix == ".png"]
    assert len(pngs) == 3
    for p in pngs:
        assert p.exists(), f"missing plot: {p}"
        assert p.stat().st_size > MIN_PNG_BYTES, (
            f"plot too small to be a real render ({p.stat().st_size}B): {p}"
        )


def test_no_placeholder_strings_in_outputs(campaign: kvf.KVCampaign, tmp_path: Path) -> None:
    """Repo incident 2026-09-28: literal DATA_URI/placeholder img srcs shipped.
    Scan generated files AND the module source itself."""
    paths = _run_all(campaign, tmp_path)
    for p in paths:
        blob = p.read_bytes().lower()
        assert b"data_uri" not in blob, f"DATA_URI marker in {p.name}"
        assert b"placeholder" not in blob, f"placeholder marker in {p.name}"
    src = Path(kvf.__file__).read_text().lower()
    assert "data_uri" not in src
    assert "placeholder" not in src


def test_summary_md_contains_honest_claim_and_limits(
    campaign: kvf.KVCampaign, tmp_path: Path
) -> None:
    md_path = kvf.write_kv_scaling_summary(campaign, tmp_path)
    text = md_path.read_text()
    assert "describe this campaign only" in text
    assert "not generalized" in text
    assert "## Limits" in text
    assert "## Method" in text
    assert "synth-kv-001" in text
    assert "27,648" in text  # theory bytes/token appears in the numbers
    assert MODEL_REVISION in text


def test_summary_md_references_real_plot_files(campaign: kvf.KVCampaign, tmp_path: Path) -> None:
    """The summary may only name plot files that actually exist on disk."""
    paths = _run_all(campaign, tmp_path)
    md_path = next(p for p in paths if p.suffix == ".md")
    text = md_path.read_text()
    for name in (
        "kv_bytes_per_token_vs_seqlen.png",
        "kv_fraction_of_gpu_ram.png",
        "theory_vs_measured.png",
    ):
        assert name in text, f"summary does not reference {name}"
        assert (tmp_path / name).exists()


def test_theory_vs_measured_single_campaign_works(campaign: kvf.KVCampaign, tmp_path: Path) -> None:
    """One campaign = one point + bands; must not crash on axis scaling."""
    out = kvf.plot_theory_vs_measured(
        [
            kvf.ParityPoint(
                THEORY_BPT,
                campaign.summary.slope_bytes_per_token or 0.0,
                campaign.summary.campaign_id,
            )
        ],
        tmp_path,
        context=kvf.title_context(campaign.cells, campaign.summary),
    )
    assert out.exists()
    assert out.stat().st_size > MIN_PNG_BYTES


def test_theory_vs_measured_many_campaigns(campaign: kvf.KVCampaign, tmp_path: Path) -> None:
    pts = [
        kvf.ParityPoint(27648.0, 27565.0, "synth-kv-001"),
        kvf.ParityPoint(32768.0, 33000.0, "llama-1b-run"),
        kvf.ParityPoint(27648.0, 24000.0, "qwen-outlier-run"),  # outside +10%
    ]
    out = kvf.plot_theory_vs_measured(pts, tmp_path, context="3 campaigns")
    assert out.exists()
    assert out.stat().st_size > MIN_PNG_BYTES


def test_main_skips_gracefully_when_campaign_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """main() must not raise when the campaign has not been run yet."""
    monkeypatch.setattr(kvf, "KV_RESULTS_ROOT", tmp_path / "empty")
    monkeypatch.setattr(kvf, "LEGACY_RESULTS_DIR", tmp_path / "legacy-empty")
    monkeypatch.setattr(kvf, "KV_FIGURES_DIR", tmp_path / "figures")
    kvf.main()
    assert "no campaign yet" in capsys.readouterr().out
