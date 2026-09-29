"""Generate KV-cache scaling figures from a campaign's artifacts.

Primary input is the sibling implementer's ``cells.json`` (written by
``bench/kv_scaling/results.py`` to
``bench/results/kv_scaling/<campaign_id>/cells.json``)::

    {"schema_version": "1.0",
     "campaign": {"campaign_id": ..., "cells": [...],
                  "theory_bytes_per_token": ...,
                  "regression": {"slope_bpt": ..., "intercept_bytes": ...,
                                 "r2": ..., "residual_se": ...},
                  "knee": {"total_tokens": ..., "kv_share": ...,
                           "batch_size": ...},
                  "model_revision": ... (optional),
                  "theory_check": {"measured": ..., "theory": ...,
                                   "within_tol": ...} (optional)}}

Cells in ``cells.json`` carry ``p50_bytes`` / ``p95_bytes`` (computed by
the schema). The legacy pair ``cells.csv`` + ``summary.json`` under
``bench/kv_scaling/results/`` is still accepted; every column is read
defensively with ``.get()`` defaults.

Model identity (``model_id`` plus optional ``model_revision``) is read
from the campaign data and rendered in titles and footers. Nothing in
this module hardcodes a model name. The theory line always comes from
the campaign's ``theory_bytes_per_token`` field, never a constant.

Gate before publishing: the parity plot only renders when the
measured-vs-theory cross-check passed. Pass ``theory_check`` (a
``TheoryCheck`` or a ``{"measured", "theory", "within_tol"}`` dict);
a failed check raises ``TheoryCheckFailed`` instead of plotting, and a
passed check renders a small caption on the parity plot.

Run::

    python3 -m bench.kv_scaling.figures

Outputs (under ``bench/kv_scaling/figures/``):
- ``kv_bytes_per_token_vs_seqlen.png`` — measured bytes/token vs seq_len
  (log x), one curve per batch size, dashed theory line, (p95-p50)
  error bars
- ``kv_fraction_of_gpu_ram.png`` — KV share of GPU RAM vs total context
  tokens (log x), 50% line, vertical knee markers
- ``theory_vs_measured.png`` — parity plot of measured regression slope
  vs theory bytes/token per campaign, with +/-10% bands and the
  theory-check caption
- ``kv_scaling_summary.md`` — per-cell numbers, regression results,
  knee, method and limits

Theory reference: KV bytes/token =
2 x layers x kv_heads x head_dim x bytes_per_element.

The figure generation is a thin wrapper around matplotlib; no GPU, no
network. Re-run after every KV scaling campaign to refresh plots.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive; safe for CI

import matplotlib.pyplot as plt  # noqa: E402

# ---------- Constants ----------

# Sibling implementer output root: bench/results/kv_scaling/<campaign_id>/
KV_RESULTS_ROOT = Path(__file__).parent.parent / "results" / "kv_scaling"
# Legacy campaign pair (original brief contract).
LEGACY_RESULTS_DIR = Path(__file__).parent / "results"
KV_FIGURES_DIR = Path(__file__).parent / "figures"

# Stable curve colours, mirroring bench/figures.py's palette.
BLUE = "#4C72B0"
GREEN = "#55A868"
ORANGE = "#DD8452"
RED = "#C44E52"
PURPLE = "#8172B2"
BATCH_COLOURS = [BLUE, GREEN, ORANGE, RED, PURPLE]

GIB = 1024**3

# GPU name fragment -> total HBM bytes. Used by the GPU-RAM fraction plot;
# unknown GPUs fall back to 80 GiB and the summary says so explicitly.
GPU_RAM_BYTES = {
    "H100 SXM": 80 * GIB,
    "H100 NVL": 94 * GIB,
    "A100 80GB": 80 * GIB,
    "A100 40GB": 40 * GIB,
    "B200": 192 * GIB,
    "L40S": 48 * GIB,
}
DEFAULT_GPU_RAM_BYTES = 80 * GIB


# ---------- Data model ----------


@dataclass
class KVCell:
    model_id: str
    model_revision: str
    gpu: str
    vllm_version: str
    seq_len: int
    batch_size: int
    n_samples: int
    p50_alloc_bytes: float
    p95_alloc_bytes: float
    baseline_bytes: float
    total_tokens: int
    theory_bytes_per_token: float
    measured_bytes_per_token: float | None
    method: str


@dataclass
class KVSummary:
    campaign_id: str
    measured_date: str
    slope_bytes_per_token: float | None
    intercept_bytes: float | None
    r2: float | None
    knee_total_tokens_by_batch: dict[str, int] = field(default_factory=dict)
    notes: str = ""


@dataclass
class KVCampaign:
    cells: list[KVCell]
    summary: KVSummary


@dataclass
class ParityPoint:
    theory_bpt: float
    measured_slope_bpt: float
    campaign_id: str


@dataclass
class TheoryCheck:
    """Measured-vs-theory cross-check result; the publish gate."""

    measured: float
    theory: float
    within_tol: bool


class TheoryCheckFailed(ValueError):
    """Raised when the theory cross-check failed: refuse to plot."""


# ---------- Loading (defensive: every column has a default) ----------


def _f(raw: str, default: float = 0.0) -> float:
    try:
        return float(raw) if raw.strip() else default
    except ValueError:
        return default


def _i(raw: str, default: int = 0) -> int:
    try:
        return int(float(raw)) if raw.strip() else default
    except ValueError:
        return default


def _get(row: dict[str, Any], key: str) -> str:
    return str(row.get(key, "") or "")


def _opt_float(data: dict[str, Any], key: str) -> float | None:
    val = data.get(key)
    if val is None or (isinstance(val, str) and not val.strip()):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _campaign_theory(cells: list[KVCell]) -> float:
    """Theory bytes/token for the campaign: mean of per-cell values.

    The only theory source the plots use. Per-cell values are filled
    from the campaign-level ``theory_bytes_per_token`` field at load
    time, so this is the campaign's number, never a module constant.
    """
    vals = [c.theory_bytes_per_token for c in cells if c.theory_bytes_per_token > 0]
    return sum(vals) / len(vals) if vals else 0.0


def _cell_from_row(
    row: dict[str, Any],
    fallback_theory: float,
    fallback_revision: str,
) -> KVCell:
    measured_raw = _get(row, "measured_bytes_per_token")
    theory_raw = _get(row, "theory_bytes_per_token")
    revision = _get(row, "model_revision") or fallback_revision
    # Sibling cells.csv names the aggregates p50_bytes/p95_bytes; the
    # legacy contract used p50_alloc_bytes/p95_alloc_bytes. Accept both.
    p50_raw = _get(row, "p50_alloc_bytes") or _get(row, "p50_bytes")
    p95_raw = _get(row, "p95_alloc_bytes") or _get(row, "p95_bytes")
    return KVCell(
        model_id=_get(row, "model_id"),
        model_revision=revision,
        gpu=_get(row, "gpu"),
        vllm_version=_get(row, "vllm_version"),
        seq_len=_i(_get(row, "seq_len")),
        batch_size=_i(_get(row, "batch_size")),
        n_samples=_i(_get(row, "n_samples")),
        p50_alloc_bytes=_f(p50_raw),
        p95_alloc_bytes=_f(p95_raw),
        baseline_bytes=_f(_get(row, "baseline_bytes")),
        total_tokens=_i(_get(row, "total_tokens")),
        theory_bytes_per_token=_f(theory_raw, fallback_theory),
        measured_bytes_per_token=_f(measured_raw) if measured_raw.strip() else None,
        method=_get(row, "method"),
    )


def _summary_from_legacy(data: dict[str, Any]) -> KVSummary:
    knee: dict[str, int] = {}
    for k, v in (data.get("knee_total_tokens_by_batch") or {}).items():
        try:
            knee[str(k)] = int(v)
        except (ValueError, TypeError):
            continue
    return KVSummary(
        campaign_id=str(data.get("campaign_id", "")),
        measured_date=str(data.get("measured_date", "")),
        slope_bytes_per_token=_opt_float(data, "slope_bytes_per_token"),
        intercept_bytes=_opt_float(data, "intercept_bytes"),
        r2=_opt_float(data, "r2"),
        knee_total_tokens_by_batch=knee,
        notes=str(data.get("notes", "")),
    )


def load_campaign(cells_csv: Path, summary_json: Path) -> KVCampaign:
    """Load the legacy cells.csv + summary.json pair.

    Missing or empty columns fall back to neutral defaults; an empty
    ``measured_bytes_per_token`` becomes None (falls back to theory on
    plot 1). Per-cell theory defaults to the summary-level
    ``theory_bytes_per_token`` when the column is absent.
    """
    payload: dict[str, Any] = json.loads(summary_json.read_text())
    summary = _summary_from_legacy(payload)
    fallback_theory = _f(str(payload.get("theory_bytes_per_token", "") or ""))
    fallback_revision = str(payload.get("model_revision", "") or "")
    cells: list[KVCell] = []
    with cells_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            cells.append(_cell_from_row(dict(row), fallback_theory, fallback_revision))
    return KVCampaign(cells=cells, summary=summary)


def load_campaign_from_json(cells_json: Path) -> tuple[KVCampaign, TheoryCheck | None]:
    """Load the sibling implementer's cells.json payload.

    Returns (campaign, theory_check). ``theory_check`` is None when the
    campaign JSON does not carry one. Per-cell theory is filled from the
    campaign-level ``theory_bytes_per_token``; ``model_revision`` is read
    per cell with a campaign-level fallback.
    """
    payload: dict[str, Any] = json.loads(cells_json.read_text())
    data: dict[str, Any] = payload.get("campaign", payload)
    theory = _f(str(data.get("theory_bytes_per_token", "") or ""))
    revision = str(data.get("model_revision", "") or "")
    reg: dict[str, Any] = data.get("regression", {}) or {}
    knee_data: dict[str, Any] = data.get("knee", {}) or {}
    knee: dict[str, int] = {}
    if knee_data.get("total_tokens") is not None:
        batch = knee_data.get("batch_size")
        knee[str(batch) if batch is not None else "pooled"] = int(knee_data["total_tokens"])
    summary = KVSummary(
        campaign_id=str(data.get("campaign_id", "")),
        measured_date=str(data.get("measured_date", "")),
        slope_bytes_per_token=_opt_float(reg, "slope_bpt"),
        intercept_bytes=_opt_float(reg, "intercept_bytes"),
        r2=_opt_float(reg, "r2"),
        knee_total_tokens_by_batch=knee,
        notes=str(data.get("notes", "")),
    )
    cells = [_cell_from_row(dict(row), theory, revision) for row in (data.get("cells", []) or [])]
    check_raw = data.get("theory_check")
    theory_check = _coerce_theory_check(check_raw) if check_raw is not None else None
    return KVCampaign(cells=cells, summary=summary), theory_check


def _coerce_theory_check(raw: TheoryCheck | dict[str, Any] | None) -> TheoryCheck | None:
    """Accept a TheoryCheck, a {"measured","theory","within_tol"} dict, or None."""
    if raw is None:
        return None
    if isinstance(raw, TheoryCheck):
        return raw
    return TheoryCheck(
        measured=float(raw.get("measured", 0.0)),
        theory=float(raw.get("theory", 0.0)),
        within_tol=bool(raw.get("within_tol", False)),
    )


def theory_check_caption(check: TheoryCheck) -> str:
    """One-line caption for a passed theory cross-check."""
    diff_pct = abs(check.measured - check.theory) / check.theory * 100.0 if check.theory else 0.0
    return (
        f"theory check: PASS (measured {check.measured:,.0f} vs "
        f"predicted {check.theory:,.0f}, {diff_pct:.1f}% diff)"
    )


def title_context(cells: list[KVCell], summary: KVSummary) -> str:
    """Evidence string for plot titles: model (+revision), GPU, vLLM, n, method."""
    first = cells[0] if cells else None
    model = first.model_id if first else "unknown-model"
    revision = first.model_revision if first else ""
    model_label = f"{model} (rev {revision})" if revision else model
    gpu = first.gpu if first else "unknown-gpu"
    vllm = first.vllm_version if first else "unknown-vllm"
    n = sum(c.n_samples for c in cells)
    methods = sorted({c.method for c in cells if c.method})
    method = "/".join(methods) if methods else "unknown-method"
    return f"{model_label} on {gpu}, vLLM {vllm}, n={n}, method={method} ({summary.campaign_id})"


def gpu_ram_bytes(gpu: str) -> tuple[int, bool]:
    """Total HBM bytes for a GPU name. Returns (bytes, assumed).

    ``assumed`` is True when the GPU is not in the lookup table, so the
    summary can say so honestly instead of presenting a guess as fact.
    """
    for name, size in GPU_RAM_BYTES.items():
        if name.lower() in gpu.lower():
            return size, False
    return DEFAULT_GPU_RAM_BYTES, True


# ---------- Figures ----------


def _batch_colour(batch_size: int, order: int) -> str:
    if batch_size in (1, 4, 8, 16, 32):
        return BATCH_COLOURS[[1, 4, 8, 16, 32].index(batch_size)]
    return BATCH_COLOURS[order % len(BATCH_COLOURS)]


def _add_footer(fig: Any, text: str) -> None:
    fig.text(0.5, 0.01, text, ha="center", fontsize=7, color="#666666")


def plot_bytes_per_token_vs_seqlen(
    cells: list[KVCell],
    summary: KVSummary,
    out_dir: Path = KV_FIGURES_DIR,
) -> Path:
    """Measured bytes/token vs seq_len (log x), one curve per batch size."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = title_context(cells, summary)
    by_batch: dict[int, list[KVCell]] = {}
    for c in cells:
        by_batch.setdefault(c.batch_size, []).append(c)
    for pts in by_batch.values():
        pts.sort(key=lambda c: c.seq_len)

    theory = _campaign_theory(cells)

    fig, ax = plt.subplots(figsize=(7, 4))
    for order, batch in enumerate(sorted(by_batch)):
        pts = by_batch[batch]
        xs = [c.seq_len for c in pts]
        ys = [
            c.measured_bytes_per_token
            if c.measured_bytes_per_token is not None
            else c.theory_bytes_per_token
            for c in pts
        ]
        yerr = [
            max(c.p95_alloc_bytes - c.p50_alloc_bytes, 0.0) / max(c.total_tokens, 1) for c in pts
        ]
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            marker="o",
            capsize=3,
            label=f"batch={batch}",
            color=_batch_colour(batch, order),
        )
    if theory > 0:
        ax.axhline(theory, linestyle="--", color=RED, label=f"theory {theory:,.0f} B/tok")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("KV bytes / token")
    ax.set_title(f"KV bytes/token vs sequence length — {ctx}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _add_footer(fig, ctx)

    out = out_dir / "kv_bytes_per_token_vs_seqlen.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_kv_fraction_of_gpu_ram(
    cells: list[KVCell],
    summary: KVSummary,
    out_dir: Path = KV_FIGURES_DIR,
) -> Path:
    """KV share of total GPU RAM vs total context tokens (log x).

    KV bytes = theory_bytes_per_token x total_tokens. The 50% line and
    the knee markers make the memory ceiling visible.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = title_context(cells, summary)
    gpu = cells[0].gpu if cells else ""
    ram_bytes, ram_assumed = gpu_ram_bytes(gpu)
    ram_label = f"{ram_bytes / GIB:.0f} GiB" + (" (assumed)" if ram_assumed else "")

    by_batch: dict[int, list[KVCell]] = {}
    for c in cells:
        by_batch.setdefault(c.batch_size, []).append(c)
    for pts in by_batch.values():
        pts.sort(key=lambda c: c.total_tokens)

    fig, ax = plt.subplots(figsize=(7, 4))
    for order, batch in enumerate(sorted(by_batch)):
        pts = by_batch[batch]
        xs = [c.total_tokens for c in pts]
        ys = [100.0 * c.theory_bytes_per_token * c.total_tokens / ram_bytes for c in pts]
        ax.plot(xs, ys, marker="o", label=f"batch={batch}", color=_batch_colour(batch, order))
    ax.axhline(50.0, linestyle="--", color="black", label="50% of GPU RAM")
    for batch_key in sorted(summary.knee_total_tokens_by_batch):
        knee = summary.knee_total_tokens_by_batch[batch_key]
        ax.axvline(knee, linestyle=":", color=RED, alpha=0.7)
        ax.text(
            knee,
            52,
            f"knee b={batch_key}",
            rotation=90,
            fontsize=8,
            color=RED,
            va="bottom",
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Total context tokens (batch x seq_len)")
    ax.set_ylabel(f"KV share of GPU RAM ({ram_label}, %)")
    ax.set_title(f"KV share of GPU RAM vs context tokens — {ctx}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _add_footer(fig, ctx)

    out = out_dir / "kv_fraction_of_gpu_ram.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_theory_vs_measured(
    points: Sequence[ParityPoint],
    out_dir: Path,
    context: str,
    theory_check: TheoryCheck | dict[str, Any] | None = None,
) -> Path:
    """Parity plot: measured regression slope vs theory bytes/token.

    Works with a single campaign (one point + bands) and with many.
    Bands are +/-10% around parity.

    Publish gate: when ``theory_check`` is given and its ``within_tol``
    is False, raise ``TheoryCheckFailed`` instead of plotting. When it
    passed, render the check caption on the plot.
    """
    check = _coerce_theory_check(theory_check)
    if check is not None and not check.within_tol:
        diff_pct = (
            abs(check.measured - check.theory) / check.theory * 100.0 if check.theory else 0.0
        )
        raise TheoryCheckFailed(
            f"theory check FAILED: measured {check.measured:,.0f} vs "
            f"predicted {check.theory:,.0f} ({diff_pct:.1f}% diff) exceeds "
            "tolerance; refusing to plot"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    xs = [p.theory_bpt for p in points]
    ys = [p.measured_slope_bpt for p in points]
    if not xs:
        raise ValueError("plot_theory_vs_measured needs at least one point")

    lo = min(min(xs), min(ys))
    hi = max(max(xs), max(ys))
    if lo == hi:  # single point: give the axes room to breathe
        lo, hi = lo * 0.8, hi * 1.2
    else:
        pad = 0.1 * (hi - lo)
        lo, hi = lo - pad, hi + pad

    fig, ax = plt.subplots(figsize=(7, 4))
    grid = [lo, hi]
    ax.plot(grid, grid, linestyle="--", color="black", label="parity")
    ax.plot(grid, [0.9 * g for g in grid], linestyle=":", color=RED, label="+/-10% bands")
    ax.plot(grid, [1.1 * g for g in grid], linestyle=":", color=RED)
    ax.scatter(xs, ys, color=BLUE, s=40, zorder=3)
    if len(points) <= 8:
        for p in points:
            ax.annotate(
                p.campaign_id,
                (p.theory_bpt, p.measured_slope_bpt),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )

    ax.set_xlabel("Theory bytes/token")
    ax.set_ylabel("Measured slope (bytes/token)")
    ax.set_title(f"Theory vs measured KV bytes/token — {context}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    footer = context
    if check is not None:
        footer += "\n" + theory_check_caption(check)
    _add_footer(fig, footer)

    out = out_dir / "theory_vs_measured.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# ---------- Markdown summary ----------


def write_kv_scaling_summary(
    campaign: KVCampaign,
    out_dir: Path = KV_FIGURES_DIR,
) -> Path:
    """Per-cell numbers, regression results, knee, method, limits.

    Each fact appears once (no repeated exposition). The limits section
    carries the honest-claim sentence: this campaign only, no
    generalization.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = campaign.cells
    summary = campaign.summary
    ctx = title_context(cells, summary)
    gpu = cells[0].gpu if cells else "unknown"
    ram_bytes, ram_assumed = gpu_ram_bytes(gpu)

    lines = [f"# KV-cache scaling — {cells[0].model_id if cells else 'unknown'}", ""]
    if cells and cells[0].model_revision:
        lines += [f"Model revision: `{cells[0].model_revision}`", ""]
    lines += ["## Campaign", ""]
    lines += [
        "| Field | Value |",
        "|-------|-------|",
        f"| Campaign | {summary.campaign_id} |",
        f"| Measured | {summary.measured_date} |",
        f"| Context | {ctx} |",
        f"| Cells | {len(cells)} |",
        f"| GPU RAM | {ram_bytes / GIB:.0f} GiB"
        + (" (assumed, not in lookup table)" if ram_assumed else " (from lookup table)")
        + " |",
        "",
    ]

    lines += ["## Regression", ""]
    slope = (
        f"{summary.slope_bytes_per_token:,.1f}"
        if summary.slope_bytes_per_token is not None
        else "n/a"
    )
    intercept = f"{summary.intercept_bytes:,.1f}" if summary.intercept_bytes is not None else "n/a"
    r2 = f"{summary.r2:.4f}" if summary.r2 is not None else "n/a"
    lines += [
        f"Slope (measured bytes/token): {slope}",
        "",
        f"Intercept: {intercept} bytes; R^2 = {r2}.",
        "",
        "The parity plot (theory_vs_measured.png) draws +/-10% bands around "
        "y=x: points inside the bands agree with theory within measurement noise. "
        "The parity plot only renders when the measured-vs-theory cross-check "
        "passed; a failed check raises instead of plotting.",
        "",
    ]

    lines += ["## Knee (tokens where KV pressure dominates)", ""]
    if summary.knee_total_tokens_by_batch:
        lines += [
            "| Batch | Knee total tokens | KV at knee (GiB) | Share of GPU RAM |",
            "|-------|-------------------|------------------|------------------|",
        ]
        for batch_key in sorted(summary.knee_total_tokens_by_batch):
            knee = summary.knee_total_tokens_by_batch[batch_key]
            kv_gib = _campaign_theory(cells) * knee / GIB if cells else 0.0
            lines.append(
                f"| {batch_key} | {knee:,} | {kv_gib:.2f} | {100 * kv_gib * GIB / ram_bytes:.1f}% |"
            )
    else:
        lines.append("No knee markers recorded for this campaign.")
    lines.append("")

    lines += ["## Cells", ""]
    lines += [
        "| seq_len | batch | n | p50 alloc (GiB) | p95-p50 (MiB) | "
        "measured B/tok | theory B/tok | ratio |",
        "|---------|-------|---|---------------|----------------|"
        "----------------|--------------|-------|",
    ]
    for c in sorted(cells, key=lambda c: (c.batch_size, c.seq_len)):
        measured = (
            f"{c.measured_bytes_per_token:,.1f}"
            if c.measured_bytes_per_token is not None
            else "n/a"
        )
        ratio = (
            f"{c.measured_bytes_per_token / c.theory_bytes_per_token:.3f}"
            if c.measured_bytes_per_token and c.theory_bytes_per_token
            else "n/a"
        )
        lines.append(
            f"| {c.seq_len:,} | {c.batch_size} | {c.n_samples} | "
            f"{c.p50_alloc_bytes / GIB:.3f} | "
            f"{max(c.p95_alloc_bytes - c.p50_alloc_bytes, 0.0) / 1024**2:.2f} | "
            f"{measured} | {c.theory_bytes_per_token:,.1f} | {ratio} |"
        )
    lines.append("")

    lines += ["## Method", ""]
    lines += [
        "Each cell records GPU allocated bytes (p50 and p95 over n samples) "
        "for a fixed (model, seq_len, batch) point, minus a baseline "
        "allocation measured with the same engine and no KV cache. "
        "Per-cell incremental measured bytes/token = "
        "(p50_alloc - baseline) / total_tokens. The campaign regression "
        "fits allocated bytes against total context tokens across all "
        "cells; its slope is the headline measured bytes/token and is "
        "plotted against the theoretical 2 x layers x kv_heads x head_dim "
        "x bytes_per_element value in theory_vs_measured.png.",
        "",
    ]

    lines += ["## Limits", ""]
    lines += [
        "These numbers describe this campaign only. "
        "They are not generalized to other models, GPUs, or vLLM versions: "
        "KV layout, block size, and allocator behavior are all "
        "engine-version dependent. Cells whose measured_bytes_per_token is "
        "empty contribute to the regression through theory only. "
        "GPU RAM totals come from a lookup table and are marked assumed "
        "where the GPU was not recognized.",
        "",
    ]

    lines += ["## Plots", ""]
    for name in (
        "kv_bytes_per_token_vs_seqlen.png",
        "kv_fraction_of_gpu_ram.png",
        "theory_vs_measured.png",
    ):
        lines.append(f"- {name}")
    lines.append("")

    out = out_dir / "kv_scaling_summary.md"
    out.write_text("\n".join(lines))
    return out


# ---------- Entry point ----------


def _newest_campaign_json(root: Path) -> Path | None:
    """Newest cells.json under <root>/<campaign_id>/cells.json, if any."""
    if not root.is_dir():
        return None
    candidates = sorted(root.glob("*/cells.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def main() -> None:
    KV_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    campaign: KVCampaign | None = None
    theory_check: TheoryCheck | None = None

    cells_json = _newest_campaign_json(KV_RESULTS_ROOT)
    if cells_json is not None:
        campaign, theory_check = load_campaign_from_json(cells_json)
        print(f"loaded campaign from {cells_json}")
    else:
        legacy_csv = LEGACY_RESULTS_DIR / "cells.csv"
        legacy_summary = LEGACY_RESULTS_DIR / "summary.json"
        if legacy_csv.exists() and legacy_summary.exists():
            campaign = load_campaign(legacy_csv, legacy_summary)
            print(f"loaded legacy campaign from {legacy_csv}")

    if campaign is None:
        print(f"no campaign yet: expected {KV_RESULTS_ROOT}/<campaign_id>/cells.json")
        print("run the KV scaling campaign first, then re-run this module.")
        return

    ctx = title_context(campaign.cells, campaign.summary)
    slope = campaign.summary.slope_bytes_per_token or 0.0
    theory = _campaign_theory(campaign.cells)

    p1 = plot_bytes_per_token_vs_seqlen(campaign.cells, campaign.summary)
    p2 = plot_kv_fraction_of_gpu_ram(campaign.cells, campaign.summary)
    p3 = plot_theory_vs_measured(
        [ParityPoint(theory, slope, campaign.summary.campaign_id)],
        KV_FIGURES_DIR,
        context=ctx,
        theory_check=theory_check,
    )
    md = write_kv_scaling_summary(campaign)

    print("Generated:")
    for p in (p1, p2, p3, md):
        print(f"  {p}")


if __name__ == "__main__":
    main()


__all__ = [
    "BATCH_COLOURS",
    "DEFAULT_GPU_RAM_BYTES",
    "GPU_RAM_BYTES",
    "KVCampaign",
    "KVCell",
    "KVSummary",
    "KV_FIGURES_DIR",
    "KV_RESULTS_ROOT",
    "LEGACY_RESULTS_DIR",
    "ParityPoint",
    "TheoryCheck",
    "TheoryCheckFailed",
    "gpu_ram_bytes",
    "load_campaign",
    "load_campaign_from_json",
    "main",
    "plot_bytes_per_token_vs_seqlen",
    "plot_kv_fraction_of_gpu_ram",
    "plot_theory_vs_measured",
    "theory_check_caption",
    "title_context",
    "write_kv_scaling_summary",
]
