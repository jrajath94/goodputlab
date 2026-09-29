"""Campaign result writer for the KV-cache scaling study.

Writes ``bench/results/kv_scaling/<campaign_id>/``:

- ``cells.csv``: one row per cell. Samples are KV-cache bytes live
  during decode from block-manager accounting (NOT allocator bytes);
  ``measured_bytes_per_token = p50_bytes / total_tokens`` per cell
  (includes block-quantization rounding; the regression slope is the
  headline number). ``schema_version`` marks the quantity definitions.
- ``cells.json``: full campaign with raw samples, immutable.
- ``summary.json``: legacy contract (``slope_bytes_per_token``,
  ``intercept_bytes``, ``r2``, knee by batch, theory, revision) for the
  plotter's CSV+JSON loader.
- ``preflight.json``: the cost preflight record from before the run.
- ``cell_s<seq>_b<batch>.json``: per-cell artifacts so a sweep resumes
  by skipping cells whose JSON already exists and validates.
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench.kv_scaling.schema import KvScalingCampaign, KvScalingCell

# 1.1: samples renamed to kv_bytes_used_samples with precise quantity
# definitions (KV bytes live during decode from block accounting;
# baseline_bytes is F, the fixed non-KV overhead). CSV gains the
# measured_bytes_per_token column so the plotter never silently falls
# back to theory for the measured curve.
SCHEMA_VERSION = "1.1"

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results" / "kv_scaling"


def campaign_dir(results_root: Path, campaign_id: str) -> Path:
    """Return the output directory for one campaign."""
    return results_root / campaign_id


def write_campaign(
    campaign: KvScalingCampaign,
    results_root: Path = RESULTS_ROOT,
    campaign_id: str = "",
) -> tuple[Path, Path]:
    """Write ``cells.csv`` + ``cells.json``; return both paths.

    When ``campaign_id`` is empty, the campaign's own id is used.
    """
    cid = campaign_id or campaign.campaign_id
    out_dir = campaign_dir(results_root, cid)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_samples = max(len(c.kv_bytes_used_samples) for c in campaign.cells)
    fieldnames = [
        "schema_version",
        "campaign_id",
        "model_revision",
        "model_id",
        "gpu",
        "vllm_version",
        "seq_len",
        "batch_size",
        "n_samples",
        "total_tokens",
        "baseline_bytes",
        "theory_bytes_per_token",
        "measured_bytes_per_token",
        "p50_bytes",
        "p95_bytes",
        "method",
    ] + [f"sample_{i}" for i in range(max_samples)]

    csv_path = out_dir / "cells.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cell in campaign.cells:
            row: dict[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "campaign_id": cid,
                "model_revision": campaign.model_revision,
                "model_id": cell.model_id,
                "gpu": cell.gpu,
                "vllm_version": cell.vllm_version,
                "seq_len": cell.seq_len,
                "batch_size": cell.batch_size,
                "n_samples": cell.n_samples,
                "total_tokens": cell.total_tokens,
                "baseline_bytes": cell.baseline_bytes,
                "theory_bytes_per_token": campaign.theory_bytes_per_token,
                "measured_bytes_per_token": cell.measured_bytes_per_token,
                "p50_bytes": cell.p50_bytes,
                "p95_bytes": cell.p95_bytes,
                "method": cell.method,
            }
            for i, sample in enumerate(cell.kv_bytes_used_samples):
                row[f"sample_{i}"] = sample
            writer.writerow(row)

    json_path = out_dir / "cells.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "campaign": campaign.model_dump(mode="json"),
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    return csv_path, json_path


def write_summary_json(out_dir: Path, campaign: KvScalingCampaign) -> Path:
    """Write the legacy ``summary.json`` contract for the plotter."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "campaign_id": campaign.campaign_id,
        "measured_date": datetime.now(UTC).isoformat(),
        "slope_bytes_per_token": campaign.regression.slope_bpt,
        "intercept_bytes": campaign.regression.intercept_bytes,
        "r2": campaign.regression.r2,
        "knee_total_tokens_by_batch": (
            {str(campaign.knee.batch_size): campaign.knee.total_tokens}
            if campaign.knee.batch_size is not None
            else {"pooled": campaign.knee.total_tokens}
        ),
        "theory_bytes_per_token": campaign.theory_bytes_per_token,
        "model_revision": campaign.model_revision,
        "notes": (
            "KV-cache scaling study: per-cell p50 KV bytes (block-manager "
            "accounting) vs total tokens; baseline_bytes is F."
        ),
    }
    path = out_dir / "summary.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def write_preflight_json(out_dir: Path, preflight: dict[str, Any]) -> Path:
    """Write the cost preflight record from before the run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "preflight.json"
    path.write_text(json.dumps(preflight, indent=2) + "\n")
    return path


def write_cell_json(path: Path, cell: KvScalingCell) -> Path:
    """Write one per-cell artifact for resume.

    Computed fields are excluded: they are re-derived on load, and
    ``extra="forbid"`` would reject them as unknown inputs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cell.model_dump_json(indent=2, exclude_computed_fields=True) + "\n")
    return path


def read_cell_json(
    path: Path,
    *,
    seq_len: int,
    batch_size: int,
    n_samples: int,
    model_id: str,
) -> KvScalingCell | None:
    """Read a per-cell artifact; return None when missing or invalid.

    A cell only counts for resume when it validates against the schema
    AND matches the expected grid point and model. Anything else is
    treated as absent so the cell is re-measured, never trusted blindly.
    """
    try:
        raw = path.read_text()
    except OSError:
        return None
    try:
        cell = KvScalingCell.model_validate_json(raw)
    except ValueError:
        return None
    if (
        cell.seq_len != seq_len
        or cell.batch_size != batch_size
        or cell.n_samples != n_samples
        or cell.model_id != model_id
    ):
        return None
    return cell


__all__ = [
    "RESULTS_ROOT",
    "SCHEMA_VERSION",
    "campaign_dir",
    "read_cell_json",
    "write_campaign",
    "write_cell_json",
    "write_preflight_json",
    "write_summary_json",
]
