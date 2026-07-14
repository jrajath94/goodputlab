"""bench.matrix_aggregator — collect CellResult JSONs into a summary.

Pure read-only module.  Given a directory of ``{cell_id}.json`` files
produced by :mod:`bench.cell_runner`, produce:

- :class:`SummaryStats` (overall TTFT/ITL/reconcile counts)
- per-topology breakdown: ``dict[Topology, list[CellResult]]``
- :class:`CampaignReport` (cells completed, cost, duration)
- ``summary.json`` written via :func:`write_summary`

Skips non-JSON + corrupt JSON files; raises ``ValueError`` if any
``*.json`` filename does not match the cell's ``cell_id`` field.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from bench.schema.cell_schema import (
    CellResult,
    SummaryStats,
    Topology,
)


class CampaignReport(BaseModel):
    """Top-level rollup: how many cells ran, how long, how much."""

    model_config = ConfigDict(extra="forbid")

    n_cells_completed: int = Field(ge=0)
    n_cells_failed: int = Field(ge=0)
    total_duration_s: float = Field(ge=0.0)
    cost_usd: float = Field(ge=0.0)
    pod_id: str
    started_at: datetime
    ended_at: datetime


def aggregate(cells_dir: Path) -> list[CellResult]:
    """Load all valid CellResult JSONs from ``cells_dir`` (sorted by cell_id).

    - Skips non-JSON files silently.
    - Skips corrupt JSON files silently (the per-cell JSON may be partial
      if a previous run crashed mid-write; those cells must be re-run).
    - Raises ``ValueError`` if any JSON's ``cell_id`` does not match its
      filename — this guards against silent rename mistakes that would
      produce a result table whose row labels don't match the file names.
    """
    cells_dir = Path(cells_dir)
    json_files = sorted(p for p in cells_dir.glob("*.json") if p.is_file())
    if not json_files:
        raise ValueError(f"no CellResult JSONs in {cells_dir}")

    results: list[CellResult] = []
    for path in json_files:
        try:
            text = path.read_text()
            result = CellResult.model_validate_json(text)
        except Exception:  # noqa: BLE001 — corrupt/partial JSON
            continue
        if result.cell_id != path.stem:
            raise ValueError(
                f"cell_id mismatch: file={path.name} cell_id={result.cell_id!r}"
            )
        results.append(result)

    if not results:
        raise ValueError(
            f"no valid CellResult JSONs in {cells_dir} "
            f"(found {len(json_files)} .json files, none parsed)"
        )

    results.sort(key=lambda r: r.cell_id)
    return results


def per_topology_breakdown(
    results: list[CellResult],
) -> dict[Topology, list[CellResult]]:
    """Group cells by topology.  Always returns an entry for every Topology."""
    grouped: dict[Topology, list[CellResult]] = defaultdict(list)
    for r in results:
        grouped[r.topology].append(r)
    return dict(grouped)


def _cost_per_cell(duration_s: float, cost_per_hour_usd: float) -> float:
    return (duration_s / 3600.0) * cost_per_hour_usd


def write_summary(
    cells_dir: Path,
    campaign: CampaignReport,
    cost_per_hour_usd: float,
) -> Path:
    """Write ``summary.json`` with campaign, summary stats, per-topology, cost.

    Layout::

        {
          "campaign": {...CampaignReport fields...},
          "summary":  {...SummaryStats fields...},
          "per_topology": { "<topo>": { "n_cells", "mean_ttft_ms", ... } },
          "cost": { "per_hour_usd", "n_cells", "total_usd" }
        }
    """
    cells_dir = Path(cells_dir)
    results = aggregate(cells_dir)
    summary = SummaryStats.from_results(results)
    grouped = per_topology_breakdown(results)

    per_topology_payload: dict[str, dict[str, object]] = {}
    for topo, cells in sorted(grouped.items(), key=lambda kv: kv[0].value):
        n = len(cells)
        reconciled = [c for c in cells if c.reconcile_passes]
        n_reconciled = len(reconciled)
        # Honest aggregate: means over reconciled only. Stub cells
        # have mean_ttft_ms=0 because they never produced telemetry —
        # averaging those zeros with real measurements silently masks
        # the truth. See SummaryStats.from_results for the same logic.
        if n_reconciled:
            mean_ttft = sum(c.mean_ttft_ms for c in reconciled) / n_reconciled
            mean_itl = sum(c.mean_itl_ms for c in reconciled) / n_reconciled
            success_rate = sum(c.success_rate for c in reconciled) / n_reconciled
        else:
            mean_ttft = 0.0
            mean_itl = 0.0
            success_rate = 0.0
        per_topology_payload[topo.value] = {
            "n_cells": n,
            "n_reconciled": n_reconciled,
            "mean_ttft_ms": mean_ttft,
            "mean_itl_ms": mean_itl,
            "success_rate": success_rate,
            "n_unreconciled": n - n_reconciled,
            "n_thermal_warnings": sum(1 for c in cells if c.has_thermal_warning),
            "total_cost_usd": sum(
                _cost_per_cell(c.duration_s, cost_per_hour_usd) for c in cells
            ),
        }

    total_cost_usd = sum(
        _cost_per_cell(c.duration_s, cost_per_hour_usd) for c in results
    )

    payload: dict[str, object] = {
        "campaign": campaign.model_dump(mode="json"),
        "summary": summary.model_dump(mode="json"),
        "per_topology": per_topology_payload,
        "cost": {
            "per_hour_usd": cost_per_hour_usd,
            "n_cells": len(results),
            "total_usd": total_cost_usd,
        },
    }

    out_path = cells_dir / "summary.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(out_path)
    return out_path


__all__ = [
    "CampaignReport",
    "aggregate",
    "per_topology_breakdown",
    "write_summary",
]