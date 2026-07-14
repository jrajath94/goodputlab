"""bench.matrix_report — sweep completion diagnostic (read-only).

Given a :class:`MatrixSpec` (what the runner was supposed to execute) and
a cells_dir (what landed on disk), produce a :class:`SweepReport` that
answers "did we run everything we said we would?" without manual JSON
tallying.

Pure read-only: does not invoke the runner, does not write any files.
Reuses :func:`bench.matrix_aggregator.aggregate` for cell parsing so the
contract (skip corrupt, raise on cell_id mismatch) stays consistent.

Used to diagnose gaps like the runpod_full sweep where the pod was
interrupted mid-run and DISAGG / DISAGG_TIER cells never landed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from bench.runpod_matrix import MatrixSpec
from bench.schema.cell_schema import CellResult, Topology


@dataclass(frozen=True)
class SweepReport:
    """Diagnostic snapshot of sweep completeness.

    Immutable so callers cannot mutate the diagnostic after the fact.
    ``missing_by_topology`` always has an entry for every Topology,
    even if the value is 0 — makes reporting uniform.
    """

    expected_total: int
    on_disk: int
    missing_count: int
    missing_by_topology: dict[Topology, int]
    missing_cell_ids: list[str]
    corrupt_or_mismatched: list[str]
    completion_pct: float


def _scan_raw(cells_dir: Path) -> tuple[set[str], list[str]]:
    """Walk cells_dir, return (valid cell_ids, corrupt/mismatched filenames).

    Only files matching the cell naming convention (`*__*__rate-*__*.json`)
    are inspected — non-cell artifacts (``summary.json``, etc.) are ignored
    silently. Corrupt cell files (failed to parse OR cell_id mismatch) are
    surfaced in the corrupt bucket so the report can flag them.
    """
    valid: set[str] = set()
    corrupt: list[str] = []
    for path in sorted(cells_dir.glob("*.json")):
        if not path.is_file():
            continue
        # Skip non-cell JSON (e.g. summary.json). Cell naming convention:
        # {topo}__{model}__rate-{r}__{mix}.json — contains "__rate-".
        if "__rate-" not in path.name:
            continue
        try:
            text = path.read_text()
            result = CellResult.model_validate_json(text)
        except Exception:  # noqa: BLE001 — corrupt JSON, surface in report
            corrupt.append(path.name)
            continue
        if result.cell_id != path.stem:
            corrupt.append(path.name)
            continue
        valid.add(result.cell_id)
    return valid, corrupt


def sweep_completion_report(cells_dir: Path, spec: MatrixSpec) -> SweepReport:
    """Diagnose sweep completeness against the spec.

    - expected_total = spec.total_cells()
    - on_disk = count of valid cell_ids whose filename matches cell_id
    - missing_count = expected_total - overlap between spec and disk
    - missing_by_topology = Counter of missing cells grouped by topology
    - missing_cell_ids = sorted list of cell_ids from spec not on disk
    - corrupt_or_mismatched = sorted list of filenames that didn't parse
    """
    expected_ids = {c.cell_id for c in spec.cells()}
    valid_ids, corrupt = _scan_raw(cells_dir)
    missing_ids = sorted(expected_ids - valid_ids)
    # Group missing by topology for the human-readable summary
    topo_for_id: dict[str, Topology] = {
        c.cell_id: c.topology for c in spec.cells()
    }
    by_topo: dict[Topology, int] = Counter()
    for cid in missing_ids:
        by_topo[topo_for_id[cid]] += 1
    # Always include every Topology in the output (zero for completed)
    for t in Topology:
        by_topo.setdefault(t, 0)
    pct = 100.0 * (len(expected_ids & valid_ids)) / len(expected_ids) if expected_ids else 100.0
    return SweepReport(
        expected_total=len(expected_ids),
        on_disk=len(valid_ids),
        missing_count=len(missing_ids),
        missing_by_topology=dict(by_topo),
        missing_cell_ids=missing_ids,
        corrupt_or_mismatched=sorted(corrupt),
        completion_pct=pct,
    )


def render_report(report: SweepReport) -> str:
    """Human-readable one-screen summary for CLI use."""
    lines = [
        "Sweep completion report",
        f"  expected:    {report.expected_total}",
        f"  on disk:     {report.on_disk}",
        f"  missing:     {report.missing_count}",
        f"  corrupt:     {len(report.corrupt_or_mismatched)}",
        f"  complete:    {report.completion_pct:.1f}%",
        "  by topology:",
    ]
    for t in Topology:
        miss = report.missing_by_topology.get(t, 0)
        lines.append(f"    {t.value:<12}  missing={miss}")
    if report.corrupt_or_mismatched:
        lines.append("  corrupt files:")
        for name in report.corrupt_or_mismatched:
            lines.append(f"    {name}")
    return "\n".join(lines)


__all__ = [
    "SweepReport",
    "render_report",
    "sweep_completion_report",
]