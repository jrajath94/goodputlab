"""CLI: sweep completion diagnostic for an already-run sweep.

Usage::

    python -m scripts.sweep_report --config configs/runpod_matrix_full.yaml

Reads the same MatrixSpec from the YAML the runner used, walks the
cells_dir the runner wrote to, and prints a one-screen summary of
"did everything we said we'd run, actually run?" plus the missing
cell_ids so the operator knows what to re-run.

Non-zero exit if any cells are missing or corrupt — useful in CI or
post-sweep automation to gate "did the sweep finish?" checks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bench.matrix_report import render_report, sweep_completion_report
from bench.schema.matrix_config import load_matrix_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sweep completion diagnostic. Compares the MatrixSpec "
        "in --config against the cells on disk under --cells-dir.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a runpod_matrix_*.yaml (sweep spec the runner used).",
    )
    parser.add_argument(
        "--cells-dir",
        type=Path,
        default=None,
        help="Directory of CellResult JSONs. Defaults to config's output_dir.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the summary header; suppress missing cell_id list.",
    )
    args = parser.parse_args(argv)

    config = load_matrix_config(args.config)
    spec = config.to_matrix_spec()
    cells_dir = args.cells_dir or config.output_dir
    report = sweep_completion_report(cells_dir, spec)
    print(render_report(report))
    if not args.quiet and report.missing_cell_ids:
        print("\nMissing cell_ids (paste into run_pending to resume):")
        for cid in report.missing_cell_ids:
            print(f"  {cid}")
    # Exit non-zero if gaps remain — useful for post-sweep gates.
    return 0 if report.missing_count == 0 and not report.corrupt_or_mismatched else 1


if __name__ == "__main__":
    raise SystemExit(main())