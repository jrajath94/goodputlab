"""REPRO-03 invariant — bench results parquet files must be committable.

REQUIREMENTS.md REPRO-03 states: "All bench results stored as parquet +
metadata JSON (HW, seed, version)". A blanket ``*.parquet`` blocklist in
``.gitignore`` would prevent ``bench/results/**/*.parquet`` from ever
being tracked.  This test guards the whitelist survives future
``echo *.parquet >> .gitignore`` carelessness.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _check_ignore(relpath: str) -> int:
    """Return git check-ignore exit code: 0=ignored, 1=NOT ignored, 128=error."""
    completed = subprocess.run(
        ["git", "check-ignore", "-v", relpath],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode


def test_bench_results_parquet_not_blocked() -> None:
    """A parquet inside a whitelisted bench subdir must be committable (REPRO-03).

    The blanket ``*.parquet`` rule (if reintroduced in the future)
    would block parquet anywhere; the previously whitelisted
    ``bench/results/real/``, ``runpod_pilot/``, ``runpod_full/``,
    ``ollama/`` paths must still commit parquet freely.
    """
    rc = _check_ignore("bench/results/real/_repro03_sample.parquet")
    assert rc != 0, (
        "bench/results/real/_repro03_sample.parquet is ignored by "
        ".gitignore; REPRO-03 requires bench parquet to be committable."
    )


def test_bench_results_parquet_not_blocked_in_runpod_full() -> None:
    """Same REPRO-03 check inside another whitelisted result subdir."""
    rc = _check_ignore("bench/results/runpod_full/_repro03_sample.parquet")
    assert rc != 0, (
        "bench/results/runpod_full/_repro03_sample.parquet is ignored; "
        "REPRO-03 requires bench parquet to be committable."
    )
