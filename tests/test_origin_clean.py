"""P5-1 invariant — origin/main must not contain leaked strategy docs.

Per workspace `suggestions/PORTFOLIO.md` finding #4, the public repo's
history must never include the Anthropic interview strategy docs.
This test invokes ``scripts/check_origin_clean.sh`` so the same gate
runs in local pytest AND in CI.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_origin_clean.sh"


def test_check_origin_clean_script_exists() -> None:
    assert SCRIPT.is_file(), f"missing {SCRIPT}"
    # Must be executable.
    import stat as _stat

    mode = SCRIPT.stat().st_mode
    assert mode & _stat.S_IXUSR, f"{SCRIPT} is not executable"


def test_check_origin_clean_script_returns_zero() -> None:
    """The script exits 0 when origin/main is clean (it currently is)."""
    if not SCRIPT.is_file():
        import pytest

        pytest.skip(f"missing {SCRIPT}")
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    # Allow either the legacy "OK: origin/main contains no leaked …" line
    # OR an exit-code failure (in which case stderr names the leaks).
    if completed.returncode != 0:
        raise AssertionError(
            f"check_origin_clean.sh exited {completed.returncode}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    assert "OK" in completed.stdout, completed.stdout


def test_check_origin_clean_script_detects_leak_pattern() -> None:
    """Negative control: feed a synthetic leak and confirm the script catches it.

    Runs the script against a redirected `git ls-tree` payload is not
    feasible without restructuring; instead we assert that the script's
    leak pattern covers every name listed in PORTFOLIO.md finding #4.
    """
    if not SCRIPT.is_file():
        import pytest

        pytest.skip(f"missing {SCRIPT}")
    body = SCRIPT.read_text(encoding="utf-8")
    for needle in (
        "Anthropic_Candidacy_Playbook",
        "MASTER_EXECUTION_PROMPT_CLAUDE",
        "Staff_Level_Projects_Spec_",
        "_Implementation_Brief",
        "EXECUTION_PROMPT",
    ):
        assert needle in body, f"leak pattern missing from script: {needle}"
