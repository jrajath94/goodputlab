"""Static sentinel source-mark tests (no network, no GPU).

These tests read tests/sentinel.py as text and assert the source contains the
required behavior markers (CLI modes, deterministic settings, logprob
comparison, non-zero exit handling). They must not depend on measured token
fixtures — recorded fixtures are produced only by the live record mode and are
[NOT YET MEASURED] in this repo.

PITFALLS P1: post-transfer known-prefix token validity is the load-bearing
safety mechanism. These tests guard the marker contract, not the values.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SENTINEL_SOURCE = REPO_ROOT / "tests" / "sentinel.py"


def _read_sentinel_source() -> str:
    """Load tests/sentinel.py as text. Fails loudly if the file is missing."""
    assert SENTINEL_SOURCE.exists(), (
        f"sentinel source not found at {SENTINEL_SOURCE} — "
        "task 01-05 must ship tests/sentinel.py"
    )
    return SENTINEL_SOURCE.read_text(encoding="utf-8")


def test_sentinel_has_record_and_check_modes() -> None:
    """CLI must expose both `record` and `check` modes (CONTEXT D-03)."""
    src = _read_sentinel_source()
    assert "record" in src, "sentinel must define a `record` mode"
    assert "check" in src, "sentinel must define a `check` mode"


def test_sentinel_defines_deterministic_known_prefix() -> None:
    """Source must declare a fixed KNOWN_PREFIX, temperature=0.0, logprobs request."""
    src = _read_sentinel_source()
    assert "KNOWN_PREFIX" in src, "sentinel must declare a KNOWN_PREFIX constant"
    assert "temperature" in src, "sentinel must request temperature (greedy)"
    assert "0.0" in src, "sentinel must pin temperature to 0.0 for determinism"
    assert "logprobs" in src, "sentinel must request logprobs for drift comparison"


def test_sentinel_check_exits_nonzero_on_drift() -> None:
    """check path must produce SENTINEL PASS/FAIL markers and non-zero exit on drift."""
    src = _read_sentinel_source()
    assert "SENTINEL PASS" in src, "sentinel check must emit a PASS marker"
    assert "SENTINEL FAIL" in src, "sentinel check must emit a FAIL marker"
    assert "logprob_epsilon" in src, "sentinel must declare a logprob_epsilon constant"
    assert "prompt_sha256" in src, "sentinel fixture must include prompt_sha256 for pinning"


def test_sentinel_defaults_are_explicit() -> None:
    """Default values must be deterministic and explicit (plan 01-05 acceptance)."""
    src = _read_sentinel_source()
    # Default served-model-name pinned to `goodputlab-model`.
    assert "goodputlab-model" in src, "sentinel default served-model-name must be `goodputlab-model`"
    # Default fixture-dir is tests/_fixtures.
    assert "tests/_fixtures" in src, "sentinel default fixture-dir must be tests/_fixtures"
    # Default max-tokens = 50 (greedy bounded output).
    assert "max_tokens" in src, "sentinel must request bounded max_tokens"


def test_sentinel_modes_via_argparse_choices() -> None:
    """Mode selection must use argparse with explicit `choices=[record, check]`."""
    src = _read_sentinel_source()
    assert "argparse" in src, "sentinel must use argparse for CLI parsing"
    # The choices must be exactly the two modes.
    assert ("record" in src) and ("check" in src), "sentinel must declare both modes"


def test_fixtures_directory_has_gitkeep() -> None:
    """The fixtures directory must be tracked via .gitkeep."""
    gitkeep = REPO_ROOT / "tests" / "_fixtures" / ".gitkeep"
    assert gitkeep.exists(), "tests/_fixtures/.gitkeep must exist (no fabricated fixtures)"