"""Cross-doc path consistency tests.

The repo docs reference several on-disk paths. When one of those paths
moves (e.g. ``autoscaler/TUNING.md`` → ``docs/autoscaler/TUNING.md`` per
GAP_REPORT.md §Gap 11), the references must move with it. These tests
pin that contract.

A failing test in this module means a doc still points at the old path
or the new path is missing. Run ``pytest tests/test_doc_paths.py`` to
audit.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _all_markdown() -> list[Path]:
    """Return every ``*.md`` under the repo (excluding ``.git`` and ``.venv``)."""
    out: list[Path] = []
    for p in REPO_ROOT.rglob("*.md"):
        rel = p.relative_to(REPO_ROOT)
        parts = rel.parts
        if parts[0] in {".git", ".venv", "node_modules", "suggestions"}:
            continue
        if any(part.startswith(".venv") for part in parts):
            continue
        out.append(p)
    return out


def test_tuning_md_lives_in_docs() -> None:
    """Gap 11: tuning doc must live under ``docs/`` (not as an orphan root dir)."""
    assert (REPO_ROOT / "docs" / "autoscaler" / "TUNING.md").is_file(), (
        "Expected docs/autoscaler/TUNING.md; the tuning doc was moved out "
        "of the repo-root autoscaler/ orphan directory per GAP_REPORT §11."
    )


def test_no_orphan_autoscaler_dir() -> None:
    """After the Gap 11 move, ``autoscaler/`` must not exist at the repo root."""
    assert not (REPO_ROOT / "autoscaler").exists(), (
        "Repo-root autoscaler/ should be removed after moving TUNING.md to docs/."
    )


def test_md_refs_point_at_new_tuning_path() -> None:
    """Active doc references must use ``docs/autoscaler/TUNING.md``.

    Skips ``docs/GAP_REPORT.md`` (historical snapshot — should not be
    rewritten) and ``CHANGELOG.md`` (historical release log).
    """
    skip_names = {"GAP_REPORT.md", "CHANGELOG.md"}
    old_pattern = re.compile(r"\bautoscaler/TUNING\.md\b")
    new_pattern = re.compile(r"\bdocs/autoscaler/TUNING\.md\b")
    offenders: list[str] = []
    for md in _all_markdown():
        if md.name in skip_names:
            continue
        text = md.read_text(encoding="utf-8")
        # Skip the test file itself (this module) which intentionally
        # references both paths in error messages.
        if md.name == "test_doc_paths.py":
            continue
        if old_pattern.search(text) and not new_pattern.search(text):
            offenders.append(str(md.relative_to(REPO_ROOT)))
    assert not offenders, (
        "Doc(s) still reference autoscaler/TUNING.md without a "
        "docs/autoscaler/TUNING.md companion link: " + ", ".join(offenders)
    )