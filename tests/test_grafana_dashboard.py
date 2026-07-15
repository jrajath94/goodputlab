"""OBS-02 invariant — Grafana dashboard JSON for GoodputLab is committable
and references every OBS-01 metric.

REQUIREMENTS.md OBS-02 (and ROADMAP.md Phase 8 success criterion #5) call
for a Grafana dashboard JSON committed to the repo with panels for:

- goodput (derived from counter deltas)
- TTFT p95 (histogram_quantile over ``goodputlab_ttft_ms``)
- ITL p95 (histogram_quantile over ``goodputlab_itl_ms``)
- queue depth per pool (``goodputlab_queue_depth``)
- KV-tier hit rate (computed from ``goodputlab_kv_tier_hit_total`` /
  ``goodputlab_kv_tier_miss_total``)
- spec acceptance rate (``goodputlab_spec_acceptance_total`` /
  ``goodputlab_spec_proposed_total``)
- role-flip count (``goodputlab_role_flip_total``)
- controller thrash + zero-drop evidence
  (``goodputlab_controller_thrash_total``,
  ``goodputlab_role_flip_inflight_dropped_total``)

This test parses the dashboard JSON, asserts it is valid Grafana schema,
and asserts every metric declared by ``obs/registry.py`` appears as a
Prometheus expression target in at least one panel. A future rename of
any metric in ``obs/registry.py`` must be mirrored here or this test
fails — that is the contract.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from prometheus_client import Counter

from obs.registry import MetricsRegistry

REPO_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_PATH = REPO_ROOT / "deploy" / "grafana" / "goodputlab.json"

# Required panels per ROADMAP Phase 8 success criterion #5.  Test asserts
# each token appears at least once in panel titles so the dashboard stays
# aligned with the spec even if metric names change.
_REQUIRED_PANEL_TOKENS = (
    "goodput",
    "ttft",
    "itl",
    "queue",
    "kv-tier",
    "spec",
    "role-flip",
    "controller-thrash",
)


def _load_dashboard() -> dict:
    assert DASHBOARD_PATH.is_file(), (
        f"Grafana dashboard missing: {DASHBOARD_PATH} (OBS-02 deliverable)."
    )
    text = DASHBOARD_PATH.read_text(encoding="utf-8")
    data = json.loads(text)  # raises if not valid JSON
    return data


def _all_promql_expressions(panels: list) -> list[str]:
    """Recursively walk every panel's ``targets[].expr`` field."""
    out: list[str] = []
    for panel in panels:
        if not isinstance(panel, dict):
            continue
        targets = panel.get("targets") or []
        for target in targets:
            expr = target.get("expr") if isinstance(target, dict) else None
            if isinstance(expr, str):
                out.append(expr)
        nested = panel.get("panels") or []
        if nested:
            out.extend(_all_promql_expressions(nested))
    return out


def _declared_metric_names() -> set[str]:
    """Every metric name declared by ``MetricsRegistry`` (the OBS-01 source of truth).

    ``prometheus_client.Counter._name`` returns the base name **without**
    the ``_total`` suffix, even though that's what is exposed on the
    scrape endpoint. We add the suffix back for Counters so the dashboard
    check matches what an operator types into PromQL.
    """
    metrics = MetricsRegistry()
    names: set[str] = set()
    for attr in (
        "ttft_ms",
        "itl_ms",
        "queue_depth",
        "prefix_index_size_bytes",
        "kv_tier_hit",
        "kv_tier_miss",
        "spec_acceptance",
        "spec_proposed",
        "role_flip",
        "request_total",
        "request_error",
        "reconcile_gate_passes",
        "no_history",
        "controller_thrash",
        "role_flip_inflight_dropped",
    ):
        handle = getattr(metrics, attr)
        name = getattr(handle, "_name", None)
        if not isinstance(name, str):
            continue
        names.add(name)
        if isinstance(handle, Counter):
            # Auto-suffix: Counters expose ``<name>_total`` on the wire.
            names.add(name + "_total")
    return names


def test_grafana_dashboard_is_valid_json() -> None:
    """Dashboard must be parseable JSON."""
    data = _load_dashboard()
    assert isinstance(data, dict), "Grafana dashboard JSON must be an object."


def test_grafana_dashboard_has_required_top_level_fields() -> None:
    """Modern Grafana dashboards require ``title``, ``panels``, ``schemaVersion``."""
    data = _load_dashboard()
    for field in ("title", "panels", "schemaVersion"):
        assert field in data, f"Grafana dashboard missing top-level field '{field}'."
    assert isinstance(data["panels"], list)
    assert len(data["panels"]) >= 1, "Dashboard must have at least one panel."
    assert isinstance(data["schemaVersion"], int)
    # Grafana 9+ uses schemaVersion >= 36; refuse to ship a v7-era JSON.
    assert data["schemaVersion"] >= 36, (
        f"schemaVersion={data['schemaVersion']} is below the modern Grafana "
        f"minimum (36). Re-author against the current Grafana schema."
    )


def test_grafana_dashboard_references_every_obs01_metric() -> None:
    """Every metric declared in ``obs/registry.py`` must be queried by some panel.

    Catches the failure mode where a new counter is added (e.g. thrash
    alarm) but the dashboard isn't updated. Without this guard, the
    dashboard silently goes stale.
    """
    data = _load_dashboard()
    expressions = _all_promql_expressions(data["panels"])
    joined = "\n".join(expressions)
    declared = _declared_metric_names()
    missing = sorted(name for name in declared if name not in joined)
    assert not missing, (
        "Grafana dashboard panels do not reference every OBS-01 metric. "
        "Add panel targets for: " + ", ".join(missing)
    )


def test_grafana_dashboard_has_required_panel_titles() -> None:
    """Dashboard must include a panel for each ROADMAP Phase 8 panel token."""
    data = _load_dashboard()
    titles: list[str] = []
    for panel in data["panels"]:
        title = panel.get("title", "") if isinstance(panel, dict) else ""
        if isinstance(title, str):
            titles.append(title.lower())
    joined = "\n".join(titles)
    missing = sorted(
        token for token in _REQUIRED_PANEL_TOKENS
        if not re.search(re.escape(token), joined)
    )
    assert not missing, (
        "Dashboard is missing panel(s) covering ROADMAP Phase 8 metrics: "
        + ", ".join(missing)
    )


def test_grafana_dashboard_marks_placeholder_state() -> None:
    """Until a sweep completes, the dashboard must be self-describing as a placeholder.

    Per GAP_REPORT.md §Gap 6: ``Option A`` (placeholder) is honest only if
    the JSON is documented as a placeholder, not "live".  We pin that
    with a top-level ``description`` or annotation field naming the gap.
    """
    data = _load_dashboard()
    description = data.get("description", "")
    assert isinstance(description, str)
    # Match either "placeholder" or an explicit reference to v1.1 or
    # benchmark sweep status — anything that signals "not yet live".
    needles = ("placeholder", "v1.1", "not yet measured", "no live data")
    assert any(n in description.lower() for n in needles), (
        "Dashboard description must self-describe as a placeholder or "
        "explicitly reference v1.1 / benchmark-sweep status so reviewers "
        "don't mistake it for a live dashboard. See GAP_REPORT §6."
    )