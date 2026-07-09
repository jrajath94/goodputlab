"""Tests for obs/registry.py + obs/server.py — Prometheus exporter (OBS-01)."""

from __future__ import annotations

from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from obs.registry import MetricsRegistry
from obs.server import build_app


def _scrape_text(registry: MetricsRegistry) -> str:
    from prometheus_client import generate_latest

    return generate_latest(registry.registry).decode("utf-8")


def _has_metric(text: str, name: str) -> bool:
    """Return True iff ``name{`` or ``name `` or ``name\n`` appears in text."""
    return any(family.name == name for family in text_string_to_metric_families(text))


def _sample_value(text: str, name: str, labels: dict[str, str] | None = None) -> float:
    labels = labels or {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name != name:
                continue
            if all(sample.labels.get(k) == v for k, v in labels.items()):
                return sample.value
    return 0.0


# ---------- Registry ----------


def test_registry_initializes_empty() -> None:
    registry = MetricsRegistry()
    text = _scrape_text(registry)
    assert "goodputlab_ttft_ms" in text
    assert "goodputlab_request_total" in text


def test_registry_observe_ttft() -> None:
    registry = MetricsRegistry()
    registry.observe_ttft("interactive", 123.0)
    text = _scrape_text(registry)
    assert _sample_value(text, "goodputlab_ttft_ms_count", {"slo_class": "interactive"}) == 1.0
    assert _sample_value(text, "goodputlab_ttft_ms_sum", {"slo_class": "interactive"}) == 123.0


def test_registry_observe_itl() -> None:
    registry = MetricsRegistry()
    registry.observe_itl("batch", 25.0)
    registry.observe_itl("batch", 35.0)
    text = _scrape_text(registry)
    assert _sample_value(text, "goodputlab_itl_ms_count", {"slo_class": "batch"}) == 2.0
    assert _sample_value(text, "goodputlab_itl_ms_sum", {"slo_class": "batch"}) == 60.0


def test_registry_set_queue_depth() -> None:
    registry = MetricsRegistry()
    registry.set_queue_depth("prefill", 7)
    registry.set_queue_depth("decode", 3)
    text = _scrape_text(registry)
    assert _sample_value(text, "goodputlab_queue_depth", {"pool": "prefill"}) == 7.0
    assert _sample_value(text, "goodputlab_queue_depth", {"pool": "decode"}) == 3.0


def test_registry_kv_tier_counters() -> None:
    registry = MetricsRegistry()
    registry.inc_kv_hit("lmcache", n=5)
    registry.inc_kv_miss("lmcache", n=2)
    text = _scrape_text(registry)
    assert _sample_value(text, "goodputlab_kv_tier_hit_total", {"tier": "lmcache"}) == 5.0
    assert _sample_value(text, "goodputlab_kv_tier_miss_total", {"tier": "lmcache"}) == 2.0


def test_registry_spec_counters() -> None:
    registry = MetricsRegistry()
    registry.inc_spec_proposed(100)
    registry.inc_spec_accepted(75)
    text = _scrape_text(registry)
    assert _sample_value(text, "goodputlab_spec_proposed_total") == 100.0
    assert _sample_value(text, "goodputlab_spec_acceptance_total") == 75.0


def test_registry_role_flip_labeled() -> None:
    registry = MetricsRegistry()
    registry.inc_role_flip("prefill", "decode")
    registry.inc_role_flip("decode", "prefill")
    registry.inc_role_flip("prefill", "decode")  # second one
    text = _scrape_text(registry)
    assert _sample_value(
        text, "goodputlab_role_flip_total", {"from": "prefill", "to": "decode"}
    ) == 2.0
    assert _sample_value(
        text, "goodputlab_role_flip_total", {"from": "decode", "to": "prefill"}
    ) == 1.0


def test_registry_request_counters() -> None:
    registry = MetricsRegistry()
    registry.inc_request(200)
    registry.inc_request(200)
    registry.inc_request(500)
    registry.inc_request_error("HTTPError")
    text = _scrape_text(registry)
    assert _sample_value(text, "goodputlab_request_total", {"status_code": "200"}) == 2.0
    assert _sample_value(text, "goodputlab_request_total", {"status_code": "500"}) == 1.0
    assert _sample_value(
        text, "goodputlab_request_error_total", {"error_class": "HTTPError"}
    ) == 1.0


def test_registry_reconcile_gate_pass_counter() -> None:
    registry = MetricsRegistry()
    registry.inc_gate_pass()
    registry.inc_gate_pass()
    text = _scrape_text(registry)
    assert _sample_value(text, "goodputlab_reconcile_gate_passes_total") == 2.0


# ---------- Server ----------


def test_obs_server_metrics_endpoint_returns_prometheus_text() -> None:
    registry = MetricsRegistry()
    registry.observe_ttft("interactive", 100.0)
    app = build_app(registry)
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert "goodputlab_ttft_ms" in resp.text


def test_obs_server_health_endpoint() -> None:
    app = build_app()
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_obs_server_isolates_registries() -> None:
    a = MetricsRegistry()
    b = MetricsRegistry()
    a.observe_ttft("interactive", 50.0)
    # b has no observations yet
    client_b = TestClient(build_app(b))
    resp = client_b.get("/metrics")
    assert _sample_value(resp.text, "goodputlab_ttft_ms_count", {"slo_class": "interactive"}) == 0.0


def test_obs_server_handles_empty_registry() -> None:
    """Fresh registry → valid Prometheus text (HELP + TYPE lines, no samples)."""
    app = build_app()
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    # Should still contain at least the HELP/TYPE lines for declared metrics.
    assert "# HELP" in resp.text
    assert "# TYPE" in resp.text


def test_obs_server_role_flip_endpoint_to_endpoint() -> None:
    """End-to-end: inc counter → scrape → value present."""
    registry = MetricsRegistry()
    app = build_app(registry)
    client = TestClient(app)
    registry.inc_role_flip("tier", "decode")
    resp = client.get("/metrics")
    assert _sample_value(
        resp.text, "goodputlab_role_flip_total", {"from": "tier", "to": "decode"}
    ) == 1.0