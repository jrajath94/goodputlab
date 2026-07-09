"""Schema uniformity tests across the four Phase 1 topology profiles.

Per D-05 (Common Endpoint Contract), every topology profile must expose
the same set of HTTP endpoints so downstream Phase 2-8 components (load
generation, router, bench) can speak to all of them uniformly.

Endpoints asserted here:

    GET  /health        → HTTP 200
    GET  /v1/models     → JSON with `data` containing `goodputlab-model`
    GET  /metrics       → Prometheus text containing `vllm:` or the proxy's
                          prefill/decode metric sections

These tests do NOT assert latency or throughput thresholds — those
numbers are [NOT YET MEASURED] until the pod run produces evidence and
live in the Phase 8 benchmark campaign.

By default the tests SKIP because the RunPod pod `t3son251d5gcvg` is
usually stopped during local development. To force execution, set:

    GOODPUTLAB_RUN_LIVE=1 pytest tests/test_schema_uniformity.py
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import requests

# Served model id per D-05 (common endpoint contract).
SERVED_MODEL_NAME = "goodputlab-model"

# Topology name → base URL (no trailing `/v1`; the individual endpoint
# paths are appended per-test).
TOPOLOGY_BASE_URLS: dict[str, str] = {
    "colocated": "http://localhost:18000",
    "chunked": "http://localhost:18001",
    "disagg": "http://localhost:19100",
    "disagg-tier": "http://localhost:19200",
}

# vLLM metric prefix used by the engine, plus the proxy's combined
# metric sections for disaggregated topologies.
VLLM_MARKERS = ("vllm:", "vllm_disagg_prefill_", "vllm_disagg_decode_")

# Live gate: default (unset / != '1') → skip. '1' → run.
LIVE_GATE_ENV = "GOODPUTLAB_RUN_LIVE"


def _live_mode() -> bool:
    """True iff GOODPUTLAB_RUN_LIVE is explicitly set to '1'."""
    return os.environ.get(LIVE_GATE_ENV) == "1"


_SKIP_REASON = (
    "set GOODPUTLAB_RUN_LIVE=1 to run live schema uniformity tests "
    "(requires `make up-colocated` / `make up-chunked` / "
    "`make up-disagg` / `make up-disagg-tier`)"
)


def _contains_vllm_marker(body: str) -> bool:
    """Return True if Prometheus body carries any recognised vLLM marker.

    We deliberately avoid checking exact metric names because they
    change between minor vLLM versions. These strings are [NOT YET
    MEASURED] values for downstream consumers — this test only checks
    the endpoint is wired and serving Prometheus text.
    """
    return any(marker in body for marker in VLLM_MARKERS)


@pytest.mark.parametrize("topology", sorted(TOPOLOGY_BASE_URLS.keys()))
@pytest.mark.skipif(not _live_mode(), reason=_SKIP_REASON)
def test_topology_exposes_common_endpoints(topology: str) -> None:
    """Every topology profile must expose `/health` returning HTTP 200."""
    base_url = TOPOLOGY_BASE_URLS[topology]
    response = requests.get(f"{base_url}/health", timeout=10)
    assert response.status_code == 200, (
        f"{topology}: /health returned {response.status_code} "
        f"(body={response.text[:200]!r})"
    )


@pytest.mark.parametrize("topology", sorted(TOPOLOGY_BASE_URLS.keys()))
@pytest.mark.skipif(not _live_mode(), reason=_SKIP_REASON)
def test_topology_v1_models_lists_served_model(topology: str) -> None:
    """`/v1/models` must list `goodputlab-model` so load gen + router code
    can resolve a known served-model id without topology-specific paths.
    """
    base_url = TOPOLOGY_BASE_URLS[topology]
    response = requests.get(f"{base_url}/v1/models", timeout=10)
    assert response.status_code == 200, (
        f"{topology}: /v1/models returned {response.status_code}"
    )
    payload: dict[str, Any] = response.json()
    assert "data" in payload, (
        f"{topology}: /v1/models response missing 'data' field: {payload!r}"
    )
    entries = payload["data"]
    assert isinstance(entries, list), (
        f"{topology}: /v1/models 'data' was not a list: {type(entries).__name__}"
    )
    ids = [entry.get("id") for entry in entries if isinstance(entry, dict)]
    assert SERVED_MODEL_NAME in ids, (
        f"{topology}: expected served model {SERVED_MODEL_NAME!r} in "
        f"/v1/models response, got {ids!r}"
    )


@pytest.mark.parametrize("topology", sorted(TOPOLOGY_BASE_URLS.keys()))
@pytest.mark.skipif(not _live_mode(), reason=_SKIP_REASON)
def test_topology_metrics_endpoint_serves_vllm_text(topology: str) -> None:
    """`/metrics` must serve Prometheus text containing vLLM (or proxy-
    combined vLLM) metric sections.

    Specific metric names and latencies are [NOT YET MEASURED] in
    Phase 1 — those land in Phase 8 BENCH.
    """
    base_url = TOPOLOGY_BASE_URLS[topology]
    response = requests.get(f"{base_url}/metrics", timeout=10)
    assert response.status_code == 200, (
        f"{topology}: /metrics returned {response.status_code}"
    )
    body = response.text
    assert _contains_vllm_marker(body), (
        f"{topology}: /metrics response did not contain any vLLM marker "
        f"({VLLM_MARKERS!r}); first 200 chars: {body[:200]!r}"
    )
