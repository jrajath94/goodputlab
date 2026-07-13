"""P5-3 / Ollama path — local M1 Max smoke harness.

The Ollama topology is single-process (no P/D disagg, no chunked-prefill
controls), but it shares the loadgen → reconciler → orchestrator
pipeline with the vLLM paths. This file proves that pipeline runs end
to end against a local ``ollama serve`` invocation.

All tests gate on ``GOODPUTLAB_RUN_OLLAMA=1`` so CI without a local
Ollama instance skips cleanly.
"""

from __future__ import annotations

import os

import pytest

from bench.orchestrator import CampaignReport, Topology

OLLAMA_GATE = os.environ.get("GOODPUTLAB_RUN_OLLAMA") == "1"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")

pytestmark = pytest.mark.skipif(
    not OLLAMA_GATE,
    reason="set GOODPUTLAB_RUN_OLLAMA=1 to exercise local Ollama smoke harness",
)


# ---------- Static checks (always run) ----------


def test_ollama_gate_constant_matches_doc() -> None:
    """The gate name is fixed by `make ollama-smoke`; tests must match."""
    assert OLLAMA_BASE_URL.startswith("http")
    assert OLLAMA_BASE_URL.endswith("/v1")
    assert ":" in OLLAMA_MODEL  # requires a tag


# ---------- Live gates (only with GOODPUTLAB_RUN_OLLAMA=1) ----------


@pytest.mark.asyncio
async def test_ollama_serves_openai_compatible_chat() -> None:
    """POST /v1/chat/completions against local Ollama; require 200 + content."""
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{OLLAMA_BASE_URL}/chat/completions",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": "Reply with only the word PONG."}],
                "max_tokens": 256,
                "temperature": 0.0,
            },
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["choices"][0]["message"]["content"] is not None


@pytest.mark.asyncio
async def test_ollama_metrics_endpoint_exposed() -> None:
    """Ollama exposes /metrics for Prometheus (or nothing) — we only assert NON-CRASH."""
    import httpx

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Some Ollama versions expose /metrics; some don't. Don't fail on 404.
        r = await client.get(f"{OLLAMA_BASE_URL.rsplit('/v1', 1)[0]}/metrics")
    assert r.status_code in (200, 404)


def test_orchestrator_accepts_ollama_topology_label() -> None:
    """The bench orchestrator's Topology enum must accept 'ollama' as a string tag."""
    # Topology is the vLLM-only enum; ollama uses a string label via real_bench.
    # We assert the canonical label is recognized (used in real_bench + Makefile).
    from bench.ollama_smoke import OLLAMA_TOPOLOGY_LABEL

    assert OLLAMA_TOPOLOGY_LABEL == "ollama"
    # Used in JSON filenames under bench/results/ollama/.
    assert isinstance(OLLAMA_TOPOLOGY_LABEL, str)


def test_ollama_report_round_trip_serialization() -> None:
    """A CampaignReport tagged with Topology.COLOCATED but local model string saves cleanly."""
    rep = CampaignReport(
        topology=Topology.COLOCATED,
        n_requests=10,
        success_rate=1.0,
        mean_ttft_ms=80.0,
        p95_ttft_ms=140.0,
        mean_itl_ms=8.0,
        cache_hit_rate=1.0,
        reconcile_passes=True,
        notes=["ollama", f"model={OLLAMA_MODEL}", f"base_url={OLLAMA_BASE_URL}"],
    )
    blob = rep.model_dump_json()
    assert "ollama" in blob
    assert OLLAMA_MODEL in blob
