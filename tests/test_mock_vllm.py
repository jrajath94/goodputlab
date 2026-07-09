"""Tests for bench/mock_vllm.py — mock OpenAI-compatible server."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from bench.mock_vllm import MockVllmServer, build_mock_app


def test_build_mock_app_returns_fastapi() -> None:
    app = build_mock_app()
    assert app.title == "mock-vllm"


def test_mock_server_label_default() -> None:
    s = MockVllmServer()
    assert s.label == "mock-vllm"


def test_mock_server_label_custom() -> None:
    app = build_mock_app(server_label="bench-prefill")
    s = MockVllmServer(app=app)
    assert s.label == "bench-prefill"


def test_mock_server_uptime_increases() -> None:
    import time

    s = MockVllmServer()
    t0 = s.uptime_s
    time.sleep(0.01)
    t1 = s.uptime_s
    assert t1 > t0


def test_mock_vllm_models_endpoint() -> None:
    app = build_mock_app()
    client = TestClient(app)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert "data" in resp.json()


@pytest.mark.asyncio
async def test_mock_vllm_streams_n_tokens() -> None:
    """ASGI transport: POST /chat/completions stream=true → N tokens + DONE."""
    app = build_mock_app(base_latency_ms=0.1)
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://mock") as client,
        client.stream(
            "POST",
            "/chat/completions",
            json={"model": "mock", "max_tokens": 5, "stream": True},
        ) as resp,
    ):
        assert resp.status_code == 200
        lines = []
        async for line in resp.aiter_lines():
            if line.strip():
                lines.append(line)
    data_lines = [line for line in lines if line.startswith("data: ")]
    assert len(data_lines) == 8
    assert data_lines[-1] == "data: [DONE]"


@pytest.mark.asyncio
async def test_mock_vllm_zero_tokens_returns_role_and_done() -> None:
    app = build_mock_app()
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://mock") as client,
        client.stream(
            "POST",
            "/chat/completions",
            json={"model": "mock", "max_tokens": 0, "stream": True},
        ) as resp,
    ):
        lines = [line async for line in resp.aiter_lines() if line.strip()]
    data_lines = [line for line in lines if line.startswith("data: ")]
    assert len(data_lines) == 3