"""Tests for loadgen/replay.py — replay runner."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from core.trace import (
    ArrivalConfig,
    RequestSpec,
    SloClass,
    Trace,
    WorkloadType,
)
from loadgen.client import VllmHttpClient
from loadgen.replay import ReplayRunner

# Minimal mock app.
mock_app = FastAPI()


@mock_app.post("/chat/completions")
async def chat_completions(request: Request) -> StreamingResponse:
    body = await request.json()
    n = body.get("max_tokens", 2)

    async def gen() -> AsyncIterator[str]:
        for i in range(n):
            await asyncio.sleep(0.001)
            content = f"t{i}"
            payload = f'{{"id":"x","choices":[{{"delta":{{"content":"{content}"}}}}]}}'
            yield f"data: {payload}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def _spec(i: int) -> RequestSpec:
    return RequestSpec(
        request_id=f"r{i:04d}",
        slo_class=SloClass.INTERACTIVE,
        workload=WorkloadType.CHAT,
        prompt_tokens=10,
        output_tokens=2,
        prompt_text=f"hi {i}",
    )


def _trace(n: int, rate: float = 100) -> Trace:
    return Trace(
        workload=WorkloadType.CHAT,
        seed=42,
        duration_s=10.0,
        arrival=ArrivalConfig(process="poisson", rate_per_sec=rate, seed=42),
        requests=[_spec(i) for i in range(n)],
    )


def _client() -> VllmHttpClient:
    return VllmHttpClient(
        base_url="http://mock",
        model="m",
        max_concurrent=4,
        transport=httpx.ASGITransport(app=mock_app),
    )


@pytest.mark.asyncio
async def test_replay_produces_telemetry_for_each_request() -> None:
    runner = ReplayRunner(_client())
    results = await runner.replay(_trace(n=4))
    assert len(results) == 4
    assert all(r.status_code == 200 for r in results)


@pytest.mark.asyncio
async def test_replay_byte_identical_request_ids() -> None:
    """LOAD-07: same trace -> same request IDs in same order."""
    runner = ReplayRunner(_client())
    trace = _trace(n=6)
    a = await runner.replay(trace)
    b = await runner.replay(trace)
    a_ids = [r.request_id for r in a]
    b_ids = [r.request_id for r in b]
    assert a_ids == b_ids


@pytest.mark.asyncio
async def test_replay_propagates_routed_pool() -> None:
    runner = ReplayRunner(_client())
    results = await runner.replay(
        _trace(n=3),
        routed_pool_for=lambda s: "prefill",
    )
    assert all(r.routed_pool == "prefill" for r in results)


@pytest.mark.asyncio
async def test_replay_with_failure_marks_status_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    client = VllmHttpClient(
        base_url="http://x",
        model="m",
        max_concurrent=2,
        transport=httpx.MockTransport(handler),
    )
    runner = ReplayRunner(client)
    results = await runner.replay(_trace(n=2))
    assert all(r.status_code == 503 for r in results)
    assert all(r.error is not None for r in results)
