"""Mock vLLM server — FastAPI app that streams fake tokens.

Used by the bench orchestrator when no real vLLM is available.  Honors
``max_tokens``; emits one token per ~1ms with a small per-token latency
distribution; returns a real finish_reason.

Production deployment swaps this for a real vLLM process behind the
same OpenAI-compatible contract.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse


def build_mock_app(
    base_latency_ms: float = 1.0,
    server_label: str = "mock-vllm",
) -> FastAPI:
    """Build a mock OpenAI-compatible streaming app."""
    app = FastAPI(title=server_label)
    app.state.label = server_label

    @app.post("/chat/completions")
    async def chat_completions(request: Request) -> Response:
        body = await request.json()
        max_tokens = int(body.get("max_tokens", 5))

        async def gen() -> AsyncIterator[str]:
            # First chunk with role.
            role_chunk = {
                "id": "cmpl-mock",
                "choices": [{"delta": {"role": "assistant"}, "index": 0}],
            }
            yield f"data: {json.dumps(role_chunk)}\n\n"
            for i in range(max_tokens):
                await asyncio.sleep(base_latency_ms / 1000.0)
                chunk = {
                    "id": "cmpl-mock",
                    "choices": [
                        {"delta": {"content": f"tok{i}"}, "index": 0}
                    ],
                }
                encoded: str = json.dumps(chunk)
                yield f"data: {encoded}\n\n"
            finish_chunk = {
                "id": "cmpl-mock",
                "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
            }
            yield f"data: {json.dumps(finish_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        return {"data": [{"id": "mock-model", "object": "model"}]}

    return app


class MockVllmServer:
    """Wraps a FastAPI app; provides the base_url the client needs."""

    def __init__(self, app: FastAPI | None = None) -> None:
        self.app = app if app is not None else build_mock_app()
        self._start_time = time.perf_counter()

    @property
    def label(self) -> str:
        return str(self.app.state.label)

    @property
    def uptime_s(self) -> float:
        return time.perf_counter() - self._start_time


__all__ = ["MockVllmServer", "build_mock_app"]