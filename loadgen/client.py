"""Async HTTP client that executes a Trace against an OpenAI-compatible endpoint.

Wraps ``httpx.AsyncClient`` with streaming + per-token timestamp capture.
Each ``RequestSpec`` becomes one ``RequestTelemetry`` (LOAD-05 schema).

The client is transport-agnostic: pass any ``httpx.AsyncBaseTransport``
in ``transport=`` to test without a real server.  Production code
constructs with ``base_url=`` and lets httpx default to the network
transport.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import httpx

from core.trace import RequestSpec, RequestTelemetry, Trace
from loadgen.arrival import OpenLoopScheduler
from loadgen.sse import TokenEvent, parse_sse_lines


def _infinite_clock() -> Iterator[int]:
    """Yield monotonic nanosecond timestamps forever (used during SSE parsing)."""
    while True:
        yield time.perf_counter_ns()


class VllmHttpClient:
    """Async client for OpenAI-compatible streaming chat completions."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "goodputlab-model",
        timeout_s: float = 120.0,
        max_concurrent: int = 64,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_s
        self._sem = asyncio.Semaphore(max_concurrent)
        self._transport = transport

    def _build_payload(self, spec: RequestSpec) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": [{"role": "user", "content": spec.prompt_text}],
            "max_tokens": spec.output_tokens,
            "stream": True,
            "temperature": 0.0,
        }

    async def _wait_until(self, arrival_ts_ns: int) -> None:
        """Sleep on the event loop until ``arrival_ts_ns`` (perf_counter_ns)."""
        now = time.perf_counter_ns()
        if arrival_ts_ns <= now:
            return
        delay_s = (arrival_ts_ns - now) / 1_000_000_000
        # asyncio.sleep with a tiny floor; 1ms is the smallest sensible wait.
        await asyncio.sleep(max(delay_s, 0.001))

    async def _stream_tokens(
        self,
        response: httpx.Response,
    ) -> AsyncIterator[TokenEvent]:
        """Yield ``TokenEvent``s from a streaming httpx response.

        The SSE parser is synchronous, so we drive the async line
        iterator manually and feed the parser one line at a time.
        """
        clock = _infinite_clock()
        line_iter = response.aiter_lines()
        line_idx = 0

        async def _next_line() -> str | None:
            try:
                return await anext(line_iter)
            except (StopAsyncIteration, httpx.RemoteProtocolError):
                return None

        # Drive the parser by feeding it lines one at a time.
        while True:
            line = await _next_line()
            if line is None:
                return
            # Recreate a single-element iterator to feed parse_sse_lines.
            def _one(s: str) -> Iterator[str]:
                yield s

            events = list(parse_sse_lines(_one(line), clock))
            for ev in events:
                yield ev
                # Cooperative yield so other tasks (in-flight requests) get a turn.
                await asyncio.sleep(0)
            line_idx += 1

    async def send_one(
        self,
        spec: RequestSpec,
        arrival_ts_ns: int,
        routed_pool: str | None = None,
    ) -> RequestTelemetry:
        """Send one request, return per-request telemetry (LOAD-05)."""
        await self._wait_until(arrival_ts_ns)
        enqueue_ts = time.perf_counter_ns()
        per_token: list[int] = []
        ttft_ms: float | None = None
        completion_ts: int | None = None
        status = 0
        error: str | None = None
        try:
            async with (
                httpx.AsyncClient(
                    timeout=self._timeout, transport=self._transport
                ) as client,
                client.stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    json=self._build_payload(spec),
                ) as resp,
            ):
                    status = resp.status_code
                    if status >= 400:
                        await resp.aread()
                        error = f"HTTP {status}: {resp.text[:200] if hasattr(resp, 'text') else ''}"
                    else:
                        async for ev in self._stream_tokens(resp):
                            if ttft_ms is None:
                                ttft_ms = (ev.ts_ns - enqueue_ts) / 1_000_000
                            per_token.append(ev.ts_ns)
                        completion_ts = time.perf_counter_ns()
        except (httpx.HTTPError, OSError) as exc:
            status = status or 599
            error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # last-resort guard; tests assert this path
            status = status or 599
            error = f"{type(exc).__name__}: {exc}"

        return RequestTelemetry(
            request_id=spec.request_id,
            enqueue_ts_ns=enqueue_ts,
            ttft_ms=ttft_ms,
            per_token_ts_ns=per_token,
            completion_ts_ns=completion_ts,
            status_code=status,
            error=error,
            routed_pool=routed_pool,
        )

    async def run(
        self,
        trace: Trace,
        routed_pool_for: Callable[[RequestSpec], str | None] | None = None,
    ) -> list[RequestTelemetry]:
        """Execute the entire trace; return telemetry in scheduler order.

        Concurrency is bounded by ``max_concurrent``; arrivals are
        respected (a request does not leave until its scheduled time,
        even if a slot is free).
        """
        scheduler = OpenLoopScheduler(trace)

        async def _one(spec: RequestSpec, ts: int) -> RequestTelemetry:
            async with self._sem:
                pool = routed_pool_for(spec) if routed_pool_for else None
                return await self.send_one(spec, ts, routed_pool=pool)

        tasks = [asyncio.create_task(_one(spec, ts)) for spec, ts in scheduler]
        return await asyncio.gather(*tasks)


__all__ = ["VllmHttpClient"]
