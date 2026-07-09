"""Replay runner — re-executes a trace from its seed.

The replay contract (LOAD-07) guarantees that the *set* of
request_ids and their *arrival order* are byte-identical across runs
of the same trace.  Wall-clock telemetry values (TTFT, completion_ts)
vary run-to-run since they depend on server response time, but the
deterministic request stream does not.
"""

from __future__ import annotations

from collections.abc import Callable

from core.trace import RequestSpec, RequestTelemetry, Trace
from loadgen.client import VllmHttpClient


class ReplayRunner:
    """Thin wrapper around ``VllmHttpClient.run`` for trace re-execution."""

    def __init__(self, client: VllmHttpClient) -> None:
        self._client = client

    async def replay(
        self,
        trace: Trace,
        routed_pool_for: Callable[[RequestSpec], str | None] | None = None,
    ) -> list[RequestTelemetry]:
        return await self._client.run(trace, routed_pool_for=routed_pool_for)


__all__ = ["ReplayRunner"]
