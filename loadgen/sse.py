"""Pure SSE (Server-Sent Events) parser for OpenAI-compatible streams.

vLLM and most OpenAI-compatible servers stream responses as:

    data: {"id":"...","choices":[{"delta":{"content":"Hello"}}]}
    data: {"id":"...","choices":[{"delta":{"content":" world"}}]}
    data: [DONE]

This module converts a line iterator into a sequence of ``TokenEvent``
records, each carrying the token text and the monotonic timestamp
recorded by the caller.  The parser is pure: it does no I/O, so it is
trivially unit-testable.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenEvent:
    """A single emitted token from a streaming completion."""

    content: str
    ts_ns: int  # caller-recorded wall-clock at parse time
    finish_reason: str | None = None  # populated on the final chunk


def parse_sse_lines(
    lines: Iterator[str],
    clock_ns: Iterator[int],
) -> Iterator[TokenEvent]:
    """Parse SSE-formatted lines into ``TokenEvent``s.

    Parameters
    ----------
    lines
        Iterator over raw response lines (the caller splits on ``\\n``).
    clock_ns
        Iterator over monotonic timestamps (ns) — one per parsed event.
        Each ``next(clock_ns)`` call is expected to return a non-decreasing
        int.  The caller wires this to ``time.perf_counter_ns`` in
        production.

    Yields
    ------
    TokenEvent for every ``data: {...}`` line that contains a non-empty
    ``choices[0].delta.content``.  Stops on ``data: [DONE]`` or stream
    end.  Skips comments (``:``) and malformed lines silently.
    """
    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data: "):
            continue
        payload_str = line[len("data: ") :].strip()
        if payload_str == "[DONE]":
            return
        try:
            payload = json.loads(payload_str)
        except json.JSONDecodeError:
            continue
        choices = payload.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if not content:
            continue
        ts = next(clock_ns)
        finish = choices[0].get("finish_reason")
        yield TokenEvent(content=content, ts_ns=ts, finish_reason=finish)


__all__ = ["TokenEvent", "parse_sse_lines"]
