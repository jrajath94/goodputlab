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
    """A single emitted token from a streaming completion.

    ``reasoning`` is True when the event came from a reasoning-model
    reasoning channel (``delta.reasoning_content`` or ``delta.reasoning``)
    rather than the visible ``delta.content`` channel. Reasoning events
    count toward TTFT and ITL the same as visible tokens because they
    are model-emitted and dominate wall-clock latency for short prompts
    on reasoning models (Ollama qwen3 family in particular).
    """

    content: str
    ts_ns: int  # caller-recorded wall-clock at parse time
    finish_reason: str | None = None  # populated on the final chunk
    reasoning: bool = False


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
        # OpenAI-compat content channel
        content = delta.get("content")
        # Ollama reasoning-model channels. qwen3 streams these BEFORE
        # any visible content; without this branch, the first delta
        # yields nothing and per_token_ts_ns ends up empty.
        reasoning_content = delta.get("reasoning_content")
        reasoning_text = delta.get("reasoning")
        if content:
            text = content
            is_reasoning = False
        elif reasoning_content or reasoning_text:
            text = reasoning_content or reasoning_text
            is_reasoning = True
        else:
            continue
        ts = next(clock_ns)
        finish = choices[0].get("finish_reason")
        yield TokenEvent(
            content=text,
            ts_ns=ts,
            finish_reason=finish,
            reasoning=is_reasoning,
        )


__all__ = ["TokenEvent", "parse_sse_lines"]
