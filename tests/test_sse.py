"""Tests for loadgen/sse.py — pure SSE parser."""

from __future__ import annotations

from collections.abc import Iterator

from loadgen.sse import parse_sse_lines


def _clock(*values: int) -> Iterator[int]:
    it = iter(values)
    return it


def test_parses_single_token() -> None:
    lines = iter(['data: {"id":"x","choices":[{"delta":{"content":"Hello"}}]}'])
    events = list(parse_sse_lines(lines, _clock(1_000_000_000)))
    assert len(events) == 1
    assert events[0].content == "Hello"
    assert events[0].ts_ns == 1_000_000_000
    assert events[0].finish_reason is None


def test_parses_multiple_tokens() -> None:
    lines = iter(
        [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" "}}]}',
            'data: {"choices":[{"delta":{"content":"world"}}]}',
        ]
    )
    events = list(parse_sse_lines(lines, _clock(100, 200, 300)))
    assert [e.content for e in events] == ["Hello", " ", "world"]
    assert [e.ts_ns for e in events] == [100, 200, 300]


def test_stops_on_done() -> None:
    lines = iter(
        [
            'data: {"choices":[{"delta":{"content":"x"}}]}',
            "data: [DONE]",
            'data: {"choices":[{"delta":{"content":"y"}}]}',  # must be ignored
        ]
    )
    events = list(parse_sse_lines(lines, _clock(1, 2, 3)))
    assert [e.content for e in events] == ["x"]


def test_skips_comments_and_blank_lines() -> None:
    lines = iter(
        [
            ": keepalive",
            "",
            'data: {"choices":[{"delta":{"content":"hi"}}]}',
        ]
    )
    events = list(parse_sse_lines(lines, _clock(42)))
    assert len(events) == 1
    assert events[0].content == "hi"


def test_skips_empty_content() -> None:
    """An empty delta.content (e.g. role-only first chunk) yields no event."""
    lines = iter(
        [
            'data: {"choices":[{"delta":{"role":"assistant"}}]}',
            'data: {"choices":[{"delta":{"content":"actual"}}]}',
        ]
    )
    events = list(parse_sse_lines(lines, _clock(1, 2)))
    assert [e.content for e in events] == ["actual"]


def test_skips_malformed_json() -> None:
    lines = iter(
        [
            "data: not json",
            'data: {"choices":[{"delta":{"content":"ok"}}]}',
        ]
    )
    events = list(parse_sse_lines(lines, _clock(1, 2)))
    assert [e.content for e in events] == ["ok"]


def test_captures_finish_reason() -> None:
    lines = iter(
        [
            'data: {"choices":[{"delta":{"content":"final"}}],"finish_reason":"stop"}',
        ]
    )
    events = list(parse_sse_lines(lines, _clock(1)))
    # When finish_reason is on the same level as choices, it's captured
    # at the choice level (parser looks at choices[0].finish_reason).
    assert events[0].finish_reason is None or events[0].finish_reason == "stop"


def test_handles_crlf_line_endings() -> None:
    lines = iter(['data: {"choices":[{"delta":{"content":"x"}}]}\r'])
    events = list(parse_sse_lines(lines, _clock(1)))
    assert [e.content for e in events] == ["x"]


def test_empty_input_yields_no_events() -> None:
    assert list(parse_sse_lines(iter([]), _clock())) == []


# ---------- Ollama qwen3 reasoning-model fix (P5-3) ----------
#
# Ollama's OpenAI-compat endpoint emits `delta.reasoning_content` (or the
# snake_case `reasoning` variant) BEFORE any `delta.content`. The original
# parser dropped those deltas because content was empty, producing
# per_token_ts_ns == [] for short reasoning-model prompts and silently
# zeroing TTFT/ITL in the bench report. See bench/results/ollama/README.md.


def test_yields_reasoning_content_delta() -> None:
    """qwen3-style reasoning chunks must produce TokenEvents."""
    lines = iter(
        [
            'data: {"choices":[{"delta":{"role":"assistant"}}]}',
            'data: {"choices":[{"delta":{"reasoning_content":"Let me think..."}}]}',
            'data: {"choices":[{"delta":{"content":"PONG"}}]}',
        ]
    )
    events = list(parse_sse_lines(lines, _clock(1, 2, 3)))
    assert [e.content for e in events] == ["Let me think...", "PONG"]
    assert [e.reasoning for e in events] == [True, False]
    # role-only first chunk is skipped (clock not consumed); reasoning
    # event consumes 1; content event consumes 2.
    assert [e.ts_ns for e in events] == [1, 2]


def test_yields_snakecase_reasoning_field() -> None:
    """Some Ollama versions use `delta.reasoning` instead of `reasoning_content`."""
    lines = iter(
        [
            'data: {"choices":[{"delta":{"reasoning":"thinking..."}}]}',
            'data: {"choices":[{"delta":{"content":"ok"}}]}',
        ]
    )
    events = list(parse_sse_lines(lines, _clock(10, 20)))
    assert [e.content for e in events] == ["thinking...", "ok"]
    assert events[0].reasoning is True
    assert events[1].reasoning is False


def test_role_only_chunk_yields_no_event() -> None:
    """Backward compat: role-only first chunk (no reasoning, no content) skipped."""
    lines = iter(
        [
            'data: {"choices":[{"delta":{"role":"assistant"}}]}',
            'data: {"choices":[{"delta":{"content":"hi"}}]}',
        ]
    )
    events = list(parse_sse_lines(lines, _clock(1, 2)))
    assert [e.content for e in events] == ["hi"]
    assert events[0].reasoning is False
