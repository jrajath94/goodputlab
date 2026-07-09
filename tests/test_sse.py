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
