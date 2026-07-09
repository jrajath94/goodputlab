"""Tests for loadgen/chat.py — chat trace generator (LOAD-01)."""

from __future__ import annotations

import pytest

from core.trace import SloClass, WorkloadType
from loadgen.chat import DEFAULT_SYSTEM_PROMPT, ChatTraceGenerator, ChatWorkloadConfig


def test_chat_trace_has_n_requests() -> None:
    cfg = ChatWorkloadConfig(n_requests=25, seed=1)
    trace = ChatTraceGenerator(cfg).generate()
    assert trace.workload == WorkloadType.CHAT
    assert len(trace.requests) == 25


def test_chat_byte_identical() -> None:
    """LOAD-07: same config -> identical Trace JSON."""
    cfg = ChatWorkloadConfig(n_requests=10, seed=99)
    a = ChatTraceGenerator(cfg).generate().model_dump_json()
    b = ChatTraceGenerator(cfg).generate().model_dump_json()
    assert a == b


def test_chat_prompt_tokens_in_range() -> None:
    lo, hi = 500, 2000
    cfg = ChatWorkloadConfig(n_requests=30, seed=5, prompt_tokens_range=(lo, hi))
    trace = ChatTraceGenerator(cfg).generate()
    for r in trace.requests:
        assert lo <= r.prompt_tokens <= hi, (
            f"request {r.request_id} prompt_tokens={r.prompt_tokens} not in [{lo}, {hi}]"
        )


def test_chat_output_tokens_in_range() -> None:
    lo, hi = 50, 500
    cfg = ChatWorkloadConfig(n_requests=30, seed=5, output_tokens_range=(lo, hi))
    trace = ChatTraceGenerator(cfg).generate()
    for r in trace.requests:
        assert lo <= r.output_tokens <= hi


def test_chat_shared_system_prompt() -> None:
    cfg = ChatTraceGenerator(ChatWorkloadConfig(n_requests=5, seed=1)).generate()
    for r in cfg.requests:
        assert DEFAULT_SYSTEM_PROMPT in r.prompt_text
        assert r.prompt_text.startswith("System:")


def test_chat_default_is_interactive_slo() -> None:
    cfg = ChatWorkloadConfig()
    trace = ChatTraceGenerator(cfg).generate()
    for r in trace.requests:
        assert r.slo_class == SloClass.INTERACTIVE


def test_chat_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="n_requests must be > 0"):
        ChatWorkloadConfig(n_requests=0)
    with pytest.raises(ValueError, match="invalid prompt_tokens_range"):
        ChatWorkloadConfig(prompt_tokens_range=(2000, 500))  # inverted
    with pytest.raises(ValueError, match="invalid n_turns_range"):
        ChatWorkloadConfig(n_turns_range=(5, 1))  # inverted


def test_chat_prompt_text_is_multi_turn() -> None:
    """With default n_turns_range, prompts should contain 'User:' markers."""
    cfg = ChatWorkloadConfig(n_requests=10, seed=1, n_turns_range=(2, 4))
    trace = ChatTraceGenerator(cfg).generate()
    for r in trace.requests:
        assert r.prompt_text.count("User:") >= 2
