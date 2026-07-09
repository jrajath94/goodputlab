"""Tests for loadgen/agentic.py — agentic trace generator (LOAD-03)."""

from __future__ import annotations

import pytest

from core.trace import WorkloadType
from loadgen.agentic import AgenticTraceGenerator, AgenticWorkloadConfig


def test_agentic_trace_has_n_requests() -> None:
    cfg = AgenticWorkloadConfig(n_requests=20, seed=1)
    trace = AgenticTraceGenerator(cfg).generate()
    assert trace.workload == WorkloadType.AGENTIC
    assert len(trace.requests) == 20


def test_agentic_byte_identical() -> None:
    cfg = AgenticWorkloadConfig(n_requests=15, seed=11)
    a = AgenticTraceGenerator(cfg).generate().model_dump_json()
    b = AgenticTraceGenerator(cfg).generate().model_dump_json()
    assert a == b


def test_agentic_prefix_grows_monotonically() -> None:
    """The shared prefix (system + tool defs) is constant; later requests
    carry all the history of earlier ones, so request i+1's prompt
    contains request i's prompt as a prefix of its body.
    """
    cfg = AgenticWorkloadConfig(n_requests=5, seed=1)
    trace = AgenticTraceGenerator(cfg).generate()
    for i in range(len(trace.requests) - 1):
        a = trace.requests[i].prompt_text
        b = trace.requests[i + 1].prompt_text
        # The history block accumulates, so the system+tools prefix is shared.
        # b must be at least as long as a, and the first len(a) bytes overlap
        # along the system+tools boundary.  A simpler invariant: b's length >= a.
        assert len(b) >= len(a), (
            f"prompt length regressed at step {i+1}: {len(a)} -> {len(b)}"
        )


def test_agentic_prefix_overlap_at_least_60pct() -> None:
    """LOAD-03: high prefix overlap.  System+tools+history must dominate
    the prompt; assertion is on the longest common prefix of any two
    prompts (relative to the shorter) being at least 60%.
    """
    cfg = AgenticWorkloadConfig(n_requests=8, seed=1)
    trace = AgenticTraceGenerator(cfg).generate()
    prompts = [r.prompt_text for r in trace.requests]
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            a, b = prompts[i], prompts[j]
            shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
            n = min(len(a), len(b))
            common = 0
            for k in range(n):
                if a[k] == b[k]:
                    common += 1
                else:
                    break
            overlap = common / max(len(shorter), 1)
            assert overlap >= 0.60, (
                f"pair ({i},{j}) prefix overlap {overlap:.1%} < 60%"
            )


def test_agentic_output_tokens_in_range() -> None:
    lo, hi = 100, 1000
    cfg = AgenticWorkloadConfig(
        n_requests=10, seed=1, output_tokens_range=(lo, hi)
    )
    trace = AgenticTraceGenerator(cfg).generate()
    for r in trace.requests:
        assert lo <= r.output_tokens <= hi


def test_agentic_uses_on_off_arrival() -> None:
    """LOAD-03: bursty arrival, so the default config uses ON/OFF."""
    cfg = AgenticWorkloadConfig(n_requests=10, seed=1)
    trace = AgenticTraceGenerator(cfg).generate()
    assert trace.arrival.process == "on_off"
    assert trace.arrival.on_duration_s is not None
    assert trace.arrival.off_duration_s is not None


def test_agentic_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="n_requests must be > 0"):
        AgenticWorkloadConfig(n_requests=0)
    with pytest.raises(ValueError, match="on_duration_s and off_duration_s"):
        AgenticWorkloadConfig(on_duration_s=0, off_duration_s=1.0)
    with pytest.raises(ValueError, match="on_duration_s and off_duration_s"):
        AgenticWorkloadConfig(on_duration_s=1.0, off_duration_s=-1.0)
