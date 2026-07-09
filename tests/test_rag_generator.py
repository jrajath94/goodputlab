"""Tests for loadgen/rag.py — RAG trace generator (LOAD-02)."""

from __future__ import annotations

import pytest

from core.trace import WorkloadType
from loadgen.rag import RagTraceGenerator, RagWorkloadConfig


def _pairwise_prefix_overlap_fraction(a: str, b: str) -> float:
    """Return the fraction of ``a``'s length that is byte-identical to the
    prefix of ``b``.  Used to assert the RAG 'shared prefix' property.

    This is a simple measure: longest common prefix of a, b divided by len(a).
    """
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i / max(len(a), 1)
    return n / max(len(a), 1)


def test_rag_trace_has_n_requests() -> None:
    cfg = RagWorkloadConfig(n_requests=20, seed=1, n_corpus_docs=6)
    trace = RagTraceGenerator(cfg).generate()
    assert trace.workload == WorkloadType.RAG
    assert len(trace.requests) == 20


def test_rag_byte_identical() -> None:
    cfg = RagWorkloadConfig(n_requests=8, seed=42, n_corpus_docs=4)
    a = RagTraceGenerator(cfg).generate().model_dump_json()
    b = RagTraceGenerator(cfg).generate().model_dump_json()
    assert a == b


def test_rag_prompt_tokens_at_least_8k_by_default() -> None:
    """LOAD-02 default: prompt_tokens ∈ [8000, 32000]."""
    cfg = RagWorkloadConfig(n_requests=10, seed=1)
    trace = RagTraceGenerator(cfg).generate()
    for r in trace.requests:
        assert r.prompt_tokens >= 8_000, (
            f"request {r.request_id} has {r.prompt_tokens} prompt tokens, expected ≥8000"
        )


def test_rag_prefix_overlap_at_least_75pct() -> None:
    """LOAD-02: ~80% prefix overlap.  Assert pairwise overlap ≥75%.

    We pick 5 random pairs from a 10-request trace and assert the longest
    common prefix (relative to the shorter request) is at least 75% of
    the shorter request's prompt length.
    """
    cfg = RagWorkloadConfig(
        n_requests=10, seed=1, n_corpus_docs=6, include_fraction=0.8
    )
    trace = RagTraceGenerator(cfg).generate()
    prompts = [r.prompt_text for r in trace.requests]
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            a, b = prompts[i], prompts[j]
            shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
            overlap = _pairwise_prefix_overlap_fraction(shorter, longer)
            assert overlap >= 0.75, (
                f"pair ({i},{j}) prefix overlap {overlap:.1%} < 75%"
            )


def test_rag_short_output() -> None:
    """LOAD-02: short output tokens."""
    cfg = RagWorkloadConfig(n_requests=10, seed=1)
    trace = RagTraceGenerator(cfg).generate()
    for r in trace.requests:
        assert r.output_tokens <= 200, f"output {r.output_tokens} > 200 (LOAD-02 short output)"


def test_rag_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="n_requests must be > 0"):
        RagWorkloadConfig(n_requests=0)
    with pytest.raises(ValueError, match="include_fraction"):
        RagWorkloadConfig(include_fraction=0.0)
    with pytest.raises(ValueError, match="include_fraction"):
        RagWorkloadConfig(include_fraction=1.5)
    with pytest.raises(ValueError, match="n_corpus_docs"):
        RagWorkloadConfig(n_corpus_docs=0)


def test_rag_corpus_is_deterministic() -> None:
    """Corpus should be identical across two generators with same seed."""
    cfg = RagWorkloadConfig(n_requests=2, seed=7, n_corpus_docs=4)
    a = RagTraceGenerator(cfg).corpus
    b = RagTraceGenerator(cfg).corpus
    assert a == b
