"""Tests for loadgen/synth_text.py — deterministic synthetic text builder."""

from __future__ import annotations

import random

from loadgen.synth_text import (
    synth_document,
    synth_paragraph,
    synth_query,
    synth_token_count,
)


def test_synth_paragraph_byte_identical() -> None:
    """Same RNG state -> same text."""
    a = random.Random(7)
    b = random.Random(7)
    assert synth_paragraph(a, 100) == synth_paragraph(b, 100)


def test_synth_paragraph_target_within_15pct() -> None:
    """Actual token count within 15% of target."""
    rng = random.Random(0)
    for target in (50, 200, 1000):
        text = synth_paragraph(rng, target)
        n = synth_token_count(text)
        assert abs(n - target) / target <= 0.15, (
            f"target {target} tokens, got {n} ({abs(n - target) / target:.1%} drift)"
        )


def test_synth_paragraph_zero_target_empty() -> None:
    rng = random.Random(1)
    assert synth_paragraph(rng, 0) == ""
    assert synth_paragraph(rng, -5) == ""


def test_synth_document_grows_with_target() -> None:
    """Doubling target roughly doubles length."""
    rng = random.Random(2)
    short = synth_document(rng, 100)
    rng2 = random.Random(2)  # reset
    long = synth_document(rng2, 400)
    assert synth_token_count(long) > synth_token_count(short) * 2


def test_synth_document_has_paragraph_breaks() -> None:
    rng = random.Random(3)
    doc = synth_document(rng, 500)
    assert "\n\n" in doc


def test_synth_query_looks_like_question() -> None:
    rng = random.Random(4)
    q = synth_query(rng, 20)
    assert q.startswith("Question: ")
    assert "answer concisely" in q


def test_synth_token_count_empty() -> None:
    assert synth_token_count("") == 0
    assert synth_token_count("   \n\t  ") == 0


def test_synth_token_count_matches_split() -> None:
    text = "hello world this is a test"
    assert synth_token_count(text) == len(text.split())


def test_synth_different_seeds_different_text() -> None:
    a = synth_paragraph(random.Random(1), 50)
    b = synth_paragraph(random.Random(2), 50)
    assert a != b, "different seeds should produce different text"
