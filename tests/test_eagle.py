"""Tests for spec/eagle.py — EAGLE-3 simulator + auto-disable + P3 gate."""

from __future__ import annotations

import pytest

from spec.eagle import DraftProposal, SpecDecoder, SpecPolicy, VerifyOutcome

# ---------- Policy ----------


def test_spec_policy_compatible_with_colocated() -> None:
    p = SpecPolicy(topology="colocated")
    assert p.is_topology_compatible()


def test_spec_policy_compatible_with_chunked() -> None:
    p = SpecPolicy(topology="chunked")
    assert p.is_topology_compatible()


def test_p3_gate_disables_spec_in_disagg() -> None:
    """P3 addendum: pure disagg topology → spec decoder disabled at init."""
    p = SpecPolicy(topology="disagg", enabled=True)
    d = SpecDecoder(policy=p, acceptance_rate=0.9)
    assert d.is_enabled is False


def test_p3_gate_disables_spec_in_disagg_tier() -> None:
    p = SpecPolicy(topology="disagg_tier", enabled=True)
    d = SpecDecoder(policy=p)
    assert d.is_enabled is False


def test_p3_gate_allows_spec_in_colocated() -> None:
    p = SpecPolicy(topology="colocated", enabled=True)
    d = SpecDecoder(policy=p)
    assert d.is_enabled is True


def test_spec_policy_rejects_invalid_thresholds() -> None:
    with pytest.raises(Exception):
        SpecPolicy(min_acceptance_rate=1.5)  # type: ignore[call-arg]
    with pytest.raises(Exception):
        SpecPolicy(min_window=0)  # type: ignore[call-arg]


# ---------- Decoder ----------


def test_decoder_returns_none_when_disabled() -> None:
    p = SpecPolicy(enabled=False)
    d = SpecDecoder(policy=p, acceptance_rate=0.9)
    assert d.propose_and_verify() is None


def test_decoder_records_proposed_and_accepted() -> None:
    d = SpecDecoder(
        policy=SpecPolicy(min_acceptance_rate=0.0, min_window=20),
        acceptance_rate=1.0,
        seed=1,
    )
    outcome = d.propose_and_verify(n_draft=5)
    assert outcome is not None
    assert outcome.proposed == 5
    assert outcome.accepted == 5


def test_decoder_computes_acceptance_rate() -> None:
    d = SpecDecoder(
        policy=SpecPolicy(min_acceptance_rate=0.0, min_window=20),
        acceptance_rate=0.5,
        seed=42,
    )
    total_proposed = 0
    total_accepted = 0
    for _ in range(50):
        outcome = d.propose_and_verify(n_draft=10)
        assert outcome is not None
        total_proposed += outcome.proposed
        total_accepted += outcome.accepted
    # Empirical rate should be ~0.5 within tolerance
    rate = total_accepted / total_proposed
    assert 0.35 <= rate <= 0.65, f"expected ~0.5, got {rate:.3f}"


def test_decoder_auto_disables_below_threshold() -> None:
    """With very low acceptance rate, decoder flips to disabled."""
    p = SpecPolicy(min_acceptance_rate=0.5, min_window=10, enabled=True)
    d = SpecDecoder(policy=p, acceptance_rate=0.0, seed=1)
    assert d.is_enabled
    for _ in range(15):  # exceeds min_window
        d.propose_and_verify(n_draft=4)
    # All rejected → mean = 0 → auto-disabled
    assert d.is_enabled is False


def test_decoder_does_not_disable_before_min_window() -> None:
    """Below min_window, even bad rates don't trigger auto-disable."""
    p = SpecPolicy(min_acceptance_rate=0.5, min_window=10, enabled=True)
    d = SpecDecoder(policy=p, acceptance_rate=0.0, seed=1)
    for _ in range(5):  # < min_window
        d.propose_and_verify(n_draft=4)
    assert d.is_enabled is True


def test_decoder_stays_disabled_once_flipped() -> None:
    p = SpecPolicy(min_acceptance_rate=0.5, min_window=5, enabled=True)
    d = SpecDecoder(policy=p, acceptance_rate=0.0, seed=2)
    for _ in range(10):
        d.propose_and_verify(n_draft=3)
    assert d.is_enabled is False
    # Subsequent calls return None even if external state changes.
    assert d.propose_and_verify() is None


def test_decoder_manual_disable_and_enable() -> None:
    d = SpecDecoder(policy=SpecPolicy(min_acceptance_rate=0.0))
    d.disable()
    assert d.is_enabled is False
    assert d.propose_and_verify() is None
    d.enable()
    assert d.is_enabled is True
    assert d.propose_and_verify() is not None


def test_decoder_seed_determinism() -> None:
    """Same seed → same accept sequence."""
    policy = SpecPolicy(min_acceptance_rate=0.0, min_window=100)
    a = SpecDecoder(policy=policy, acceptance_rate=0.5, seed=99)
    b = SpecDecoder(policy=policy, acceptance_rate=0.5, seed=99)
    outcomes_a = [d.propose_and_verify(n_draft=5).accepted for d in [a] for _ in range(20)]  # type: ignore[union-attr]
    outcomes_b = [d.propose_and_verify(n_draft=5).accepted for d in [b] for _ in range(20)]  # type: ignore[union-attr]
    assert outcomes_a == outcomes_b


def test_decoder_total_proposed_accepted_track() -> None:
    d = SpecDecoder(
        policy=SpecPolicy(min_acceptance_rate=0.0, min_window=100),
        acceptance_rate=0.5,
        seed=7,
    )
    for _ in range(5):
        d.propose_and_verify(n_draft=4)
    assert d.total_proposed == 20
    assert 0 <= d.total_accepted <= 20


def test_decoder_zero_draft_returns_zero_outcome() -> None:
    d = SpecDecoder(policy=SpecPolicy(min_acceptance_rate=0.0))
    outcome = d.propose_and_verify(n_draft=0)
    assert outcome is not None
    assert outcome.proposed == 0
    assert outcome.accepted == 0
    assert outcome.acceptance_rate == 0.0


# ---------- Models ----------


def test_draft_proposal_rejects_extra_fields() -> None:
    with pytest.raises(Exception):
        DraftProposal.model_validate(
            {"draft_tokens": ["x"], "proposer": "eagle3", "wat": "no"}
        )


def test_verify_outcome_acceptance_rate_helper() -> None:
    o = VerifyOutcome(accepted_tokens=["a", "b"], proposed=4, accepted=2)
    assert o.acceptance_rate == 0.5


def test_verify_outcome_rejects_extra_fields() -> None:
    with pytest.raises(Exception):
        VerifyOutcome.model_validate(
            {"accepted_tokens": [], "proposed": 0, "accepted": 0, "wat": "no"}
        )