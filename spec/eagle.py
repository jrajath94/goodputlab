"""EAGLE-3 speculative decoding — simulator + auto-disable + P3 gate.

Pure Python simulator (no GPU).  Models the *control-plane* behavior:
- propose N draft tokens
- verifier accepts/rejects each independently
- track acceptance rate
- auto-disable when rate drops below ``min_acceptance_rate``
- refuse to operate in pure disagg topologies (P3 addendum)

A real EAGLE-3 integration replaces the simulator; the control loop
is identical.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DraftProposal(BaseModel):
    """A draft model's proposed continuation."""

    model_config = ConfigDict(extra="forbid")

    draft_tokens: list[str]
    proposer: Literal["eagle3", "eagle2", "medusa"] = "eagle3"


class VerifyOutcome(BaseModel):
    """Verifier's accept/reject decision on a draft."""

    model_config = ConfigDict(extra="forbid")

    accepted_tokens: list[str]
    proposed: int = Field(ge=0)
    accepted: int = Field(ge=0)

    @property
    def acceptance_rate(self) -> float:
        if self.proposed == 0:
            return 0.0
        return self.accepted / self.proposed


class SpecPolicy(BaseModel):
    """Auto-disable + P3 topology gate."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    min_acceptance_rate: float = Field(default=0.4, ge=0.0, le=1.0)
    min_window: int = Field(default=20, gt=0)
    topology: Literal["colocated", "chunked", "disagg", "disagg_tier"] = "colocated"

    def is_topology_compatible(self) -> bool:
        """P3 addendum: speculative decoding only helps colocated/chunked.

        Pure disagg (P→D) prefill is already cheap; the draft-verify
        round-trip adds latency without savings.  Disagg + tier is also
        off because the KV transfer overhead dominates.
        """
        return self.topology in ("colocated", "chunked")


class SpecDecoder:
    """Deterministic draft-verify simulator.

    Holds a sliding window of acceptance rates; auto-disables when the
    mean over the window drops below ``policy.min_acceptance_rate``.
    Stays disabled once flipped (no auto-recovery — that's SPEC-03).
    """

    def __init__(
        self,
        policy: SpecPolicy | None = None,
        acceptance_rate: float = 0.8,
        seed: int = 42,
    ) -> None:
        self._policy = policy or SpecPolicy()
        self._target_rate = acceptance_rate
        self._rng = random.Random(seed)
        self._window: deque[int] = deque(maxlen=self._policy.min_window)
        self._total_proposed = 0
        self._total_accepted = 0
        # P3 gate: if topology is incompatible, flip enabled=False at init.
        if not self._policy.is_topology_compatible():
            self._policy.enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._policy.enabled

    @property
    def observed_acceptance_rate(self) -> float:
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    @property
    def total_proposed(self) -> int:
        return self._total_proposed

    @property
    def total_accepted(self) -> int:
        return self._total_accepted

    def enable(self) -> None:
        """Re-enable the decoder (after topology change, e.g.)."""
        self._policy.enabled = True

    def disable(self) -> None:
        """Manual disable (does not reset window)."""
        self._policy.enabled = False

    def propose_and_verify(self, n_draft: int = 5) -> VerifyOutcome | None:
        """Simulate one draft-verify round.

        Returns ``None`` when the decoder is disabled (caller falls back
        to non-speculative decode).  When enabled, returns a
        ``VerifyOutcome`` with per-token accept decisions.
        """
        if not self.is_enabled:
            return None

        accepted: list[str] = []
        for i in range(n_draft):
            token = f"draft-{i}"
            if self._rng.random() < self._target_rate:
                accepted.append(token)
        # Track window + cumulative counters.
        per_round_rate = len(accepted) / n_draft if n_draft > 0 else 0.0
        rate_bin = 1 if per_round_rate >= self._policy.min_acceptance_rate else 0
        self._window.append(rate_bin)
        self._total_proposed += n_draft
        self._total_accepted += len(accepted)

        # Auto-disable when window mean drops below threshold AND window is full.
        if (
            len(self._window) >= self._policy.min_window
            and self.observed_acceptance_rate < self._policy.min_acceptance_rate
        ):
            self.disable()

        return VerifyOutcome(
            accepted_tokens=accepted,
            proposed=n_draft,
            accepted=len(accepted),
        )


__all__ = ["DraftProposal", "SpecDecoder", "SpecPolicy", "VerifyOutcome"]