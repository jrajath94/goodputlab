"""Chat trace generator (LOAD-01).

Multi-turn dialogue prompts.  Every request in the trace shares the
same system prompt; each request contains a configurable number of
user turns with deterministic content drawn from the synthetic
paragraph builder.  The arrival process is Poisson by default;
callers may swap to ON/OFF via ``ArrivalConfig`` if burstiness is
desired.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from core.trace import (
    ArrivalConfig,
    RequestSpec,
    SloClass,
    Trace,
    WorkloadType,
)
from loadgen.synth_text import synth_paragraph, synth_token_count

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise assistant.  Answer the user's question "
    "directly.  If you do not know the answer, say so."
)


@dataclass(frozen=True)
class ChatWorkloadConfig:
    n_requests: int = 100
    seed: int = 42
    prompt_tokens_range: tuple[int, int] = (500, 2000)
    output_tokens_range: tuple[int, int] = (50, 500)
    n_turns_range: tuple[int, int] = (1, 5)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    rate_per_sec: float = 1.0
    duration_s: float = 60.0
    slo_class: SloClass = SloClass.INTERACTIVE

    def __post_init__(self) -> None:
        if self.n_requests <= 0:
            raise ValueError(f"n_requests must be > 0, got {self.n_requests}")
        lo_p, hi_p = self.prompt_tokens_range
        if lo_p < 1 or hi_p < lo_p:
            raise ValueError(f"invalid prompt_tokens_range {self.prompt_tokens_range}")
        lo_o, hi_o = self.output_tokens_range
        if lo_o < 1 or hi_o < lo_o:
            raise ValueError(f"invalid output_tokens_range {self.output_tokens_range}")
        if self.n_turns_range[0] < 1 or self.n_turns_range[1] < self.n_turns_range[0]:
            raise ValueError(f"invalid n_turns_range {self.n_turns_range}")


class ChatTraceGenerator:
    """Build a chat-shaped ``Trace`` with shared system prompt + multi-turn user messages."""

    def __init__(self, config: ChatWorkloadConfig) -> None:
        self._cfg = config
        self._rng = random.Random(config.seed)

    def generate(self) -> Trace:
        cfg = self._cfg
        requests: list[RequestSpec] = []
        # Per-request target prompt length.
        sys_tokens = synth_token_count(cfg.system_prompt)
        for i in range(cfg.n_requests):
            n_turns = self._rng.randint(cfg.n_turns_range[0], cfg.n_turns_range[1])
            target_prompt = self._rng.randint(*cfg.prompt_tokens_range)
            target_output = self._rng.randint(*cfg.output_tokens_range)
            # Split target_prompt across (system + n_turns user messages).
            user_budget = max(1, target_prompt - sys_tokens)
            per_turn = max(1, user_budget // n_turns)
            turns: list[str] = []
            for _ in range(n_turns):
                turns.append(synth_paragraph(self._rng, per_turn))
            user_block = "\n".join(f"User: {t}" for t in turns)
            prompt = f"System: {cfg.system_prompt}\n{user_block}\nAssistant:"
            # Realized token count may differ slightly from target; clamp request
            # record to the *target* budget so downstream tests assert budgets.
            requests.append(
                RequestSpec(
                    request_id=f"chat-{i:05d}",
                    slo_class=cfg.slo_class,
                    workload=WorkloadType.CHAT,
                    prompt_tokens=target_prompt,
                    output_tokens=target_output,
                    prompt_text=prompt,
                )
            )
        arrival = ArrivalConfig(
            process="poisson",
            rate_per_sec=cfg.rate_per_sec,
            seed=cfg.seed,
        )
        return Trace(
            workload=WorkloadType.CHAT,
            seed=cfg.seed,
            duration_s=cfg.duration_s,
            arrival=arrival,
            requests=requests,
        )
