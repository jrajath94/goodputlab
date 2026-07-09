"""Agentic trace generator (LOAD-03).

Each request in an agentic trace is one tool call in a long agentic
loop.  The shared prefix is the system prompt + tool definitions +
accumulated history.  Earlier requests share less prefix than later
ones (history grows), but the system + tool defs are constant, so
pairwise overlap is still high.  Arrival is ON/OFF by default
(bursty), reflecting how an agent actually issues calls.
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
from loadgen.synth_text import (
    synth_paragraph,
    synth_token_count,
)

DEFAULT_AGENTIC_SYSTEM_PROMPT = (
    "You are an autonomous agent.  You have access to tools.  Think step "
    "by step.  When you have enough information, call the finish tool."
)

DEFAULT_TOOL_DEFS = (
    "Tools:\n"
    "  - search(query: str) -> str: search the corpus for relevant passages\n"
    "  - read(path: str) -> str: read a file by path\n"
    "  - write(path: str, content: str) -> bool: write a file\n"
    "  - finish(answer: str) -> None: end the loop and return the answer\n"
)


@dataclass(frozen=True)
class AgenticWorkloadConfig:
    n_requests: int = 60
    seed: int = 42
    history_tokens_range: tuple[int, int] = (500, 4000)
    observation_tokens_range: tuple[int, int] = (50, 500)
    output_tokens_range: tuple[int, int] = (100, 1000)
    system_prompt: str = DEFAULT_AGENTIC_SYSTEM_PROMPT
    tool_defs: str = DEFAULT_TOOL_DEFS
    on_duration_s: float = 1.5
    off_duration_s: float = 0.3
    rate_per_sec: float = 20.0
    duration_s: float = 60.0
    slo_class: SloClass = SloClass.INTERACTIVE

    def __post_init__(self) -> None:
        if self.n_requests <= 0:
            raise ValueError(f"n_requests must be > 0, got {self.n_requests}")
        if self.on_duration_s <= 0 or self.off_duration_s <= 0:
            raise ValueError("on_duration_s and off_duration_s must be > 0")


class AgenticTraceGenerator:
    """Build an agentic-shaped ``Trace`` with growing history per call."""

    def __init__(self, config: AgenticWorkloadConfig) -> None:
        self._cfg = config
        self._rng = random.Random(config.seed)

    def generate(self) -> Trace:
        cfg = self._cfg
        requests: list[RequestSpec] = []
        history_blocks: list[str] = []  # grows over the loop
        for i in range(cfg.n_requests):
            # New observation (tool result) for this call.
            obs_tokens = self._rng.randint(*cfg.observation_tokens_range)
            observation = synth_paragraph(self._rng, obs_tokens)
            history_blocks.append(
                f"Step {i}: tool_result({i}) = {observation}"
            )
            history_text = "\n".join(history_blocks)
            prompt = (
                f"System: {cfg.system_prompt}\n\n"
                f"{cfg.tool_defs}\n\n"
                f"History:\n{history_text}\n\n"
                f"What is the next action?"
            )
            realized = synth_token_count(prompt)
            target_output = self._rng.randint(*cfg.output_tokens_range)
            requests.append(
                RequestSpec(
                    request_id=f"agent-{i:05d}",
                    slo_class=cfg.slo_class,
                    workload=WorkloadType.AGENTIC,
                    prompt_tokens=realized,
                    output_tokens=target_output,
                    prompt_text=prompt,
                )
            )
        arrival = ArrivalConfig(
            process="on_off",
            rate_per_sec=cfg.rate_per_sec,
            seed=cfg.seed,
            on_duration_s=cfg.on_duration_s,
            off_duration_s=cfg.off_duration_s,
        )
        return Trace(
            workload=WorkloadType.AGENTIC,
            seed=cfg.seed,
            duration_s=cfg.duration_s,
            arrival=arrival,
            requests=requests,
        )
