"""RAG trace generator (LOAD-02).

Long-context prompts with high prefix overlap.  Every request shares
the system prompt + a fixed corpus of "retrieved" documents; a
configurable fraction of the corpus is included per request (default
80%) plus a unique user query.  This shape is what makes cache-aware
routing (Phase 3) worth doing — the prefix-hash on the shared context
should hit the same prefill worker across requests.
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
from loadgen.synth_text import synth_document, synth_query, synth_token_count

DEFAULT_RAG_SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant.  Use the provided documents "
    "to answer the user's question.  If the answer is not in the documents, "
    "say 'I do not know'."
)


@dataclass(frozen=True)
class RagWorkloadConfig:
    n_requests: int = 50
    seed: int = 42
    n_corpus_docs: int = 8
    doc_tokens_range: tuple[int, int] = (1000, 4000)
    include_fraction: float = 0.8  # 80% of corpus per request (LOAD-02 spec)
    query_tokens_range: tuple[int, int] = (50, 200)
    output_tokens_range: tuple[int, int] = (50, 200)
    system_prompt: str = DEFAULT_RAG_SYSTEM_PROMPT
    rate_per_sec: float = 0.5
    duration_s: float = 120.0
    slo_class: SloClass = SloClass.INTERACTIVE

    def __post_init__(self) -> None:
        if self.n_requests <= 0:
            raise ValueError(f"n_requests must be > 0, got {self.n_requests}")
        if not 0.0 < self.include_fraction <= 1.0:
            raise ValueError(f"include_fraction must be in (0, 1], got {self.include_fraction}")
        if self.n_corpus_docs < 1:
            raise ValueError(f"n_corpus_docs must be >= 1, got {self.n_corpus_docs}")
        lo_d, hi_d = self.doc_tokens_range
        if lo_d < 100 or hi_d < lo_d:
            raise ValueError(f"invalid doc_tokens_range {self.doc_tokens_range}")


class RagTraceGenerator:
    """Build a RAG-shaped ``Trace`` with long, high-prefix-overlap prompts."""

    def __init__(self, config: RagWorkloadConfig) -> None:
        self._cfg = config
        self._rng = random.Random(config.seed)
        # Build the corpus once.  Every request draws a (deterministic) subset.
        self._corpus: list[str] = [
            synth_document(self._rng, target_tokens=self._rng.randint(
                config.doc_tokens_range[0], config.doc_tokens_range[1]
            ))
            for _ in range(config.n_corpus_docs)
        ]

    @property
    def corpus(self) -> list[str]:
        return list(self._corpus)

    def generate(self) -> Trace:
        cfg = self._cfg
        requests: list[RequestSpec] = []
        # Stable subset: take the first K documents in corpus order.  The
        # shared prefix is "System: ... + Documents: [corpus[0..K]]" — same
        # across every request, so the prefix hash (Phase 3 router) hits
        # the same prefill worker regardless of which request is routed.
        n_include = max(1, round(cfg.include_fraction * len(self._corpus)))
        stable_corpus = self._corpus[:n_include]
        docs_block = "\n\n---\n\n".join(stable_corpus)
        shared_prefix = (
            f"System: {cfg.system_prompt}\n\nDocuments:\n{docs_block}\n\nUser: "
        )
        for i in range(cfg.n_requests):
            query = synth_query(
                self._rng,
                self._rng.randint(*cfg.query_tokens_range),
            )
            prompt = shared_prefix + query + "\n\nAssistant:"
            realized = synth_token_count(prompt)
            target_output = self._rng.randint(*cfg.output_tokens_range)
            requests.append(
                RequestSpec(
                    request_id=f"rag-{i:05d}",
                    slo_class=cfg.slo_class,
                    workload=WorkloadType.RAG,
                    prompt_tokens=realized,
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
            workload=WorkloadType.RAG,
            seed=cfg.seed,
            duration_s=cfg.duration_s,
            arrival=arrival,
            requests=requests,
        )
