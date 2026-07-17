"""bench.preflight — cost + prompt/context preflight for paid matrix runs.

Two gates run before any GPU money is spent:

1. **Cost preflight** — pending cell count, sweep dimensions, request
   counts, estimated wall time and cost at the configured hourly rate.
   Non-smoke runs require explicit approval (``--approve-cost`` or
   ``APPROVE_GPU_SPEND=yes``).

2. **Prompt preflight** — generates the exact traces the cells would
   fire (same generators, same seeds) and checks prompt + output token
   budgets against ``max_model_len``.  The prior RAG/agentic failures
   were context-window overflows discovered *on the pod*; this catches
   them at $0 on the laptop.

Pure computation — no printing, no I/O beyond trace generation — so the
gates are unit-testable without a GPU.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from bench.schema.cell_schema import CellSpec, Mix

# Fixed per-cell overhead: client setup, warmup drain, JSON write, and
# decode tail after the last arrival. Conservative for small cells.
CELL_OVERHEAD_S = 45.0

# Env var that authorizes paid (non-smoke) runs without --approve-cost.
APPROVE_ENV = "APPROVE_GPU_SPEND"

# Cap trace generation per mix so preflighting a 216-cell matrix stays
# fast; prompt shape depends on (mix, seed), not on topology/rate.
MAX_SAMPLED_CELLS_PER_MIX = 6


# ---------- cost preflight ----------


def estimate_cell_seconds(spec: CellSpec) -> float:
    """Wall-time estimate for one cell: arrivals at rate + fixed overhead."""
    total_requests = spec.n_warmup + spec.n_measure
    arrival_window_s = total_requests / max(float(spec.rate_rps), 1e-6)
    return arrival_window_s + CELL_OVERHEAD_S


@dataclass(frozen=True)
class CostPreflight:
    """Everything the operator must see before a paid run starts."""

    n_total_cells: int
    n_pending_cells: int
    n_skipped_cells: int
    topologies: list[str]
    models: list[str]
    rates_rps: list[int]
    mixes: list[str]
    n_warmup: int
    n_measure: int
    est_wall_minutes: float
    cost_per_hour_usd: float
    est_cost_usd: float
    output_dir: str
    smoke: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "n_total_cells": self.n_total_cells,
            "n_pending_cells": self.n_pending_cells,
            "n_skipped_cells": self.n_skipped_cells,
            "topologies": self.topologies,
            "models": self.models,
            "rates_rps": self.rates_rps,
            "mixes": self.mixes,
            "n_warmup": self.n_warmup,
            "n_measure": self.n_measure,
            "est_wall_minutes": round(self.est_wall_minutes, 1),
            "cost_per_hour_usd": self.cost_per_hour_usd,
            "est_cost_usd": round(self.est_cost_usd, 2),
            "output_dir": self.output_dir,
            "smoke": self.smoke,
        }


def build_cost_preflight(
    pending: list[CellSpec],
    n_total_cells: int,
    cost_per_hour_usd: float,
    output_dir: str,
    smoke: bool,
) -> CostPreflight:
    """Estimate wall time + cost for the *pending* subset of the matrix."""
    est_wall_s = sum(estimate_cell_seconds(s) for s in pending)
    est_cost = (est_wall_s / 3600.0) * cost_per_hour_usd

    def _uniq(values: list[str]) -> list[str]:
        seen: dict[str, None] = {}
        for v in values:
            seen.setdefault(v, None)
        return list(seen)

    return CostPreflight(
        n_total_cells=n_total_cells,
        n_pending_cells=len(pending),
        n_skipped_cells=n_total_cells - len(pending),
        topologies=_uniq([s.topology.value for s in pending]),
        models=_uniq([s.model.value for s in pending]),
        rates_rps=sorted({s.rate_rps for s in pending}),
        mixes=_uniq([s.mix.value for s in pending]),
        n_warmup=pending[0].n_warmup if pending else 0,
        n_measure=pending[0].n_measure if pending else 0,
        est_wall_minutes=est_wall_s / 60.0,
        cost_per_hour_usd=cost_per_hour_usd,
        est_cost_usd=est_cost,
        output_dir=output_dir,
        smoke=smoke,
    )


def format_cost_preflight(pf: CostPreflight) -> str:
    run_class = "SMOKE (gate exempt)" if pf.smoke else "PAID (approval required)"
    lines = [
        "[preflight] ---- cost preflight ----",
        f"[preflight] pending cells:     {pf.n_pending_cells} of {pf.n_total_cells}"
        f" ({pf.n_skipped_cells} already on disk, skipped)",
        f"[preflight] topologies:        {pf.topologies}",
        f"[preflight] models:            {pf.models}",
        f"[preflight] rates_rps:         {pf.rates_rps}",
        f"[preflight] mixes:             {pf.mixes}",
        f"[preflight] warmup/measure:    {pf.n_warmup}/{pf.n_measure} per cell",
        f"[preflight] est wall time:     {pf.est_wall_minutes:.1f} min",
        f"[preflight] hourly rate:       ${pf.cost_per_hour_usd:.2f}/hr",
        f"[preflight] est cost:          ${pf.est_cost_usd:.2f}",
        f"[preflight] output dir:        {pf.output_dir}",
        f"[preflight] run class:         {run_class}",
        "[preflight] ------------------------",
    ]
    return "\n".join(lines)


def spend_approved(
    approve_flag: bool, env: Mapping[str, str] | None = None
) -> bool:
    """True if the operator explicitly authorized paid GPU spend."""
    if approve_flag:
        return True
    env = os.environ if env is None else env
    return env.get(APPROVE_ENV, "").strip().lower() == "yes"


# ---------- prompt/context preflight ----------


@dataclass(frozen=True)
class MixPromptStats:
    """Prompt-token distribution for one mix, sampled from real traces."""

    mix: str
    n_requests: int
    min_prompt_tokens: int
    mean_prompt_tokens: float
    p95_prompt_tokens: int
    max_prompt_tokens: int
    max_prompt_plus_output_tokens: int

    def to_dict(self) -> dict[str, object]:
        return {
            "mix": self.mix,
            "n_requests": self.n_requests,
            "min_prompt_tokens": self.min_prompt_tokens,
            "mean_prompt_tokens": round(self.mean_prompt_tokens, 1),
            "p95_prompt_tokens": self.p95_prompt_tokens,
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_prompt_plus_output_tokens": self.max_prompt_plus_output_tokens,
        }


@dataclass(frozen=True)
class PromptPreflight:
    """Overflow verdict across every mix in the pending cell set."""

    max_model_len: int | None
    per_mix: list[MixPromptStats] = field(default_factory=list)

    @property
    def overflow_mixes(self) -> list[str]:
        """Mixes whose worst request cannot fit in max_model_len."""
        if self.max_model_len is None:
            return []
        return [
            s.mix
            for s in self.per_mix
            if s.max_prompt_plus_output_tokens > self.max_model_len
        ]

    @property
    def ok(self) -> bool:
        return not self.overflow_mixes

    def to_dict(self) -> dict[str, object]:
        return {
            "max_model_len": self.max_model_len,
            "per_mix": [s.to_dict() for s in self.per_mix],
            "overflow_mixes": self.overflow_mixes,
            "ok": self.ok,
        }


def _p95(values: list[int]) -> int:
    if not values:
        return 0
    s = sorted(values)
    k = int(round(0.95 * (len(s) - 1)))
    return s[k]


def build_prompt_preflight(
    specs: list[CellSpec], max_model_len: int | None
) -> PromptPreflight:
    """Generate real traces for the pending cells; report token budgets.

    Uses the same generators + seeds the cells will use on the pod, so the
    reported distribution is exact, not a heuristic.  Sampling is capped
    per mix because prompt shape varies with (mix, seed), not with
    topology or rate.
    """
    # Import here so schema/config validation never drags in loadgen.
    from bench.cell_runner import CellRunner, StubThermalSource  # noqa: PLC0415
    from bench.schema.cell_schema import ThermalReading  # noqa: PLC0415

    sampled: dict[Mix, list[CellSpec]] = {}
    for spec in specs:
        bucket = sampled.setdefault(spec.mix, [])
        if len(bucket) < MAX_SAMPLED_CELLS_PER_MIX:
            bucket.append(spec)

    # Stub deps: build_trace never fires requests or reads thermal.
    build_trace = CellRunner(
        client_factory=lambda: None,
        replay_factory=lambda _c: None,
        thermal=StubThermalSource(
            ThermalReading(gpu_temp_c=0, gpu_util_pct=0, gpu_mem_used_mb=0)
        ),
    ).build_trace

    per_mix: list[MixPromptStats] = []
    for mix, mix_specs in sampled.items():
        prompt_tokens: list[int] = []
        prompt_plus_output: list[int] = []
        for spec in mix_specs:
            trace = build_trace(spec)
            for req in trace.requests:
                prompt_tokens.append(req.prompt_tokens)
                prompt_plus_output.append(req.prompt_tokens + req.output_tokens)
        if not prompt_tokens:
            continue
        per_mix.append(
            MixPromptStats(
                mix=mix.value,
                n_requests=len(prompt_tokens),
                min_prompt_tokens=min(prompt_tokens),
                mean_prompt_tokens=sum(prompt_tokens) / len(prompt_tokens),
                p95_prompt_tokens=_p95(prompt_tokens),
                max_prompt_tokens=max(prompt_tokens),
                max_prompt_plus_output_tokens=max(prompt_plus_output),
            )
        )
    return PromptPreflight(max_model_len=max_model_len, per_mix=per_mix)


def format_prompt_preflight(pf: PromptPreflight) -> str:
    lines = ["[preflight] ---- prompt/context preflight ----"]
    if pf.max_model_len is None:
        lines.append(
            "[preflight] max_model_len not set in config — overflow check skipped;"
            " set it to the vLLM --max-model-len value to gate paid runs."
        )
    for s in pf.per_mix:
        lines.append(
            f"[preflight] {s.mix:8s} n={s.n_requests:4d}"
            f" prompt tokens min/mean/p95/max ="
            f" {s.min_prompt_tokens}/{s.mean_prompt_tokens:.0f}"
            f"/{s.p95_prompt_tokens}/{s.max_prompt_tokens}"
            f"  worst prompt+output = {s.max_prompt_plus_output_tokens}"
        )
    if pf.max_model_len is not None:
        if pf.ok:
            lines.append(
                f"[preflight] context check OK: worst request fits in"
                f" max_model_len={pf.max_model_len}"
            )
        else:
            lines.append(
                f"[preflight] CONTEXT OVERFLOW: mixes {pf.overflow_mixes} exceed"
                f" max_model_len={pf.max_model_len}."
                " Fix before paying: raise --max-model-len (check GPU memory"
                " headroom first) or shrink the workload's prompt shape."
            )
    lines.append("[preflight] ------------------------")
    return "\n".join(lines)


__all__ = [
    "APPROVE_ENV",
    "CELL_OVERHEAD_S",
    "CostPreflight",
    "MixPromptStats",
    "PromptPreflight",
    "build_cost_preflight",
    "build_prompt_preflight",
    "estimate_cell_seconds",
    "format_cost_preflight",
    "format_prompt_preflight",
    "spend_approved",
]
