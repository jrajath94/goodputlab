"""bench.schema.cell_schema — Pydantic v2 models for the cell JSON contract.

One ``CellResult`` JSON per (topology, model, rate, mix) cell.  Aggregator
consumes a directory of these to produce ``summary.json``.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Topology(StrEnum):
    COLOCATED = "colocated"
    CHUNKED = "chunked"
    DISAGG = "disagg"
    DISAGG_TIER = "disagg_tier"


class Model(StrEnum):
    QWEN3_1_7B = "qwen3-1.7b"
    QWEN2_5_7B = "qwen2.5-7b"
    QWEN3_30B = "qwen3-30b"


class Mix(StrEnum):
    CHAT = "chat"
    RAG = "rag"
    AGENTIC = "agentic"


def _derive_seed(topology: Topology, model: Model, rate_rps: int, mix: Mix) -> int:
    s = f"{topology.value}|{model.value}|{rate_rps}|{mix.value}"
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h


class CellSpec(BaseModel):
    """Specification of one cell to be measured."""

    model_config = ConfigDict(extra="forbid")

    topology: Topology
    model: Model
    rate_rps: int = Field(ge=1, le=64)
    mix: Mix
    n_warmup: int = Field(default=5, ge=0)
    n_measure: int = Field(default=30, ge=1)
    seed: int = Field(default=0, ge=0)

    @property
    def cell_id(self) -> str:
        return f"{self.topology.value}__{self.model.value}__rate-{self.rate_rps}__{self.mix.value}"

    @model_validator(mode="after")
    def _default_seed(self) -> CellSpec:
        if self.seed == 0:
            object.__setattr__(
                self, "seed", _derive_seed(self.topology, self.model, self.rate_rps, self.mix)
            )
        return self


class ThermalReading(BaseModel):
    """GPU telemetry captured at cell start (via nvidia-smi)."""

    model_config = ConfigDict(extra="forbid")

    gpu_temp_c: int = Field(ge=0, le=120)
    gpu_util_pct: int = Field(ge=0, le=100)
    gpu_mem_used_mb: int = Field(ge=0)

    THERMAL_WARN_C: int = 80

    @property
    def is_overheating(self) -> bool:
        return self.gpu_temp_c > self.THERMAL_WARN_C


class CellResult(BaseModel):
    """One measured cell."""

    model_config = ConfigDict(extra="forbid")

    cell_id: str
    topology: Topology
    model: Model
    rate_rps: int
    mix: Mix
    n_warmup: int
    n_measure: int
    seed: int
    mean_ttft_ms: float
    p95_ttft_ms: float
    mean_itl_ms: float
    success_rate: float = Field(ge=0.0, le=1.0)
    cache_hit_rate: float = Field(ge=0.0, le=1.0)
    reconcile_passes: bool
    thermal: ThermalReading
    started_at: datetime
    duration_s: float = Field(ge=0.0)
    notes: list[str] = Field(default_factory=list)

    @property
    def has_thermal_warning(self) -> bool:
        return self.thermal.is_overheating


class SummaryStats(BaseModel):
    """Aggregate stats across all cells in a campaign.

    Latency means are computed over the **reconciled** subset only.
    Stub cells (``reconcile_passes=False``) carry ``mean_ttft_ms=0``
    because they never produced real telemetry; averaging their zeros
    together with real measurements silently masks performance and is
    the bug fixed in this version. ``n_cells_reconciled`` exposes the
    sample size so the reader knows what the mean is over.
    """

    model_config = ConfigDict(extra="forbid")

    n_cells: int
    n_cells_reconciled: int
    n_unreconciled: int
    n_thermal_warnings: int
    all_reconciled: bool
    mean_ttft_ms: float
    mean_itl_ms: float

    @classmethod
    def from_results(cls, results: list[CellResult]) -> SummaryStats:
        if not results:
            return cls(
                n_cells=0,
                n_cells_reconciled=0,
                n_unreconciled=0,
                n_thermal_warnings=0,
                all_reconciled=True,
                mean_ttft_ms=0.0,
                mean_itl_ms=0.0,
            )
        reconciled = [r for r in results if r.reconcile_passes]
        n_unreconciled = len(results) - len(reconciled)
        n_thermal = sum(1 for r in results if r.has_thermal_warning)
        n_reconciled = len(reconciled)
        # Honest aggregate: over reconciled only. If none reconciled,
        # the means are 0.0 (no real telemetry) rather than crashing
        # on a ZeroDivisionError or averaging zeros with non-zeros.
        if n_reconciled:
            mean_ttft = sum(r.mean_ttft_ms for r in reconciled) / n_reconciled
            mean_itl = sum(r.mean_itl_ms for r in reconciled) / n_reconciled
        else:
            mean_ttft = 0.0
            mean_itl = 0.0
        return cls(
            n_cells=len(results),
            n_cells_reconciled=n_reconciled,
            n_unreconciled=n_unreconciled,
            n_thermal_warnings=n_thermal,
            all_reconciled=n_unreconciled == 0,
            mean_ttft_ms=mean_ttft,
            mean_itl_ms=mean_itl,
        )


class CampaignResult(BaseModel):
    """Top-level campaign rollup."""

    model_config = ConfigDict(extra="forbid")

    n_cells_completed: int
    n_cells_failed: int
    total_duration_s: float
    cost_usd: float
    pod_id: str
    started_at: datetime
    ended_at: datetime


__all__ = [
    "CampaignResult",
    "CellResult",
    "CellSpec",
    "Mix",
    "Model",
    "SummaryStats",
    "ThermalReading",
    "Topology",
]