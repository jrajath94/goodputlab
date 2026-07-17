"""bench.schema.matrix_config — Pydantic schema for runpod_matrix.yaml.

Validates the pilot/full matrix config at load time so a typo (e.g.
``rates: 4`` instead of ``rates_rps: [4]``) blows up before GPU spend.

Two top-level shapes are supported:

- ``MatrixSweepConfig`` — sweep subset + load + cost + output.
- ``load_matrix_config(path)`` — read YAML, return validated config.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from bench.runpod_matrix import MatrixSpec
from bench.schema.cell_schema import Mix, Model, Topology


class MatrixSweepConfig(BaseModel):
    """Schema for ``configs/runpod_matrix.yaml``."""

    model_config = ConfigDict(extra="forbid")

    # Sweep subset (each defaults to "all" if omitted).
    topologies: list[Topology] | None = None
    models: list[Model] | None = None
    rates_rps: list[int] | None = Field(default=None)
    mixes: list[Mix] | None = None

    # Per-cell load.
    n_warmup: int = Field(default=5, ge=0)
    n_measure: int = Field(default=30, ge=1)

    # Cost model.
    cost_per_hour_usd: float = Field(default=1.79, gt=0.0)

    # Spend gating: smoke configs (single cheap health cell) skip the
    # --approve-cost / APPROVE_GPU_SPEND gate; everything else requires it.
    smoke: bool = False

    # Context budget for the vLLM server (--max-model-len). When set, the
    # prompt preflight aborts the run if any generated prompt + output
    # budget would overflow it — instead of paying for HTTP 400s.
    max_model_len: int | None = Field(default=None, ge=1)

    # Output.
    output_dir: Path = Field(default=Path("bench/results/runpod_pilot"))
    pod_id: str = Field(default="local-pilot", min_length=1)

    # vLLM endpoint source.
    vllm_base_url_env: str = Field(default="RUNPOD_VLLM_BASE_URL", min_length=1)

    @field_validator("rates_rps")
    @classmethod
    def _positive_rates(cls, v: list[int] | None) -> list[int] | None:
        if v is not None and any(r <= 0 for r in v):
            raise ValueError(f"rates_rps must be positive, got {v}")
        return v

    def to_matrix_spec(self) -> MatrixSpec:
        """Translate to a runnable :class:`MatrixSpec`."""
        return MatrixSpec(
            topologies=self.topologies if self.topologies is not None else list(Topology),
            models=self.models if self.models is not None else list(Model),
            rates_rps=self.rates_rps if self.rates_rps is not None else [1, 2, 4, 8, 16, 32],
            mixes=self.mixes if self.mixes is not None else list(Mix),
            n_warmup=self.n_warmup,
            n_measure=self.n_measure,
        )


def load_matrix_config(path: Path) -> MatrixSweepConfig:
    """Load + validate a ``MatrixSweepConfig`` from YAML."""
    text = Path(path).read_text()
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"matrix config must be a YAML mapping, got {type(raw).__name__}")
    return MatrixSweepConfig.model_validate(raw)


__all__ = ["MatrixSweepConfig", "load_matrix_config"]