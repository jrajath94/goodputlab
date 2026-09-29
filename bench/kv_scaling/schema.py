"""Pydantic v2 schema for the KV-cache scaling study.

One ``KvScalingCell`` records the KV-cache samples for a single
(seq_len, batch_size) point. A ``KvScalingCampaign`` groups the cells
with the theory prediction, the fitted regression, and the knee point.

Quantity definitions (schema version 1.1):

- ``kv_bytes_used_samples``: per-sample KV-cache bytes **live during
  decode**, from vLLM's block-manager accounting
  (``used_blocks * block_size * theory_bytes_per_token``). These are
  NOT ``torch.cuda.memory_allocated()`` readings: vLLM pre-allocates
  the whole KV pool at engine init, so the allocator counter does not
  move with sequence length.
- ``baseline_bytes``: ``F``, the fixed non-KV overhead (weights,
  workspace, CUDA context), measured once per campaign right after
  engine load as
  ``memory_allocated() - num_gpu_blocks * block_size * theory_bpt``.
- ``total_tokens``: live tokens (prompt + decoded so far) summed over
  the batch at the decode reading.
- ``measured_bytes_per_token`` (computed): ``p50_bytes / total_tokens``
  per cell. It includes block-quantization rounding; the regression
  slope over cells is the headline number.

Raw samples are stored, never just the aggregates, so the regression
can be re-derived from the JSON later. Extra fields are forbidden so
schema drift fails loudly.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile over an ascending-sorted list."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    rank = q * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


class KvScalingCell(BaseModel):
    """One measured (seq_len, batch_size) point.

    ``kv_bytes_used_samples`` holds one KV-bytes-live-during-decode
    value per sample, from block-manager accounting
    (``used_blocks * block_size * theory_bytes_per_token``). The
    optional ``kv_bytes_prefill_samples`` holds the matching
    post-prefill readings; the decode reading must never be smaller.
    ``baseline_bytes`` is ``F``, the fixed non-KV overhead measured once
    per campaign after engine load. p50/p95 are computed from the raw
    decode samples, not supplied.
    """

    model_config = ConfigDict(extra="forbid")

    model_id: str
    gpu: str
    vllm_version: str
    seq_len: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    n_samples: int = Field(gt=0)
    kv_bytes_used_samples: list[int] = Field(min_length=1)
    kv_bytes_prefill_samples: list[int] | None = None
    total_tokens: int = Field(gt=0)
    baseline_bytes: int = Field(ge=0)
    method: str

    @model_validator(mode="after")
    def _samples_match_n_samples(self) -> KvScalingCell:
        if len(self.kv_bytes_used_samples) != self.n_samples:
            raise ValueError(
                f"n_samples={self.n_samples} but got "
                f"{len(self.kv_bytes_used_samples)} decode samples"
            )
        if (
            self.kv_bytes_prefill_samples is not None
            and len(self.kv_bytes_prefill_samples) != self.n_samples
        ):
            raise ValueError(
                f"n_samples={self.n_samples} but got "
                f"{len(self.kv_bytes_prefill_samples)} prefill samples"
            )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def p50_bytes(self) -> float:
        """Median of the raw KV-bytes decode samples."""
        return _percentile(sorted(float(s) for s in self.kv_bytes_used_samples), 0.50)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def p95_bytes(self) -> float:
        """95th percentile of the raw KV-bytes decode samples."""
        return _percentile(sorted(float(s) for s in self.kv_bytes_used_samples), 0.95)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def measured_bytes_per_token(self) -> float:
        """p50 KV bytes per live token for this cell.

        Includes block-quantization rounding; the campaign regression
        slope is the headline number, this column is the per-cell view.
        """
        return self.p50_bytes / self.total_tokens


class RegressionFit(BaseModel):
    """Least-squares fit of per-cell p50 KV bytes vs total_tokens."""

    model_config = ConfigDict(extra="forbid")

    slope_bpt: float = Field(ge=0.0)
    intercept_bytes: float
    r2: float = Field(ge=0.0, le=1.0)
    residual_se: float = Field(ge=0.0)


class KneePoint(BaseModel):
    """Where the KV cache starts to dominate total GPU memory.

    ``total_tokens`` is the smallest measured point where
    ``p50_kv_bytes / (F + p50_kv_bytes) >= 0.5``. ``batch_size`` names
    the curve the knee was found on; None means the knee was computed
    over pooled curves.
    """

    model_config = ConfigDict(extra="forbid")

    total_tokens: int = Field(gt=0)
    kv_share: float = Field(ge=0.0, le=1.0)
    batch_size: int | None = Field(default=None, gt=0)


class KvScalingCampaign(BaseModel):
    """Full campaign: cells plus theory, regression, and knee."""

    model_config = ConfigDict(extra="forbid")

    campaign_id: str
    model_revision: str = ""
    cells: list[KvScalingCell] = Field(min_length=1)
    theory_bytes_per_token: int = Field(gt=0)
    regression: RegressionFit
    knee: KneePoint


__all__ = [
    "KneePoint",
    "KvScalingCampaign",
    "KvScalingCell",
    "RegressionFit",
]
