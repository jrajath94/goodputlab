"""Client-side vs server-side metric reconciliation (LOAD-06, P10 mitigation).

Single source of truth for the "did the benchmark capture what vLLM saw?"
gate.  ``reconcile(...)`` computes per-metric deltas and returns a
``ReconciliationReport`` whose ``gate_passes()`` decides whether the run
is acceptable as evidence.

Threshold defaults to 2.0% — anything looser hides clock-skew bugs;
anything tighter fails on legitimate jitter.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from core.metrics import (
    parse_prometheus,
    vllm_mean_ttft_s,
    vllm_request_counts,
    vllm_token_totals,
)
from core.trace import RequestTelemetry


class ReconciliationReport(BaseModel):
    """Result of client-vs-server metric comparison.

    ``*_delta_pct`` is ``|client - server| / max(server, 1) * 100``.
    NaN-free; missing server metrics collapse to large positive deltas.
    """

    model_config = ConfigDict(extra="forbid")

    success_count_delta_pct: float
    prompt_tokens_delta_pct: float
    completion_tokens_delta_pct: float
    mean_ttft_delta_pct: float
    window_s: float = Field(gt=0)
    n_client_requests: int = Field(ge=0)
    n_server_requests: int = Field(ge=0)

    def gate_passes(self, threshold: float = 2.0) -> bool:
        """True iff every delta is ``<= threshold``."""
        return all(
            getattr(self, field) <= threshold
            for field in (
                "success_count_delta_pct",
                "prompt_tokens_delta_pct",
                "completion_tokens_delta_pct",
                "mean_ttft_delta_pct",
            )
        )


def _delta_pct(client: float, server: float) -> float:
    """``|client - server| / max(server, 1) * 100``. Never divides by zero."""
    denom = max(abs(server), 1.0)
    return abs(client - server) / denom * 100.0


def reconcile(
    client: list[RequestTelemetry],
    vllm_metrics_text: str,
    window_s: float = 60.0,
) -> ReconciliationReport:
    """Compare client telemetry to parsed vLLM /metrics text.

    ``client`` is the full list of ``RequestTelemetry`` returned by the
    loadgen (every request that left the client, success or fail).
    ``vllm_metrics_text`` is the raw body of vLLM's ``/metrics`` scraped
    once at end-of-run.

    Empty ``vllm_metrics_text`` is treated as "server saw nothing" —
    deltas collapse to 100% (or whatever the client total is).  The
    orchestrator should treat this as a scrape failure, not a pass.
    """
    parsed = parse_prometheus(vllm_metrics_text)

    # Client aggregates.
    client_success = sum(1 for r in client if r.status_code == 200)
    client_prompt_tokens = sum(r.prompt_tokens for r in client)
    client_completion_tokens = sum(len(r.per_token_ts_ns) for r in client)
    client_mean_ttft_ms = _mean_ttft_ms(client)

    # Server aggregates.
    server_prompt, server_completion = vllm_token_totals(parsed)
    server_success, server_total = vllm_request_counts(parsed)
    server_mean_ttft_ms = vllm_mean_ttft_s(parsed) * 1000.0  # s → ms

    return ReconciliationReport(
        success_count_delta_pct=_delta_pct(float(client_success), server_success),
        prompt_tokens_delta_pct=_delta_pct(
            float(client_prompt_tokens), server_prompt
        ),
        completion_tokens_delta_pct=_delta_pct(
            float(client_completion_tokens), server_completion
        ),
        mean_ttft_delta_pct=_delta_pct(client_mean_ttft_ms, server_mean_ttft_ms),
        window_s=window_s,
        n_client_requests=len(client),
        n_server_requests=int(server_total),
    )
def _mean_ttft_ms(client: list[RequestTelemetry]) -> float:
    vals = [r.ttft_ms for r in client if r.ttft_ms is not None]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


__all__ = ["ReconciliationReport", "reconcile"]
