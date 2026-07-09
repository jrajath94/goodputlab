"""Prometheus text-format parser + vLLM-specific extractors.

Parses the standard Prometheus exposition format into a flat
``{metric_name_labels: value}`` map. Used by ``core/reconcile.py``
to compare client-side telemetry against vLLM's ``/metrics`` body.

We hand-roll a focused parser using ``prometheus_client.parser``
(already in deps).  The parser handles label escapes, ``_sum`` /
``_count`` / ``_bucket`` histogram suffixes, and ``# HELP`` /
``# TYPE`` comment lines.
"""

from __future__ import annotations

from collections.abc import Iterable

from prometheus_client.parser import text_string_to_metric_families

# vLLM metric names we care about (P10 mitigation, OBS-02).
# Keep these as module constants so tests don't typo.
VLLM_PROMPT_TOKENS_TOTAL = "vllm:prompt_tokens_total"
VLLM_GENERATION_TOKENS_TOTAL = "vllm:generation_tokens_total"
VLLM_REQUEST_SUCCESS_TOTAL = "vllm:request_success_total"
VLLM_E2E_REQUEST_LATENCY_SECONDS_SUM = "vllm:e2e_request_latency_seconds_sum"
VLLM_E2E_REQUEST_LATENCY_SECONDS_COUNT = "vllm:e2e_request_latency_seconds_count"
VLLM_TTFT_SECONDS_SUM = "vllm:time_to_first_token_seconds_sum"
VLLM_TTFT_SECONDS_COUNT = "vllm:time_to_first_token_seconds_count"


def parse_prometheus(text: str) -> dict[str, float]:
    """Parse Prometheus exposition text into a flat ``{name_labels: float}`` map.

    Histogram series (``foo_bucket``, ``foo_sum``, ``foo_count``) are
    included verbatim — callers that want sum-only or count-only can
    index by suffix.

    Empty / whitespace input returns an empty dict (no exception).
    """
    out: dict[str, float] = {}
    if not text or not text.strip():
        return out
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            # sample.name includes the metric name (e.g. "vllm:prompt_tokens_total")
            # sample.labels is a dict of label values.
            label_key = ",".join(
                f"{k}={v}" for k, v in sorted(sample.labels.items())
            )
            key = f"{sample.name}{{{label_key}}}" if label_key else sample.name
            out[key] = float(sample.value)
    return out


def vllm_token_totals(parsed: dict[str, float]) -> tuple[float, float]:
    """Return ``(prompt_tokens_total, generation_tokens_total)``.

    Sums across all label combinations so the caller compares against
    client-side totals, not per-route per-model slices.
    """
    prompt = _sum_matching(parsed, VLLM_PROMPT_TOKENS_TOTAL)
    generation = _sum_matching(parsed, VLLM_GENERATION_TOKENS_TOTAL)
    return prompt, generation


def vllm_request_counts(parsed: dict[str, float]) -> tuple[float, float]:
    """Return ``(success_count, total_count)``.

    ``total_count`` comes from any of ``vllm:e2e_request_latency_seconds_count``
    or ``vllm:time_to_first_token_seconds_count`` (every finished request,
    success or fail).  ``success_count`` comes from
    ``vllm:request_success_total`` when present; otherwise falls back to
    total count (assumes all succeeded).
    """
    total = _sum_matching(parsed, VLLM_E2E_REQUEST_LATENCY_SECONDS_COUNT)
    if total == 0.0:
        total = _sum_matching(parsed, VLLM_TTFT_SECONDS_COUNT)
    success = _sum_matching(parsed, VLLM_REQUEST_SUCCESS_TOTAL)
    if success == 0.0 and total > 0.0:
        success = total
    return success, total


def vllm_mean_latency_s(parsed: dict[str, float]) -> float:
    """Return mean end-to-end latency in seconds (sum / count).

    Returns 0.0 if either is absent.
    """
    s = _sum_matching(parsed, VLLM_E2E_REQUEST_LATENCY_SECONDS_SUM)
    c = _sum_matching(parsed, VLLM_E2E_REQUEST_LATENCY_SECONDS_COUNT)
    if c <= 0.0:
        return 0.0
    return s / c


def vllm_mean_ttft_s(parsed: dict[str, float]) -> float:
    """Return mean time-to-first-token in seconds (vLLM TTFT metric).

    Uses ``vllm:time_to_first_token_seconds_{sum,count}`` — the
    authoritative server-side TTFT measurement. Returns 0.0 if either
    metric is absent (older vLLM versions may not expose this).
    """
    s = _sum_matching(parsed, VLLM_TTFT_SECONDS_SUM)
    c = _sum_matching(parsed, VLLM_TTFT_SECONDS_COUNT)
    if c <= 0.0:
        return 0.0
    return s / c


def _sum_matching(parsed: dict[str, float], name: str) -> float:
    """Sum all sample values whose key starts with ``name{`` or equals ``name``."""
    total = 0.0
    for key, value in parsed.items():
        if key == name or key.startswith(name + "{"):
            total += value
    return total


def metric_lines(text: str) -> Iterable[str]:
    """Yield non-comment, non-empty lines from Prometheus text (test helper)."""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        yield line


__all__ = [
    "VLLM_E2E_REQUEST_LATENCY_SECONDS_COUNT",
    "VLLM_E2E_REQUEST_LATENCY_SECONDS_SUM",
    "VLLM_GENERATION_TOKENS_TOTAL",
    "VLLM_PROMPT_TOKENS_TOTAL",
    "VLLM_REQUEST_SUCCESS_TOTAL",
    "VLLM_TTFT_SECONDS_COUNT",
    "VLLM_TTFT_SECONDS_SUM",
    "metric_lines",
    "parse_prometheus",
    "vllm_mean_latency_s",
    "vllm_mean_ttft_s",
    "vllm_request_counts",
    "vllm_token_totals",
]