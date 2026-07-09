"""Tests for core/metrics.py + core/reconcile.py — P10 mitigation gate."""

from __future__ import annotations

import pytest

from core.metrics import (
    parse_prometheus,
    vllm_mean_latency_s,
    vllm_request_counts,
    vllm_token_totals,
)
from core.reconcile import ReconciliationReport, reconcile
from core.trace import RequestTelemetry

# ---------- Prometheus parser ----------


def test_parse_empty_text_returns_empty_dict() -> None:
    assert parse_prometheus("") == {}
    assert parse_prometheus("   \n  \n") == {}


def test_parse_single_counter_line() -> None:
    text = "# HELP vllm:prompt_tokens_total Total prompt tokens.\n"
    text += "# TYPE vllm:prompt_tokens_total counter\n"
    text += "vllm:prompt_tokens_total{model_name=\"llama\"} 1234.0\n"
    parsed = parse_prometheus(text)
    # prometheus_client.parser strips label quotes — key uses bare model name.
    assert parsed["vllm:prompt_tokens_total{model_name=llama}"] == 1234.0


def test_parse_multiple_label_sets_sum_correctly() -> None:
    text = "vllm:prompt_tokens_total{model_name=\"a\"} 100\n"
    text += "vllm:prompt_tokens_total{model_name=\"b\"} 200\n"
    parsed = parse_prometheus(text)
    prompt, _ = vllm_token_totals(parsed)
    assert prompt == 300.0


def test_vllm_token_totals_distinguishes_prompt_vs_generation() -> None:
    parsed = parse_prometheus(
        "vllm:prompt_tokens_total{m=\"x\"} 500\n"
        "vllm:generation_tokens_total{m=\"x\"} 750\n"
    )
    p, g = vllm_token_totals(parsed)
    assert p == 500.0
    assert g == 750.0


def test_vllm_request_counts_falls_back_to_total_when_no_success_metric() -> None:
    parsed = parse_prometheus(
        "vllm:e2e_request_latency_seconds_count 42\n"
    )
    success, total = vllm_request_counts(parsed)
    assert success == 42.0
    assert total == 42.0


def test_vllm_mean_latency_s_computes_ratio() -> None:
    parsed = parse_prometheus(
        "vllm:e2e_request_latency_seconds_sum 12.0\n"
        "vllm:e2e_request_latency_seconds_count 6\n"
    )
    assert vllm_mean_latency_s(parsed) == 2.0


def test_vllm_mean_latency_s_returns_zero_when_no_count() -> None:
    assert vllm_mean_latency_s(parse_prometheus("")) == 0.0


# ---------- Reconciliation ----------


def _tel(
    rid: str = "r",
    status: int = 200,
    ttft_ms: float | None = 50.0,
    n_tokens: int = 10,
) -> RequestTelemetry:
    return RequestTelemetry(
        request_id=rid,
        enqueue_ts_ns=0,
        ttft_ms=ttft_ms,
        per_token_ts_ns=[i + 1 for i in range(n_tokens)],
        completion_ts_ns=n_tokens,
        status_code=status,
        error=None,
        routed_pool="colocated",
    )


def test_reconcile_zero_delta_when_metrics_match() -> None:
    """Synthesize a vLLM body that matches client-side aggregates exactly."""
    client = [_tel(rid=f"r{i}", ttft_ms=50.0, n_tokens=10) for i in range(4)]
    # 4 successes, 40 completion tokens, mean TTFT 50ms = 0.05s
    text = (
        "vllm:request_success_total 4\n"
        "vllm:prompt_tokens_total 0\n"
        "vllm:generation_tokens_total 40\n"
        "vllm:time_to_first_token_seconds_sum 0.2\n"
        "vllm:time_to_first_token_seconds_count 4\n"
    )
    report = reconcile(client, text, window_s=60.0)
    assert report.success_count_delta_pct == 0.0
    assert report.completion_tokens_delta_pct == 0.0
    assert report.mean_ttft_delta_pct == pytest.approx(0.0, abs=0.01)
    assert report.n_client_requests == 4
    assert report.n_server_requests == 4


def test_reconcile_detects_drift_in_completion_tokens() -> None:
    client = [_tel(n_tokens=10) for _ in range(3)]  # 30 client completion tokens
    text = (
        "vllm:request_success_total 3\n"
        "vllm:prompt_tokens_total 0\n"
        "vllm:generation_tokens_total 60\n"  # server says 60, 2x client → 50% delta
        "vllm:time_to_first_token_seconds_sum 0.15\n"
        "vllm:time_to_first_token_seconds_count 3\n"
    )
    report = reconcile(client, text)
    assert report.completion_tokens_delta_pct == pytest.approx(50.0)


def test_reconcile_detects_drift_in_request_count() -> None:
    client = [_tel() for _ in range(5)]
    text = (
        "vllm:request_success_total 4\n"  # server saw only 4
        "vllm:prompt_tokens_total 0\n"
        "vllm:generation_tokens_total 50\n"
        "vllm:time_to_first_token_seconds_sum 0.2\n"
        "vllm:time_to_first_token_seconds_count 4\n"
    )
    report = reconcile(client, text)
    assert report.success_count_delta_pct == pytest.approx(25.0)
    assert report.n_server_requests == 4
    assert report.n_client_requests == 5


def test_reconcile_handles_missing_server_metrics_as_zero() -> None:
    """Empty /metrics body → server saw nothing → large delta."""
    client = [_tel() for _ in range(3)]
    report = reconcile(client, "", window_s=60.0)
    assert report.success_count_delta_pct == 300.0  # 3 client, 0 server
    assert report.n_server_requests == 0


def test_reconcile_gate_passes_under_threshold() -> None:
    """1.5% drift is within tolerance."""
    client = [_tel(ttft_ms=100.0, n_tokens=100) for _ in range(100)]
    text = (
        "vllm:request_success_total 100\n"
        "vllm:prompt_tokens_total 0\n"
        "vllm:generation_tokens_total 10000\n"  # 100×100 client completion tokens
        "vllm:time_to_first_token_seconds_sum 10.15\n"  # 101.5ms mean vs 100ms = 1.5%
        "vllm:time_to_first_token_seconds_count 100\n"
    )
    report = reconcile(client, text)
    assert report.gate_passes(threshold=2.0)
    assert report.gate_passes(threshold=1.5)


def test_reconcile_gate_fails_over_threshold() -> None:
    """3% drift exceeds default gate."""
    client = [_tel(ttft_ms=100.0, n_tokens=100) for _ in range(100)]
    text = (
        "vllm:request_success_total 97\n"  # 3% drift
        "vllm:prompt_tokens_total 0\n"
        "vllm:generation_tokens_total 100\n"
        "vllm:time_to_first_token_seconds_sum 10.0\n"
        "vllm:time_to_first_token_seconds_count 97\n"
    )
    report = reconcile(client, text)
    assert not report.gate_passes(threshold=2.0)


def test_reconcile_window_s_propagates() -> None:
    client = [_tel()]
    report = reconcile(client, "", window_s=42.0)
    assert report.window_s == 42.0


def test_reconcile_rejects_extra_fields() -> None:
    with pytest.raises(Exception):
        ReconciliationReport(
            success_count_delta_pct=0.0,
            prompt_tokens_delta_pct=0.0,
            completion_tokens_delta_pct=0.0,
            mean_ttft_delta_pct=0.0,
            window_s=1.0,
            n_client_requests=0,
            n_server_requests=0,
            surprise_field="nope",
        )  # type: ignore[call-arg]


def test_reconcile_handles_request_failures() -> None:
    """Failed requests don't count as success on either side."""
    client = [
        _tel(rid="ok1", status=200, ttft_ms=50.0, n_tokens=10),
        _tel(rid="bad", status=500, ttft_ms=None, n_tokens=0),
        _tel(rid="ok2", status=200, ttft_ms=60.0, n_tokens=10),
    ]
    text = (
        "vllm:request_success_total 2\n"  # server agrees: 2 success
        "vllm:prompt_tokens_total 0\n"
        "vllm:generation_tokens_total 20\n"
        "vllm:time_to_first_token_seconds_sum 0.11\n"  # 55ms mean (0.11s / 2)
        "vllm:time_to_first_token_seconds_count 2\n"
    )
    report = reconcile(client, text)
    assert report.success_count_delta_pct == 0.0
    assert report.completion_tokens_delta_pct == 0.0
    assert report.mean_ttft_delta_pct == pytest.approx(0.0, abs=0.01)
    assert report.gate_passes(threshold=2.0)