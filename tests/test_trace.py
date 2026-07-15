"""Tests for core/trace.py — canonical telemetry + trace schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.trace import (
    ArrivalConfig,
    RequestSpec,
    RequestTelemetry,
    SloClass,
    Trace,
    WorkloadType,
)


def _spec(i: int) -> RequestSpec:
    return RequestSpec(
        request_id=f"r{i:04d}",
        slo_class=SloClass.INTERACTIVE,
        workload=WorkloadType.CHAT,
        prompt_tokens=512,
        output_tokens=128,
        prompt_text=f"hello-{i}",
    )


def test_request_spec_round_trip() -> None:
    s = _spec(0)
    dumped = s.model_dump()
    restored = RequestSpec.model_validate(dumped)
    assert restored == s


def test_request_spec_is_frozen() -> None:
    s = _spec(0)
    with pytest.raises(ValidationError):
        s.prompt_tokens = 999


def test_request_spec_rejects_empty_id() -> None:
    with pytest.raises(ValidationError, match="request_id must be non-empty"):
        RequestSpec(
            request_id="   ",
            slo_class=SloClass.INTERACTIVE,
            workload=WorkloadType.CHAT,
            prompt_tokens=10,
            output_tokens=10,
            prompt_text="x",
        )


def test_request_telemetry_round_trip() -> None:
    t = RequestTelemetry(
        request_id="r0001",
        prompt_tokens=512,
        enqueue_ts_ns=1_000_000_000,
        ttft_ms=42.5,
        per_token_ts_ns=[1_010_000_000, 1_020_000_000, 1_030_000_000],
        completion_ts_ns=1_040_000_000,
        status_code=200,
        error=None,
        routed_pool="prefill",
    )
    restored = RequestTelemetry.model_validate(t.model_dump())
    assert restored == t


def test_request_telemetry_per_token_must_be_monotonic() -> None:
    with pytest.raises(ValidationError, match="monotonically non-decreasing"):
        RequestTelemetry(
            request_id="r0001",
            prompt_tokens=512,
            enqueue_ts_ns=100,
            per_token_ts_ns=[200, 150, 300],  # 150 < 200 — out of order
            status_code=200,
        )


def test_request_telemetry_rejects_extra_fields() -> None:
    """Schema is closed: typos in field names fail loudly."""
    with pytest.raises(ValidationError):
        RequestTelemetry.model_validate(
            {
                "request_id": "r1",
                "prompt_tokens": 10,
                "enqueue_ts_ns": 1,
                "ttft_mss": 10,  # typo
                "status_code": 200,
            }
        )


def test_arrival_config_poisson_ok() -> None:
    c = ArrivalConfig(process="poisson", rate_per_sec=10, seed=1)
    assert c.on_duration_s is None and c.off_duration_s is None


def test_arrival_config_on_off_requires_pair() -> None:
    with pytest.raises(ValidationError, match="on_off process requires"):
        ArrivalConfig(
            process="on_off",
            rate_per_sec=10,
            seed=1,
            on_duration_s=1.0,
            # off_duration_s missing
        )


def test_arrival_config_rejects_non_positive_rate() -> None:
    with pytest.raises(ValidationError):
        ArrivalConfig(process="poisson", rate_per_sec=0, seed=1)


def test_trace_n_requests_and_expected() -> None:
    arrival = ArrivalConfig(process="poisson", rate_per_sec=10, seed=1)
    trace = Trace(
        workload=WorkloadType.CHAT,
        seed=1,
        duration_s=5.0,
        arrival=arrival,
        requests=[_spec(i) for i in range(20)],
    )
    assert trace.n_requests() == 20
    assert trace.expected_arrivals() == pytest.approx(50.0)


def test_trace_on_off_expected_arrivals() -> None:
    arrival = ArrivalConfig(
        process="on_off",
        rate_per_sec=20,
        seed=1,
        on_duration_s=1.0,
        off_duration_s=3.0,  # 25% duty cycle
    )
    trace = Trace(
        workload=WorkloadType.RAG,
        seed=1,
        duration_s=10.0,
        arrival=arrival,
        requests=[_spec(i) for i in range(5)],
    )
    # 20 req/s * 0.25 duty * 10s = 50
    assert trace.expected_arrivals() == pytest.approx(50.0)


def test_trace_metadata_is_free_form() -> None:
    arrival = ArrivalConfig(process="poisson", rate_per_sec=1, seed=1)
    trace = Trace(
        workload=WorkloadType.CHAT,
        seed=1,
        duration_s=1.0,
        arrival=arrival,
        requests=[],
        metadata={"gpu": "H100", "driver": "555.42", "anything": "goes"},
    )
    assert trace.metadata["gpu"] == "H100"
