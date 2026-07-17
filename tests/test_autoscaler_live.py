"""Tests for scripts.autoscaler_live — the live workload-shift harness.

Pure parts only (queue-signal parsing, autoscaler wiring, record shape);
the network loop is exercised on the pod, not in CI.
"""

from __future__ import annotations

from scripts.autoscaler_live import (
    DEFAULT_PHASES,
    build_autoscaler,
    queue_signal,
    tick_record,
)

METRICS_BODY = """\
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="goodputlab-model"} 7.0
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="goodputlab-model"} 3.0
"""


def test_queue_signal_extracts_waiting_and_running() -> None:
    assert queue_signal(METRICS_BODY) == (7, 3)


def test_queue_signal_empty_body_reads_zero() -> None:
    assert queue_signal("") == (0, 0)


def test_queue_signal_sums_label_sets() -> None:
    body = METRICS_BODY + (
        'vllm:num_requests_waiting{engine="1",model_name="goodputlab-model"} 2.0\n'
    )
    assert queue_signal(body) == (9, 3)


def test_build_autoscaler_single_colocated_pool() -> None:
    from control.pool import Pool

    scaler = build_autoscaler(min_dwell_s=10.0)
    # One controller, COLOCATED pool, dwell wired through.
    assert set(scaler._controllers.keys()) == {Pool.COLOCATED}
    assert scaler._min_dwell_s == 10.0


def test_tick_record_shape() -> None:
    rec = tick_record(123.456, "prompt_heavy", 7, 3, 2, 1, "queue_high")
    assert rec == {
        "ts": 123.456,
        "phase": "prompt_heavy",
        "queue_waiting": 7,
        "in_flight_running": 3,
        "replicas_virtual": 2,
        "decision_delta": 1,
        "decision_reason": "queue_high",
    }


def test_default_phases_shift_prompt_to_decode() -> None:
    """The experiment must actually shift the workload shape."""
    assert [p.name for p in DEFAULT_PHASES] == ["prompt_heavy", "decode_heavy"]
    prompt_heavy, decode_heavy = DEFAULT_PHASES
    assert prompt_heavy.prompt_words > 10 * decode_heavy.prompt_words
    assert decode_heavy.max_tokens > 10 * prompt_heavy.max_tokens
