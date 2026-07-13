"""Tests for bench/cell_runner.py — one cell executor.

Run one (topology, model, rate, mix) cell: build trace, fire via client,
collect metrics, snapshot thermal, emit CellResult JSON.

Inject all external deps (vllm client, router, thermal source) so no
GPU required.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from bench.cell_runner import (
    CellRunner,
    JsonCellSink,
    StubThermalSource,
    load_cell_result,
)
from bench.schema.cell_schema import (
    CellSpec,
    Mix,
    Model,
    ThermalReading,
    Topology,
)
from core.trace import (
    RequestTelemetry,
    Trace,
)

# ---------- helpers ----------


def _spec(**overrides: Any) -> CellSpec:
    base: dict[str, Any] = {
        "topology": Topology.COLOCATED,
        "model": Model.QWEN2_5_7B,
        "rate_rps": 4,
        "mix": Mix.CHAT,
        "n_warmup": 2,
        "n_measure": 3,
    }
    base.update(overrides)
    return CellSpec(**base)


def _telemetry(
    status: int = 200,
    ttft_ms: float = 50.0,
    token_ts_ms: tuple[float, ...] = (10.0, 15.0, 21.0),
    routed_pool: str | None = "PREFILL",
    request_id: str = "req-000",
) -> RequestTelemetry:
    # ``token_ts_ms`` are absolute offsets from enqueue (ms) at which each
    # token arrived. Successive deltas = the inter-token latencies the
    # aggregator measures.
    per_token_ts_ns = [int(v * 1_000_000) for v in token_ts_ms]
    completion = per_token_ts_ns[-1] + 1_000_000 if per_token_ts_ns else None
    return RequestTelemetry(
        request_id=request_id,
        enqueue_ts_ns=0,
        ttft_ms=ttft_ms,
        per_token_ts_ns=per_token_ts_ns,
        completion_ts_ns=completion,
        status_code=status,
        routed_pool=routed_pool,
    )


class _FakeReplay:
    """ReplayRunner stand-in. Returns canned telemetries regardless of trace."""

    def __init__(self, telemetries: list[RequestTelemetry]) -> None:
        self._telemetries = telemetries
        self.calls: list[Trace] = []

    async def replay(
        self,
        trace: Trace,
        routed_pool_for: Any = None,
    ) -> list[RequestTelemetry]:
        del routed_pool_for
        self.calls.append(trace)
        return list(self._telemetries)


class _FakeClientFactory:
    """Returns a client object (never invoked by FakeReplay)."""

    def __init__(self) -> None:
        self.instances: list[Any] = []

    def __call__(self) -> Any:
        instance = object()
        self.instances.append(instance)
        return instance


# ---------- trace building ----------


def test_builds_trace_with_n_warmup_plus_n_measure() -> None:
    runner = CellRunner(
        client_factory=_FakeClientFactory(),
        replay_factory=lambda _client: _FakeReplay([]),
        thermal=StubThermalSource(
            ThermalReading(gpu_temp_c=60, gpu_util_pct=70, gpu_mem_used_mb=40000)
        ),
    )
    trace = runner.build_trace(_spec(n_warmup=2, n_measure=3))
    assert len(trace.requests) == 5
    assert trace.requests[0].prompt_text  # non-empty


def test_trace_seed_is_stable_for_same_cell() -> None:
    runner = CellRunner(
        client_factory=_FakeClientFactory(),
        replay_factory=lambda _client: _FakeReplay([]),
        thermal=StubThermalSource(
            ThermalReading(gpu_temp_c=60, gpu_util_pct=70, gpu_mem_used_mb=40000)
        ),
    )
    a = runner.build_trace(_spec())
    b = runner.build_trace(_spec())
    assert a.seed == b.seed


def test_trace_seed_differs_across_cells() -> None:
    runner = CellRunner(
        client_factory=_FakeClientFactory(),
        replay_factory=lambda _client: _FakeReplay([]),
        thermal=StubThermalSource(
            ThermalReading(gpu_temp_c=60, gpu_util_pct=70, gpu_mem_used_mb=40000)
        ),
    )
    a = runner.build_trace(_spec(rate_rps=4))
    b = runner.build_trace(_spec(rate_rps=8))
    assert a.seed != b.seed


# ---------- metrics from telemetries ----------


def test_metrics_aggregate_ttft_and_itl() -> None:
    runner = CellRunner(
        client_factory=_FakeClientFactory(),
        replay_factory=lambda _client: _FakeReplay([]),
        thermal=StubThermalSource(
            ThermalReading(gpu_temp_c=60, gpu_util_pct=70, gpu_mem_used_mb=40000)
        ),
    )
    telemetries = [
        _telemetry(ttft_ms=50.0, token_ts_ms=(10.0, 15.0)),
        _telemetry(ttft_ms=70.0, token_ts_ms=(10.0, 17.0)),
        _telemetry(ttft_ms=100.0, token_ts_ms=(10.0, 19.0)),
    ]
    m = runner.metrics_from_telemetries(telemetries)
    assert m.mean_ttft_ms == pytest.approx(73.333, rel=1e-3)
    assert m.p95_ttft_ms == pytest.approx(97.0, rel=1e-3)
    # Deltas: (15-10)=5, (17-10)=7, (19-10)=9. Mean = 7.0
    assert m.mean_itl_ms == pytest.approx(7.0, rel=1e-3)
    assert m.success_rate == 1.0


def test_metrics_count_failures() -> None:
    runner = CellRunner(
        client_factory=_FakeClientFactory(),
        replay_factory=lambda _client: _FakeReplay([]),
        thermal=StubThermalSource(
            ThermalReading(gpu_temp_c=60, gpu_util_pct=70, gpu_mem_used_mb=40000)
        ),
    )
    telemetries = [
        _telemetry(status=200),
        _telemetry(status=500),
    ]
    m = runner.metrics_from_telemetries(telemetries)
    assert m.success_rate == 0.5


# ---------- end-to-end cell run ----------


def test_run_cell_returns_cell_result(tmp_path: Path) -> None:
    thermal = ThermalReading(gpu_temp_c=65, gpu_util_pct=80, gpu_mem_used_mb=42000)
    telemetries = [
        _telemetry(ttft_ms=80.0, token_ts_ms=(10.0, 15.0, 21.0)),
        _telemetry(ttft_ms=90.0, token_ts_ms=(10.0, 16.0, 23.0)),
        _telemetry(ttft_ms=100.0, token_ts_ms=(10.0, 17.0, 25.0)),
    ]
    runner = CellRunner(
        client_factory=_FakeClientFactory(),
        replay_factory=lambda _client: _FakeReplay(telemetries),
        thermal=StubThermalSource(thermal),
        sink=JsonCellSink(tmp_path),
    )
    result = runner.run_cell(_spec())
    assert result.topology == Topology.COLOCATED
    assert result.mean_ttft_ms == pytest.approx(90.0, rel=1e-3)
    assert result.thermal.gpu_temp_c == 65
    assert result.thermal.gpu_mem_used_mb == 42000
    assert result.n_measure == 3


def test_run_cell_writes_json_file(tmp_path: Path) -> None:
    thermal = ThermalReading(gpu_temp_c=60, gpu_util_pct=70, gpu_mem_used_mb=40000)
    runner = CellRunner(
        client_factory=_FakeClientFactory(),
        replay_factory=lambda _client: _FakeReplay([_telemetry()]),
        thermal=StubThermalSource(thermal),
        sink=JsonCellSink(tmp_path),
    )
    result = runner.run_cell(_spec())
    path = tmp_path / f"{result.cell_id}.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["cell_id"] == result.cell_id
    assert data["topology"] == "colocated"


def test_run_cell_idempotent_skip(tmp_path: Path) -> None:
    thermal = ThermalReading(gpu_temp_c=60, gpu_util_pct=70, gpu_mem_used_mb=40000)
    spec = _spec()
    # First run: writes JSON.
    runner = CellRunner(
        client_factory=_FakeClientFactory(),
        replay_factory=lambda _client: _FakeReplay([_telemetry()]),
        thermal=StubThermalSource(thermal),
        sink=JsonCellSink(tmp_path),
    )
    first = runner.run_cell(spec)
    pre_path = tmp_path / f"{spec.cell_id}.json"
    pre_data = pre_path.read_text()
    # Second run with same cell_id: must skip + return identical result.
    second = runner.run_cell(spec)
    assert first == second
    assert pre_path.read_text() == pre_data


def test_run_cell_self_heals_corrupt_json(tmp_path: Path) -> None:
    """A partial / corrupt JSON file must trigger a re-execute, not a crash."""
    thermal = ThermalReading(gpu_temp_c=60, gpu_util_pct=70, gpu_mem_used_mb=40000)
    spec = _spec()
    pre_path = tmp_path / f"{spec.cell_id}.json"
    pre_path.write_text('{"cell_id": "preexisting_but_corrupt"')
    runner = CellRunner(
        client_factory=_FakeClientFactory(),
        replay_factory=lambda _client: _FakeReplay([_telemetry()]),
        thermal=StubThermalSource(thermal),
        sink=JsonCellSink(tmp_path),
    )
    loaded = runner.run_cell(spec)
    assert loaded.cell_id == spec.cell_id
    assert json.loads(pre_path.read_text())["cell_id"] == spec.cell_id


def test_load_cell_result_round_trip(tmp_path: Path) -> None:
    thermal = ThermalReading(gpu_temp_c=60, gpu_util_pct=70, gpu_mem_used_mb=40000)
    runner = CellRunner(
        client_factory=_FakeClientFactory(),
        replay_factory=lambda _client: _FakeReplay([_telemetry()]),
        thermal=StubThermalSource(thermal),
        sink=JsonCellSink(tmp_path),
    )
    spec = _spec()
    written = runner.run_cell(spec)
    path = tmp_path / f"{spec.cell_id}.json"
    loaded = load_cell_result(path)
    assert loaded == written


def test_run_cell_populates_started_at(tmp_path: Path) -> None:
    thermal = ThermalReading(gpu_temp_c=60, gpu_util_pct=70, gpu_mem_used_mb=40000)
    runner = CellRunner(
        client_factory=_FakeClientFactory(),
        replay_factory=lambda _client: _FakeReplay([_telemetry()]),
        thermal=StubThermalSource(thermal),
        sink=JsonCellSink(tmp_path),
    )
    result = runner.run_cell(_spec())
    assert isinstance(result.started_at, datetime)
    assert result.duration_s >= 0.0


def test_stub_thermal_is_overheating() -> None:
    hot = StubThermalSource(
        ThermalReading(gpu_temp_c=85, gpu_util_pct=99, gpu_mem_used_mb=60000)
    )
    cold = StubThermalSource(
        ThermalReading(gpu_temp_c=60, gpu_util_pct=70, gpu_mem_used_mb=40000)
    )
    assert hot.read().is_overheating is True
    assert cold.read().is_overheating is False