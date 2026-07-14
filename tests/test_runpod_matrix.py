"""Tests for bench/runpod_matrix.py — orchestrates the 216-cell sweep.

Drives a :class:`MatrixSpec` of (topology × model × rate × mix) through
:class:`bench.cell_runner.CellRunner`, persists one ``CellResult`` JSON per
cell into ``cells_dir``, returns a :class:`CampaignReport`.

Inject all external deps (client, replay, thermal) — tests run without GPU.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from bench.cell_runner import JsonCellSink, StubThermalSource
from bench.runpod_matrix import BenchMatrix, MatrixSpec
from bench.schema.cell_schema import (
    Mix,
    Model,
    ThermalReading,
    Topology,
)
from core.trace import RequestTelemetry, Trace

# ---------- helpers ----------


def _telemetry(status: int = 200, ttft_ms: float = 50.0) -> RequestTelemetry:
    return RequestTelemetry(
        request_id="req-000",
        enqueue_ts_ns=0,
        ttft_ms=ttft_ms,
        per_token_ts_ns=[10_000_000, 16_000_000],
        completion_ts_ns=17_000_000,
        status_code=status,
        routed_pool="PREFILL",
    )


class _FakeReplay:
    """ReplayRunner stand-in.  Sleeps briefly so duration_s > 0."""

    def __init__(self, telemetries: list[RequestTelemetry] | None = None) -> None:
        self._telemetries = telemetries or [_telemetry()]
        self.calls = 0

    async def replay(
        self,
        trace: Trace,
        routed_pool_for: Any = None,
    ) -> list[RequestTelemetry]:
        del routed_pool_for
        self.calls += 1
        return list(self._telemetries)


class _FakeClientFactory:
    def __init__(self) -> None:
        self.instances: list[Any] = []

    def __call__(self) -> Any:
        instance = object()
        self.instances.append(instance)
        return instance


class _FlakyReplay:
    """ReplayRunner that fails after N successful calls (shared counter)."""

    # Shared across instances so the test can fail after N cells total.
    _calls = 0

    def __init__(self, fail_after: int) -> None:
        self._fail_after = fail_after

    async def replay(
        self,
        trace: Trace,
        routed_pool_for: Any = None,
    ) -> list[RequestTelemetry]:
        del routed_pool_for
        type(self)._calls += 1
        if type(self)._calls > self._fail_after:
            raise RuntimeError(f"boom on call {type(self)._calls}")
        return [_telemetry()]

    @classmethod
    def reset(cls) -> None:
        cls._calls = 0


# ---------- matrix spec enumeration ----------


def test_matrix_spec_default_sweep_is_216_cells() -> None:
    spec = MatrixSpec()
    cells = list(spec.cells())
    assert len(cells) == 216
    # Sanity: all 4 topologies, 3 models, 6 rates, 3 mixes present
    assert {c.topology for c in cells} == set(Topology)
    assert {c.model for c in cells} == set(Model)
    assert {c.rate_rps for c in cells} == {1, 2, 4, 8, 16, 32}
    assert {c.mix for c in cells} == set(Mix)


def test_matrix_spec_custom_sweep_size() -> None:
    spec = MatrixSpec(
        topologies=[Topology.COLOCATED],
        models=[Model.QWEN2_5_7B],
        rates_rps=[4],
        mixes=[Mix.CHAT],
    )
    cells = list(spec.cells())
    assert len(cells) == 1
    assert cells[0].cell_id == "colocated__qwen2.5-7b__rate-4__chat"


def test_matrix_spec_cell_ids_are_unique() -> None:
    spec = MatrixSpec()
    cells = list(spec.cells())
    ids = [c.cell_id for c in cells]
    assert len(set(ids)) == 216


def test_matrix_spec_filters_to_subset() -> None:
    """Smaller pilot sweep: 1 topo × 1 model × 2 rates × 1 mix = 2 cells."""
    spec = MatrixSpec(
        topologies=[Topology.COLOCATED, Topology.DISAGG],
        models=[Model.QWEN2_5_7B],
        rates_rps=[2, 8],
        mixes=[Mix.CHAT],
    )
    cells = list(spec.cells())
    assert len(cells) == 4  # 2 topos × 1 model × 2 rates × 1 mix
    assert {c.topology for c in cells} == {Topology.COLOCATED, Topology.DISAGG}


# ---------- cell discovery ----------


def test_bench_matrix_lists_all_cell_specs(tmp_path: Path) -> None:
    matrix = BenchMatrix(
        cells_dir=tmp_path,
        cost_per_hour_usd=1.79,
        pod_id="pod-test",
        client_factory=_FakeClientFactory(),
        replay_factory=lambda _c: _FakeReplay(),
        thermal=StubThermalSource(
            ThermalReading(gpu_temp_c=60, gpu_util_pct=70, gpu_mem_used_mb=40000)
        ),
    )
    assert len(matrix.all_cell_specs()) == 216


def test_bench_matrix_pending_filters_existing(tmp_path: Path) -> None:
    matrix = BenchMatrix(
        cells_dir=tmp_path,
        cost_per_hour_usd=1.79,
        pod_id="pod-test",
        client_factory=_FakeClientFactory(),
        replay_factory=lambda _c: _FakeReplay(),
        thermal=StubThermalSource(
            ThermalReading(gpu_temp_c=60, gpu_util_pct=70, gpu_mem_used_mb=40000)
        ),
    )
    specs = matrix.all_cell_specs()
    # Pretend the first 5 cells were already done.
    for s in specs[:5]:
        sink = JsonCellSink(tmp_path)
        from bench.cell_runner import CellRunner
        CellRunner(
            client_factory=_FakeClientFactory(),
            replay_factory=lambda _c: _FakeReplay([_telemetry()]),
            thermal=StubThermalSource(
                ThermalReading(gpu_temp_c=60, gpu_util_pct=70, gpu_mem_used_mb=40000)
            ),
            sink=sink,
        ).run_cell(s)
    pending = matrix.pending_cell_specs()
    assert len(pending) == 211
    pending_ids = {c.cell_id for c in pending}
    done_ids = {s.cell_id for s in specs[:5]}
    assert done_ids.isdisjoint(pending_ids)


# ---------- run_pending ----------


def test_run_pending_completes_only_unstarted(tmp_path: Path) -> None:
    JsonCellSink(tmp_path)
    thermal = StubThermalSource(
        ThermalReading(gpu_temp_c=60, gpu_util_pct=70, gpu_mem_used_mb=40000)
    )
    # Pre-write 2 cells.
    pre = MatrixSpec(
        topologies=[Topology.COLOCATED],
        models=[Model.QWEN2_5_7B],
        rates_rps=[2],
        mixes=[Mix.CHAT],
    )
    pre_runner = BenchMatrix(
        cells_dir=tmp_path,
        cost_per_hour_usd=1.79,
        pod_id="pod-pre",
        client_factory=_FakeClientFactory(),
        replay_factory=lambda _c: _FakeReplay([_telemetry()]),
        thermal=thermal,
        matrix_spec=pre,
    )
    pre_report = pre_runner.run_pending()
    assert pre_report.n_cells_completed == 1

    # Run a 2-cell sweep via run_pending → second cell should be added.
    sweep = MatrixSpec(
        topologies=[Topology.COLOCATED],
        models=[Model.QWEN2_5_7B],
        rates_rps=[2, 8],
        mixes=[Mix.CHAT],
    )
    sweep_matrix = BenchMatrix(
        cells_dir=tmp_path,
        cost_per_hour_usd=1.79,
        pod_id="pod-sweep",
        client_factory=_FakeClientFactory(),
        replay_factory=lambda _c: _FakeReplay([_telemetry()]),
        thermal=thermal,
        matrix_spec=sweep,
    )
    report = sweep_matrix.run_pending()
    assert report.n_cells_completed == 1  # only the new one
    assert report.n_cells_failed == 0


# ---------- run_all ----------


def test_run_all_completes_all_216_cells(tmp_path: Path) -> None:
    matrix = BenchMatrix(
        cells_dir=tmp_path,
        cost_per_hour_usd=1.79,
        pod_id="pod-216",
        client_factory=_FakeClientFactory(),
        replay_factory=lambda _c: _FakeReplay([_telemetry()]),
        thermal=StubThermalSource(
            ThermalReading(gpu_temp_c=60, gpu_util_pct=70, gpu_mem_used_mb=40000)
        ),
    )
    report = matrix.run_all()
    assert report.n_cells_completed == 216
    assert report.n_cells_failed == 0
    assert len(list(tmp_path.glob("*.json"))) == 216
    assert report.pod_id == "pod-216"


def test_run_all_records_started_and_ended(tmp_path: Path) -> None:
    matrix = BenchMatrix(
        cells_dir=tmp_path,
        cost_per_hour_usd=1.79,
        pod_id="pod-times",
        client_factory=_FakeClientFactory(),
        replay_factory=lambda _c: _FakeReplay([_telemetry()]),
        thermal=StubThermalSource(
            ThermalReading(gpu_temp_c=60, gpu_util_pct=70, gpu_mem_used_mb=40000)
        ),
    )
    report = matrix.run_all()
    assert isinstance(report.started_at, datetime)
    assert isinstance(report.ended_at, datetime)
    assert report.ended_at >= report.started_at


def test_run_all_tracks_cost_from_pod_runtime(tmp_path: Path) -> None:
    """Cost = wall-clock pod duration × hourly_rate."""
    matrix = BenchMatrix(
        cells_dir=tmp_path,
        cost_per_hour_usd=1.79,
        pod_id="pod-cost",
        client_factory=_FakeClientFactory(),
        replay_factory=lambda _c: _FakeReplay([_telemetry()]),
        thermal=StubThermalSource(
            ThermalReading(gpu_temp_c=60, gpu_util_pct=70, gpu_mem_used_mb=40000)
        ),
    )
    report = matrix.run_all()
    expected_max = ((report.ended_at - report.started_at).total_seconds() / 3600.0) * 1.79
    # Cost uses wall-clock, so it should be <= total cell duration + overhead.
    # Fake replay returns instantly so total duration is ~0 and cost ~0.
    assert report.cost_usd >= 0.0
    assert report.cost_usd <= expected_max + 1e-6


def test_run_all_records_total_cell_duration(tmp_path: Path) -> None:
    matrix = BenchMatrix(
        cells_dir=tmp_path,
        cost_per_hour_usd=1.79,
        pod_id="pod-dur",
        client_factory=_FakeClientFactory(),
        replay_factory=lambda _c: _FakeReplay([_telemetry()]),
        thermal=StubThermalSource(
            ThermalReading(gpu_temp_c=60, gpu_util_pct=70, gpu_mem_used_mb=40000)
        ),
    )
    report = matrix.run_all()
    assert report.total_duration_s >= 0.0


def test_run_all_counts_failures(tmp_path: Path) -> None:
    """A cell whose replay raises must increment n_cells_failed."""
    _FlakyReplay.reset()
    spec = MatrixSpec(
        topologies=[Topology.COLOCATED],
        models=[Model.QWEN2_5_7B],
        rates_rps=[2, 8],
        mixes=[Mix.CHAT],
    )
    matrix = BenchMatrix(
        cells_dir=tmp_path,
        cost_per_hour_usd=1.79,
        pod_id="pod-fail",
        client_factory=_FakeClientFactory(),
        # fail_after=1 → first cell OK, second cell raises.
        replay_factory=lambda _c: _FlakyReplay(fail_after=1),
        thermal=StubThermalSource(
            ThermalReading(gpu_temp_c=60, gpu_util_pct=70, gpu_mem_used_mb=40000)
        ),
        matrix_spec=spec,
    )
    report = matrix.run_all()
    assert report.n_cells_completed == 1
    assert report.n_cells_failed == 1


def test_run_all_records_pod_id(tmp_path: Path) -> None:
    matrix = BenchMatrix(
        cells_dir=tmp_path,
        cost_per_hour_usd=1.79,
        pod_id="my-runpod-id-abc123",
        client_factory=_FakeClientFactory(),
        replay_factory=lambda _c: _FakeReplay([_telemetry()]),
        thermal=StubThermalSource(
            ThermalReading(gpu_temp_c=60, gpu_util_pct=70, gpu_mem_used_mb=40000)
        ),
        matrix_spec=MatrixSpec(
            topologies=[Topology.COLOCATED],
            models=[Model.QWEN2_5_7B],
            rates_rps=[4],
            mixes=[Mix.CHAT],
        ),
    )
    report = matrix.run_all()
    assert report.pod_id == "my-runpod-id-abc123"


def test_run_all_creates_pilot_sweep(tmp_path: Path) -> None:
    """Pilot: 1 topo × 1 model × 2 rates × 1 mix = 2 cells."""
    spec = MatrixSpec(
        topologies=[Topology.COLOCATED],
        models=[Model.QWEN2_5_7B],
        rates_rps=[2, 8],
        mixes=[Mix.CHAT],
    )
    matrix = BenchMatrix(
        cells_dir=tmp_path,
        cost_per_hour_usd=1.79,
        pod_id="pod-pilot",
        client_factory=_FakeClientFactory(),
        replay_factory=lambda _c: _FakeReplay([_telemetry()]),
        thermal=StubThermalSource(
            ThermalReading(gpu_temp_c=60, gpu_util_pct=70, gpu_mem_used_mb=40000)
        ),
        matrix_spec=spec,
    )
    report = matrix.run_all()
    assert report.n_cells_completed == 2
    assert len(list(tmp_path.glob("*.json"))) == 2