"""bench.cell_runner — execute one (topology, model, rate, mix) cell.

A cell fires ``n_warmup + n_measure`` requests through a vLLM client,
collects ``RequestTelemetry`` objects, snapshots GPU thermal, and
emits one ``CellResult`` JSON.  All external deps (client, replay
runner, thermal source) are injected so tests run without GPU.

Idempotent: if the JSON file for the cell already exists, the runner
loads and returns it without re-firing requests.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from bench.schema.cell_schema import (
    CellResult,
    CellSpec,
    Mix,
    ThermalReading,
    Topology,
)
from control.pool import Pool
from core.trace import RequestTelemetry, Trace, WorkloadType
from loadgen.agentic import AgenticTraceGenerator, AgenticWorkloadConfig
from loadgen.chat import ChatTraceGenerator, ChatWorkloadConfig
from loadgen.rag import RagTraceGenerator, RagWorkloadConfig

logger = logging.getLogger(__name__)

# ---------- metrics ----------


@dataclass(frozen=True)
class CellMetrics:
    """Aggregated request metrics for one cell."""

    mean_ttft_ms: float
    p95_ttft_ms: float
    mean_itl_ms: float
    success_rate: float


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    k = (len(s) - 1) * pct
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def aggregate_metrics(telemetries: list[RequestTelemetry]) -> CellMetrics:
    """Compute TTFT/ITL/success aggregates over the telemetries."""
    successes = [t for t in telemetries if t.status_code == 200]
    ttft_values = [t.ttft_ms for t in successes if t.ttft_ms is not None]
    itl_values: list[float] = []
    for t in successes:
        ts = t.per_token_ts_ns
        for i in range(1, len(ts)):
            itl_values.append((ts[i] - ts[i - 1]) / 1_000_000.0)
    return CellMetrics(
        mean_ttft_ms=statistics.mean(ttft_values) if ttft_values else 0.0,
        p95_ttft_ms=_percentile(ttft_values, 0.95),
        mean_itl_ms=statistics.mean(itl_values) if itl_values else 0.0,
        success_rate=(len(successes) / len(telemetries)) if telemetries else 0.0,
    )


# ---------- thermal ----------


class ThermalSource(Protocol):
    """Anything that can read current GPU telemetry."""

    def read(self) -> ThermalReading: ...


class StubThermalSource:
    """Returns a fixed reading — for tests + offline benches."""

    def __init__(self, reading: ThermalReading) -> None:
        self._reading = reading

    def read(self) -> ThermalReading:
        return self._reading


class NvidiaSmiThermalSource:
    """Snapshot GPU thermal via ``nvidia-smi``.

    Used on real GPU pods.  Falls back to zeros if ``nvidia-smi`` is
    missing or the GPU is unreachable.
    """

    def read(self) -> ThermalReading:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu,utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                timeout=2,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return ThermalReading(gpu_temp_c=0, gpu_util_pct=0, gpu_mem_used_mb=0)
        line = out.decode().strip().splitlines()[0]
        temp_s, util_s, mem_s = (p.strip() for p in line.split(","))
        return ThermalReading(
            gpu_temp_c=int(float(temp_s)),
            gpu_util_pct=int(float(util_s)),
            gpu_mem_used_mb=int(float(mem_s)),
        )


# ---------- trace building ----------


def _mix_to_workload(mix: Mix) -> WorkloadType:
    if mix == Mix.CHAT:
        return WorkloadType.CHAT
    if mix == Mix.RAG:
        return WorkloadType.RAG
    if mix == Mix.AGENTIC:
        return WorkloadType.AGENTIC
    raise ValueError(f"unknown mix: {mix!r}")


def _build_trace_for_mix(spec: CellSpec) -> Trace:
    """Generate a workload trace for the cell's mix + seed + size."""
    total = spec.n_warmup + spec.n_measure
    # Translate cell rate (rps) to a duration+rate that yields ~total requests.
    rate_per_sec = float(spec.rate_rps)
    duration_s = max(1.0, total / max(rate_per_sec, 1e-6))
    trace: Trace
    if spec.mix == Mix.CHAT:
        trace = ChatTraceGenerator(
            ChatWorkloadConfig(
                n_requests=total,
                seed=spec.seed,
                rate_per_sec=rate_per_sec,
                duration_s=duration_s,
            )
        ).generate()
    elif spec.mix == Mix.RAG:
        trace = RagTraceGenerator(
            RagWorkloadConfig(
                n_requests=total,
                seed=spec.seed,
                rate_per_sec=rate_per_sec,
                duration_s=duration_s,
            )
        ).generate()
    elif spec.mix == Mix.AGENTIC:
        trace = AgenticTraceGenerator(
            AgenticWorkloadConfig(
                n_requests=total,
                seed=spec.seed,
                rate_per_sec=rate_per_sec,
                duration_s=duration_s,
            )
        ).generate()
    else:
        raise ValueError(f"unknown mix: {spec.mix!r}")
    # Tag the workload type so downstream readers see the right enum.
    return trace.model_copy(update={"workload": _mix_to_workload(spec.mix)})


# ---------- result sink ----------


class CellSink(Protocol):
    """Anything that can write + load a CellResult."""

    def exists(self, cell_id: str) -> bool: ...
    def write(self, result: CellResult) -> Path: ...
    def load(self, cell_id: str) -> CellResult: ...


class JsonCellSink:
    """Atomic JSON sink: ``{out_dir}/{cell_id}.json``."""

    def __init__(self, out_dir: Path) -> None:
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, cell_id: str) -> Path:
        return self._out_dir / f"{cell_id}.json"

    def exists(self, cell_id: str) -> bool:
        return self._path(cell_id).exists()

    def write(self, result: CellResult) -> Path:
        path = self._path(result.cell_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(result.model_dump_json(indent=2))
        tmp.replace(path)
        return path

    def load(self, cell_id: str) -> CellResult:
        return CellResult.model_validate_json(self._path(cell_id).read_text())


def load_cell_result(path: Path) -> CellResult:
    """Load a CellResult from a JSON file path."""
    return CellResult.model_validate_json(Path(path).read_text())


# ---------- runner ----------


ReplayFactory = Callable[[Any], Any]
ClientFactory = Callable[[], Any]


class CellRunner:
    """Execute one cell: trace → client → metrics → thermal → CellResult."""

    def __init__(
        self,
        client_factory: ClientFactory,
        replay_factory: ReplayFactory,
        thermal: ThermalSource,
        sink: CellSink | None = None,
        routed_pool_for: Callable[[Any], str | None] | None = None,
    ) -> None:
        self._client_factory = client_factory
        self._replay_factory = replay_factory
        self._thermal = thermal
        # Optional routing decision — if None, requests carry routed_pool=None
        # (which is fine for COLOCATED cells; DISAGG/DISAGG_TIER need a router).
        self._routed_pool_for = routed_pool_for
        # Default sink writes to /tmp so build_trace / metrics tests
        # don't need to provision a tmpdir just to instantiate.
        self._sink: CellSink = sink if sink is not None else JsonCellSink(
            Path("/tmp/goodputlab_cells")
        )

    # --- public API ---

    def run_cell(self, spec: CellSpec) -> CellResult:
        """Execute the cell.  Idempotent: skip if a valid JSON exists.

        If the existing file is corrupt (partial write from a crashed
        previous run), we re-execute and overwrite so the bench is
        self-healing across crashes.
        """
        if self._sink.exists(spec.cell_id):
            try:
                return self._sink.load(spec.cell_id)
            except Exception:  # noqa: BLE001 — corrupt JSON, self-heal
                logger.warning(
                    "cell %s json corrupt, re-executing", spec.cell_id
                )
        return self._execute(spec)

    def build_trace(self, spec: CellSpec) -> Trace:
        """Public so tests can assert trace shape without firing requests."""
        return _build_trace_for_mix(spec)

    def metrics_from_telemetries(
        self, telemetries: list[RequestTelemetry]
    ) -> CellMetrics:
        """Public so tests can assert metric math without firing requests."""
        return aggregate_metrics(telemetries)

    # --- internals ---

    def _execute(self, spec: CellSpec) -> CellResult:
        client = self._client_factory()
        replay = self._replay_factory(client)
        trace = _build_trace_for_mix(spec)

        started = datetime.now(UTC)
        t0 = time.perf_counter()
        telemetries = asyncio.run(
            replay.replay(trace, routed_pool_for=self._routed_pool_for)
        )
        duration_s = time.perf_counter() - t0

        metrics = aggregate_metrics(telemetries)
        thermal = self._thermal.read()
        # cache_hit_rate: only disagg_tier measures meaningfully; here
        # we expose the proportion of requests that routed through the
        # TIER pool when topology == disagg_tier; else 0.0.
        cache_hit_rate = self._cache_hit_rate(spec, telemetries)

        result = CellResult(
            cell_id=spec.cell_id,
            topology=spec.topology,
            model=spec.model,
            rate_rps=spec.rate_rps,
            mix=spec.mix,
            n_warmup=spec.n_warmup,
            n_measure=spec.n_measure,
            seed=spec.seed,
            mean_ttft_ms=metrics.mean_ttft_ms,
            p95_ttft_ms=metrics.p95_ttft_ms,
            mean_itl_ms=metrics.mean_itl_ms,
            success_rate=metrics.success_rate,
            cache_hit_rate=cache_hit_rate,
            # Reconcile gate: dry-run has no /metrics, so it passes
            # iff success_rate >= 0.99.  Real runpod runner overrides
            # this after scraping vLLM /metrics.
            reconcile_passes=metrics.success_rate >= 0.99,
            thermal=thermal,
            started_at=started,
            duration_s=duration_s,
            notes=[],
        )
        self._sink.write(result)
        return result

    @staticmethod
    def _cache_hit_rate(
        spec: CellSpec, telemetries: list[RequestTelemetry]
    ) -> float:
        if spec.topology != Topology.DISAGG_TIER:
            return 0.0
        if not telemetries:
            return 0.0
        # Pool.TIER.value == "tier" (lowercase).  Match it exactly.
        tier_pool = Pool.TIER.value
        tier = [t for t in telemetries if t.routed_pool == tier_pool]
        return len(tier) / len(telemetries)


__all__ = [
    "CellMetrics",
    "CellRunner",
    "CellSink",
    "ClientFactory",
    "JsonCellSink",
    "NvidiaSmiThermalSource",
    "ReplayFactory",
    "StubThermalSource",
    "ThermalSource",
    "aggregate_metrics",
    "load_cell_result",
]