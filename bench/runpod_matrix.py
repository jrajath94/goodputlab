"""bench.runpod_matrix — orchestrate the full 216-cell sweep on a RunPod pod.

Given a :class:`MatrixSpec` (subset of topologies × models × rates × mixes),
drive every cell through :class:`bench.cell_runner.CellRunner`, write one
``CellResult`` JSON per cell into ``cells_dir``, and return a
:class:`CampaignReport` (n_cells_completed/failed, total_duration_s,
cost_usd, pod_id, started_at, ended_at).

Resumable: :meth:`BenchMatrix.run_pending` skips cells that already have
a valid JSON on disk — if the pod dies at cell 87/216, the next invocation
picks up at cell 88 without re-burning the first 87.

Cost model: ``cost_usd = (wall_clock_pod_runtime_s / 3600) × cost_per_hour_usd``.
Sequential cells on one pod → wall-clock ≈ sum of cell durations + overhead.
If the sweep is ever parallelized across pods, swap in wall-clock from the
pod's start/stop, not the per-cell sum.

Pure orchestration: this module does NOT talk to RunPod directly. The
caller (deploy/runpod script) is responsible for ``start_pod`` /
``stop_pod`` and passing in a ``ClientFactory`` that targets the live
vLLM endpoint.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from bench.cell_runner import (
    CellRunner,
    ClientFactory,
    JsonCellSink,
    ReplayFactory,
    ThermalSource,
)
from bench.matrix_aggregator import CampaignReport
from bench.schema.cell_schema import CellResult, CellSpec, Mix, Model, Topology

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatrixSpec:
    """Defines the sweep: which topologies × models × rates × mixes.

    Defaults cover the full 216-cell campaign:
    4 topologies × 3 models × 6 rates × 3 mixes.
    """

    topologies: list[Topology] = field(default_factory=lambda: list(Topology))
    models: list[Model] = field(default_factory=lambda: list(Model))
    rates_rps: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    mixes: list[Mix] = field(default_factory=lambda: list(Mix))
    n_warmup: int = 5
    n_measure: int = 30

    def cells(self) -> Iterator[CellSpec]:
        """Yield one ``CellSpec`` per (topology, model, rate, mix) cell."""
        for topo in self.topologies:
            for model in self.models:
                for rate in self.rates_rps:
                    for mix in self.mixes:
                        yield CellSpec(
                            topology=topo,
                            model=model,
                            rate_rps=rate,
                            mix=mix,
                            n_warmup=self.n_warmup,
                            n_measure=self.n_measure,
                        )

    def total_cells(self) -> int:
        return (
            len(self.topologies)
            * len(self.models)
            * len(self.rates_rps)
            * len(self.mixes)
        )


class BenchMatrix:
    """Drive a :class:`MatrixSpec` through :class:`CellRunner`."""

    def __init__(
        self,
        cells_dir: Path,
        cost_per_hour_usd: float,
        pod_id: str,
        client_factory: ClientFactory,
        replay_factory: ReplayFactory,
        thermal: ThermalSource,
        matrix_spec: MatrixSpec | None = None,
    ) -> None:
        self._cells_dir = Path(cells_dir)
        self._cells_dir.mkdir(parents=True, exist_ok=True)
        self._cost_per_hour = float(cost_per_hour_usd)
        self._pod_id = pod_id
        self._matrix_spec = matrix_spec or MatrixSpec()
        self._runner = CellRunner(
            client_factory=client_factory,
            replay_factory=replay_factory,
            thermal=thermal,
            sink=JsonCellSink(self._cells_dir),
        )

    # --- discovery ---

    def all_cell_specs(self) -> list[CellSpec]:
        return list(self._matrix_spec.cells())

    def pending_cell_specs(self) -> list[CellSpec]:
        """Cells without a valid (parseable + cell_id-matching) JSON on disk.

        Mirrors :meth:`CellRunner.run_cell`'s self-heal contract: a corrupt
        JSON from a crashed previous run is NOT treated as "done" — the
        cell is re-run.
        """
        existing: set[str] = set()
        # Sidecar artifacts the runner itself writes next to cell JSONs.
        sidecars = {"summary.json", "preflight.json"}
        for path in self._cells_dir.glob("*.json"):
            if path.name in sidecars:
                continue
            try:
                result = CellResult.model_validate_json(path.read_text())
            except Exception:  # noqa: BLE001 — corrupt, treat as pending
                logger.warning(
                    "pending: %s corrupt, will re-run", path.name
                )
                continue
            if result.cell_id != path.stem:
                logger.warning(
                    "pending: %s cell_id mismatch (%r), will re-run",
                    path.name,
                    result.cell_id,
                )
                continue
            existing.add(result.cell_id)
        return [c for c in self.all_cell_specs() if c.cell_id not in existing]

    # --- execution ---

    def run_all(self, stop_on_unreconciled: bool = False) -> CampaignReport:
        """Run every cell in the matrix, regardless of existing JSON."""
        return self._run(self.all_cell_specs(), stop_on_unreconciled)

    def run_pending(self, stop_on_unreconciled: bool = False) -> CampaignReport:
        """Resume-safe: skip cells that already have a valid JSON."""
        return self._run(self.pending_cell_specs(), stop_on_unreconciled)

    # --- internals ---

    def _run(
        self, specs: list[CellSpec], stop_on_unreconciled: bool = False
    ) -> CampaignReport:
        started = datetime.now(UTC)
        n_completed = 0
        n_failed = 0
        total_duration_s = 0.0
        for spec in specs:
            try:
                result = self._runner.run_cell(spec)
            except Exception:  # noqa: BLE001 — one cell's failure must not abort the sweep
                logger.exception("cell %s failed", spec.cell_id)
                n_failed += 1
                if stop_on_unreconciled:
                    logger.error(
                        "cell %s raised — stopping sweep (stop_on_unreconciled). "
                        "Fix the failure before re-running; do not burn GPU on repeats.",
                        spec.cell_id,
                    )
                    break
                continue
            n_completed += 1
            total_duration_s += result.duration_s
            if stop_on_unreconciled and not result.reconcile_passes:
                logger.error(
                    "cell %s did not reconcile — stopping sweep "
                    "(stop_on_unreconciled). Result JSON is marked "
                    "reconcile_passes=false; diagnose before spending more.",
                    spec.cell_id,
                )
                break
        ended = datetime.now(UTC)
        wall_s = (ended - started).total_seconds()
        cost_usd = (wall_s / 3600.0) * self._cost_per_hour
        return CampaignReport(
            n_cells_completed=n_completed,
            n_cells_failed=n_failed,
            total_duration_s=total_duration_s,
            cost_usd=cost_usd,
            pod_id=self._pod_id,
            started_at=started,
            ended_at=ended,
        )


__all__ = ["BenchMatrix", "MatrixSpec"]