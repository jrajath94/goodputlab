"""End-to-end smoke test: full pipeline (YAML -> MatrixSpec -> BenchMatrix
-> JSON -> summary) using MockVllmServer.

Proves the entire 216-cell pipeline is wired correctly before burning
GPU on the pilot.  No GPU, no real network — all in-process via httpx
ASGI transport.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from bench.cell_runner import NvidiaSmiThermalSource
from bench.matrix_aggregator import aggregate, write_summary
from bench.mock_vllm import MockVllmServer
from bench.runpod_matrix import BenchMatrix
from bench.schema.matrix_config import load_matrix_config
from loadgen.client import VllmHttpClient
from loadgen.replay import ReplayRunner


def _build_mock_factory(mock: MockVllmServer):
    """ClientFactory that mints a VllmHttpClient targeting the mock app."""

    def factory() -> VllmHttpClient:
        return VllmHttpClient(
            base_url="http://mock-bench",
            model="mock-model",
            max_concurrent=8,
            transport=httpx.ASGITransport(app=mock.app),
        )

    return factory


def _build_replay_factory():
    def factory(client: VllmHttpClient) -> ReplayRunner:
        return ReplayRunner(client)

    return factory


@pytest.fixture
def pilot_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "matrix.yaml"
    path.write_text(
        """\
topologies: [colocated]
models: [qwen2.5-7b]
rates_rps: [2, 4]
mixes: [chat]
n_warmup: 1
n_measure: 3
cost_per_hour_usd: 1.79
output_dir: {out}
pod_id: smoke-test
""".format(out=str(tmp_path / "results"))
    )
    return path


def test_full_pipeline_produces_per_cell_jsons_and_summary(
    tmp_path: Path, pilot_yaml: Path
) -> None:
    cfg = load_matrix_config(pilot_yaml)
    cells_dir = cfg.output_dir
    mock = MockVllmServer()

    matrix = BenchMatrix(
        cells_dir=cells_dir,
        cost_per_hour_usd=cfg.cost_per_hour_usd,
        pod_id=cfg.pod_id,
        client_factory=_build_mock_factory(mock),
        replay_factory=_build_replay_factory(),
        thermal=NvidiaSmiThermalSource(),
        matrix_spec=cfg.to_matrix_spec(),
    )
    report = matrix.run_all()

    # 2 cells from the pilot yaml
    assert report.n_cells_completed == 2
    assert report.n_cells_failed == 0

    json_files = sorted(cells_dir.glob("*.json"))
    assert len(json_files) == 2

    # Validate every JSON round-trips through the aggregator
    results = aggregate(cells_dir)
    assert len(results) == 2
    cell_ids = {r.cell_id for r in results}
    assert cell_ids == {
        "colocated__qwen2.5-7b__rate-2__chat",
        "colocated__qwen2.5-7b__rate-4__chat",
    }

    # write_summary() consumes the same JSONs
    summary_path = write_summary(cells_dir, report, cfg.cost_per_hour_usd)
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["campaign"]["pod_id"] == "smoke-test"
    assert summary["campaign"]["n_cells_completed"] == 2
    assert summary["summary"]["n_cells"] == 2
    assert "colocated" in summary["per_topology"]
    assert summary["cost"]["per_hour_usd"] == 1.79


def test_resume_skips_already_completed_cells(
    tmp_path: Path, pilot_yaml: Path
) -> None:
    cfg = load_matrix_config(pilot_yaml)
    cells_dir = cfg.output_dir
    mock = MockVllmServer()

    # First run — completes both cells
    matrix1 = BenchMatrix(
        cells_dir=cells_dir,
        cost_per_hour_usd=cfg.cost_per_hour_usd,
        pod_id=cfg.pod_id,
        client_factory=_build_mock_factory(mock),
        replay_factory=_build_replay_factory(),
        thermal=NvidiaSmiThermalSource(),
        matrix_spec=cfg.to_matrix_spec(),
    )
    matrix1.run_all()
    assert len(list(cells_dir.glob("*.json"))) == 2

    # Second run — should skip both (already done)
    matrix2 = BenchMatrix(
        cells_dir=cells_dir,
        cost_per_hour_usd=cfg.cost_per_hour_usd,
        pod_id=cfg.pod_id,
        client_factory=_build_mock_factory(mock),
        replay_factory=_build_replay_factory(),
        thermal=NvidiaSmiThermalSource(),
        matrix_spec=cfg.to_matrix_spec(),
    )
    report = matrix2.run_pending()
    assert report.n_cells_completed == 0
    assert report.n_cells_failed == 0
    assert len(list(cells_dir.glob("*.json"))) == 2  # unchanged