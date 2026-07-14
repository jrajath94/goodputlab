"""Tests for bench/schema/matrix_config.py — pilot YAML validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bench.runpod_matrix import MatrixSpec
from bench.schema.matrix_config import load_matrix_config


def _write_yaml(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "matrix.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def test_load_pilot_yaml_returns_config(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        {
            "topologies": ["colocated"],
            "models": ["qwen2.5-7b"],
            "rates_rps": [4, 8],
            "mixes": ["chat"],
            "cost_per_hour_usd": 1.79,
            "output_dir": "bench/results/pilot",
            "pod_id": "pod-abc",
        },
    )
    cfg = load_matrix_config(path)
    assert cfg.topologies == ["colocated"]  # type: ignore[comparison-overlap]
    assert cfg.rates_rps == [4, 8]
    assert cfg.pod_id == "pod-abc"


def test_to_matrix_spec_yields_expected_size(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        {
            "topologies": ["colocated"],
            "models": ["qwen2.5-7b"],
            "rates_rps": [4, 8],
            "mixes": ["chat"],
        },
    )
    cfg = load_matrix_config(path)
    spec = cfg.to_matrix_spec()
    assert isinstance(spec, MatrixSpec)
    assert len(list(spec.cells())) == 2  # 1 x 1 x 2 x 1


def test_default_topologies_models_rates_when_omitted(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, {"output_dir": "x", "pod_id": "y"})
    cfg = load_matrix_config(path)
    # 4 topologies x 3 models x 6 rates x 3 mixes = 216
    assert len(list(cfg.to_matrix_spec().cells())) == 216


def test_unknown_topology_rejected(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        {"topologies": ["not_a_topo"], "models": ["qwen2.5-7b"]},
    )
    with pytest.raises(Exception):  # ValidationError
        load_matrix_config(path)


def test_negative_rate_rejected(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        {
            "topologies": ["colocated"],
            "models": ["qwen2.5-7b"],
            "rates_rps": [-1],
            "mixes": ["chat"],
        },
    )
    with pytest.raises(Exception):
        load_matrix_config(path)


def test_extra_fields_rejected(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        {
            "topologies": ["colocated"],
            "models": ["qwen2.5-7b"],
            "rates_rps": [4],
            "mixes": ["chat"],
            "bogus_key": "should_fail",
        },
    )
    with pytest.raises(Exception):
        load_matrix_config(path)


def test_default_cost_per_hour(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        {"topologies": ["colocated"], "models": ["qwen2.5-7b"]},
    )
    cfg = load_matrix_config(path)
    assert cfg.cost_per_hour_usd == 1.79  # H100 SXM spot default


def test_empty_yaml_uses_all_defaults(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, {})
    cfg = load_matrix_config(path)
    # 4 x 3 x 6 x 3
    assert len(list(cfg.to_matrix_spec().cells())) == 216
    assert cfg.n_warmup == 5
    assert cfg.n_measure == 30


def test_non_mapping_yaml_rejected(tmp_path: Path) -> None:
    path = tmp_path / "matrix.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_matrix_config(path)


def test_pilot_yaml_loads_without_error() -> None:
    """The committed pilot config must validate end-to-end."""
    repo_root = Path(__file__).parent.parent
    path = repo_root / "configs" / "runpod_matrix.yaml"
    assert path.exists(), "pilot config missing"
    cfg = load_matrix_config(path)
    cells = list(cfg.to_matrix_spec().cells())
    # Pilot: 1 topo x 1 model x 2 rates x 1 mix = 2 cells
    assert len(cells) == 2
    assert cfg.pod_id == "local-pilot"