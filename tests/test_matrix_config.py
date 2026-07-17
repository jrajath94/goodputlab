"""Tests for bench/schema/matrix_config.py — pilot YAML validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bench.runpod_matrix import MatrixSpec
from bench.schema.matrix_config import load_matrix_config


def _write_yaml(tmp_path: Path, payload: dict[str, object]) -> Path:
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
    assert cfg.topologies == ["colocated"]
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


_REPO_ROOT = Path(__file__).parent.parent


@pytest.mark.parametrize(
    ("name", "expected_cells"),
    [
        ("runpod_matrix.yaml", 2),
        ("runpod_matrix_full.yaml", 72),
        ("runpod_v11.yaml", 54),
        ("runpod_smoke.yaml", 1),
        ("runpod_paired_chat.yaml", 4),
        ("runpod_paired_disagg.yaml", 2),
        ("runpod_context_repair.yaml", 2),
    ],
)
def test_committed_config_validates(name: str, expected_cells: int) -> None:
    """Every committed runpod config must load + enumerate correctly."""
    path = _REPO_ROOT / "configs" / name
    assert path.exists(), f"committed config missing: {name}"
    cfg = load_matrix_config(path)
    assert cfg.to_matrix_spec().total_cells() == expected_cells


def test_no_unregistered_runpod_configs() -> None:
    """New runpod_*.yaml files must be registered in the table above
    (forces a conscious cell-count + cost review before they can run)."""
    registered = {
        "runpod_matrix.yaml",
        "runpod_matrix_full.yaml",
        "runpod_v11.yaml",
        "runpod_smoke.yaml",
        "runpod_paired_chat.yaml",
        "runpod_paired_disagg.yaml",
        "runpod_context_repair.yaml",
    }
    on_disk = {p.name for p in (_REPO_ROOT / "configs").glob("runpod_*.yaml")}
    assert on_disk == registered, (
        f"unregistered configs: {sorted(on_disk - registered)}; "
        f"missing configs: {sorted(registered - on_disk)}"
    )

def test_restrict_topologies_narrows_sweep() -> None:
    """--topologies filter: paired config narrows to one topology."""
    cfg = load_matrix_config(_REPO_ROOT / "configs" / "runpod_paired_disagg.yaml")
    narrowed = cfg.restrict_topologies(["disagg"])
    spec = narrowed.to_matrix_spec()
    assert spec.total_cells() == 1
    assert [t.value for t in spec.topologies] == ["disagg"]
    # Original config untouched (model_copy semantics).
    assert cfg.to_matrix_spec().total_cells() == 2


def test_restrict_topologies_rejects_names_outside_sweep() -> None:
    cfg = load_matrix_config(_REPO_ROOT / "configs" / "runpod_paired_disagg.yaml")
    with pytest.raises(ValueError, match="disagg_tier"):
        cfg.restrict_topologies(["disagg_tier"])


def test_restrict_topologies_rejects_unknown_name() -> None:
    cfg = load_matrix_config(_REPO_ROOT / "configs" / "runpod_paired_disagg.yaml")
    with pytest.raises(ValueError):
        cfg.restrict_topologies(["not-a-topology"])
