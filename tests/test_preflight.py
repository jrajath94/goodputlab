"""Tests for bench/preflight.py — cost + prompt/context gates before GPU spend.

Everything here runs CPU-only: the prompt preflight uses the real trace
generators (same seeds as the pod would), so overflow detection is exact
without renting anything.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.preflight import (
    APPROVE_ENV,
    CELL_OVERHEAD_S,
    build_cost_preflight,
    build_prompt_preflight,
    estimate_cell_seconds,
    format_cost_preflight,
    format_prompt_preflight,
    spend_approved,
)
from bench.schema.cell_schema import CellSpec, Mix, Model, Topology
from bench.schema.matrix_config import load_matrix_config

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def _spec(
    mix: Mix = Mix.CHAT,
    rate: int = 4,
    n_warmup: int = 2,
    n_measure: int = 8,
    topology: Topology = Topology.COLOCATED,
) -> CellSpec:
    return CellSpec(
        topology=topology,
        model=Model.QWEN2_5_7B,
        rate_rps=rate,
        mix=mix,
        n_warmup=n_warmup,
        n_measure=n_measure,
    )


# ---------- cost preflight ----------


def test_estimate_cell_seconds_is_arrivals_plus_overhead() -> None:
    spec = _spec(rate=4, n_warmup=2, n_measure=8)
    assert estimate_cell_seconds(spec) == pytest.approx(10 / 4 + CELL_OVERHEAD_S)


def test_cost_preflight_counts_and_cost_formula() -> None:
    pending = [_spec(rate=4), _spec(rate=8)]
    pf = build_cost_preflight(
        pending=pending,
        n_total_cells=6,
        cost_per_hour_usd=2.0,
        output_dir="bench/results/x",
        smoke=False,
    )
    assert pf.n_pending_cells == 2
    assert pf.n_skipped_cells == 4
    expected_wall_s = sum(estimate_cell_seconds(s) for s in pending)
    assert pf.est_wall_minutes == pytest.approx(expected_wall_s / 60.0)
    assert pf.est_cost_usd == pytest.approx((expected_wall_s / 3600.0) * 2.0)
    assert pf.topologies == ["colocated"]
    assert pf.rates_rps == [4, 8]
    assert pf.smoke is False


def test_cost_preflight_empty_pending() -> None:
    pf = build_cost_preflight(
        pending=[],
        n_total_cells=4,
        cost_per_hour_usd=1.99,
        output_dir="bench/results/x",
        smoke=True,
    )
    assert pf.n_pending_cells == 0
    assert pf.n_skipped_cells == 4
    assert pf.est_cost_usd == 0.0


def test_format_cost_preflight_mentions_every_required_field() -> None:
    pf = build_cost_preflight(
        pending=[_spec()],
        n_total_cells=1,
        cost_per_hour_usd=1.99,
        output_dir="bench/results/runpod_smoke",
        smoke=True,
    )
    text = format_cost_preflight(pf)
    for needle in [
        "pending cells",
        "topologies",
        "models",
        "rates_rps",
        "mixes",
        "warmup/measure",
        "est wall time",
        "hourly rate",
        "est cost",
        "output dir",
        "SMOKE",
    ]:
        assert needle in text


# ---------- spend approval gate ----------


def test_spend_approved_by_flag() -> None:
    assert spend_approved(True, env={}) is True


def test_spend_approved_by_env() -> None:
    assert spend_approved(False, env={APPROVE_ENV: "yes"}) is True
    assert spend_approved(False, env={APPROVE_ENV: "YES"}) is True


def test_spend_not_approved_by_default() -> None:
    assert spend_approved(False, env={}) is False
    assert spend_approved(False, env={APPROVE_ENV: "no"}) is False
    assert spend_approved(False, env={APPROVE_ENV: ""}) is False


# ---------- prompt/context preflight ----------


def test_prompt_preflight_chat_fits_8k() -> None:
    pf = build_prompt_preflight([_spec(mix=Mix.CHAT)], max_model_len=8192)
    assert pf.ok
    assert pf.overflow_mixes == []
    (stats,) = pf.per_mix
    assert stats.mix == "chat"
    assert stats.n_requests == 10
    assert 0 < stats.min_prompt_tokens <= stats.max_prompt_tokens
    assert stats.max_prompt_plus_output_tokens <= 8192


def test_prompt_preflight_detects_rag_overflow_at_16k() -> None:
    """The exact failure mode of the 72-cell reduced sweep: RAG prompts
    are ~18K tokens, so a 16384 context window 400s every request."""
    pf = build_prompt_preflight([_spec(mix=Mix.RAG)], max_model_len=16384)
    assert not pf.ok
    assert pf.overflow_mixes == ["rag"]
    text = format_prompt_preflight(pf)
    assert "CONTEXT OVERFLOW" in text


def test_prompt_preflight_rag_fits_20480() -> None:
    pf = build_prompt_preflight([_spec(mix=Mix.RAG)], max_model_len=20480)
    assert pf.ok


def test_prompt_preflight_no_max_model_len_skips_check() -> None:
    pf = build_prompt_preflight([_spec(mix=Mix.RAG)], max_model_len=None)
    assert pf.ok  # no budget -> no verdict, but distribution still reported
    assert pf.per_mix
    assert "overflow check skipped" in format_prompt_preflight(pf)


def test_prompt_preflight_samples_are_capped_per_mix() -> None:
    specs = [_spec(rate=r, topology=t) for r in (1, 2, 4, 8, 16, 32) for t in Topology]
    pf = build_prompt_preflight(specs, max_model_len=8192)
    (stats,) = pf.per_mix
    # 24 cells but at most MAX_SAMPLED_CELLS_PER_MIX (6) × 10 requests sampled.
    assert stats.n_requests <= 6 * 10


def test_prompt_preflight_to_dict_round_trips() -> None:
    pf = build_prompt_preflight([_spec(mix=Mix.CHAT)], max_model_len=8192)
    d = pf.to_dict()
    assert d["ok"] is True
    assert d["max_model_len"] == 8192
    per_mix = d["per_mix"]
    assert isinstance(per_mix, list)
    assert per_mix[0]["mix"] == "chat"


# ---------- frugal config files on disk ----------


@pytest.mark.parametrize(
    ("name", "expected_cells", "smoke"),
    [
        ("runpod_smoke.yaml", 1, True),
        ("runpod_paired_chat.yaml", 4, False),
        ("runpod_paired_disagg.yaml", 4, False),
        ("runpod_context_repair.yaml", 2, False),
    ],
)
def test_frugal_configs_validate(name: str, expected_cells: int, smoke: bool) -> None:
    cfg = load_matrix_config(CONFIGS_DIR / name)
    assert cfg.to_matrix_spec().total_cells() == expected_cells
    assert cfg.smoke is smoke
    assert cfg.max_model_len is not None


def test_only_smoke_config_is_gate_exempt() -> None:
    """Exactly one config may spend without --approve-cost."""
    smoke_configs = [
        p.name
        for p in sorted(CONFIGS_DIR.glob("runpod_*.yaml"))
        if load_matrix_config(p).smoke
    ]
    assert smoke_configs == ["runpod_smoke.yaml"]


def test_context_repair_budget_clears_rag_worst_case() -> None:
    """The config's max_model_len must actually fit the RAG workload it
    exists to repair — verified against the real generators."""
    cfg = load_matrix_config(CONFIGS_DIR / "runpod_context_repair.yaml")
    specs = list(cfg.to_matrix_spec().cells())
    pf = build_prompt_preflight(specs, cfg.max_model_len)
    assert pf.ok, f"overflow in {pf.overflow_mixes} vs {cfg.max_model_len}"
