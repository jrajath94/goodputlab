"""Run the GoodputLab matrix sweep on a RunPod H100 pod.

Loads a ``MatrixSweepConfig`` from YAML (see ``configs/runpod_matrix.yaml``),
builds a :class:`BenchMatrix`, drives it through the cell pipeline, and
writes a ``summary.json`` next to the per-cell JSONs.

Usage (staged ladder — see docs/GPU_COST_OPTIMIZATION.md)::

    # Rung 2 — one-cell smoke (gate-exempt, ~$0.50-$0.80):
    python -m scripts.run_matrix --config configs/runpod_smoke.yaml

    # Rung 3 — paired topology probe (needs approval, ~$1-$2):
    python -m scripts.run_matrix --config configs/runpod_paired_chat.yaml --approve-cost

    # Rung 6 — full 216-cell sweep (FINAL POLISH ONLY, never for debugging):
    python -m scripts.run_matrix --config configs/runpod_matrix_full.yaml --approve-cost

Safety defaults: resume (skip valid cell JSONs), stop after the first
unreconciled measured cell, abort on predicted context overflow, and
refuse paid runs without --approve-cost / APPROVE_GPU_SPEND=yes.

Environment::

    RUNPOD_VLLM_BASE_URL  — OpenAI-compatible endpoint, e.g.
                            http://127.0.0.1:8000/v1 (in-cluster) or
                            https://<pod-id>-8000.proxy.runpod.net/v1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

from bench.cell_runner import NvidiaSmiThermalSource
from bench.matrix_aggregator import write_summary
from bench.preflight import (
    APPROVE_ENV,
    build_cost_preflight,
    build_prompt_preflight,
    format_cost_preflight,
    format_prompt_preflight,
    spend_approved,
)
from bench.runpod_matrix import BenchMatrix
from bench.schema.matrix_config import load_matrix_config
from loadgen.client import VllmHttpClient
from loadgen.replay import ReplayRunner

logger = logging.getLogger(__name__)


def _build_client_factory(
    base_url: str, model_name: str, max_concurrent: int
) -> Callable[[], VllmHttpClient]:
    """Return a zero-arg ClientFactory that mints a fresh VllmHttpClient.

    Each cell gets its own client so connection state from the prior cell
    can't bleed into the next.  ``base_url`` should be the OpenAI-compatible
    vLLM endpoint (e.g. ``http://127.0.0.1:8000/v1``).
    """

    def factory() -> VllmHttpClient:
        return VllmHttpClient(
            base_url=base_url,
            model=model_name,
            max_concurrent=max_concurrent,
        )

    return factory


def _build_replay_factory() -> Callable[[VllmHttpClient], ReplayRunner]:
    """Return a replay factory that wraps each fresh client in ReplayRunner."""

    def factory(client: VllmHttpClient) -> ReplayRunner:
        return ReplayRunner(client)

    return factory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GoodputLab matrix sweep")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to MatrixSweepConfig YAML (e.g. configs/runpod_matrix.yaml)",
    )
    parser.add_argument(
        "--model-name",
        default="goodputlab-model",
        help="vLLM model name (must match --served-model-name on the vLLM side)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=64,
        help="Per-client max concurrent requests",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help=(
            "Explicit rerun: re-execute every cell, overwriting existing "
            "JSONs (default: resume — skip cells with a valid JSON on disk)"
        ),
    )
    parser.add_argument(
        "--approve-cost",
        action="store_true",
        help=(
            "Authorize paid GPU spend for non-smoke runs. Alternative: "
            f"export {APPROVE_ENV}=yes. Smoke configs (smoke: true) are exempt."
        ),
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help=(
            "Do not stop after the first unreconciled measured cell "
            "(default: stop immediately — never burn GPU past a broken cell)"
        ),
    )
    parser.add_argument(
        "--allow-overflow",
        action="store_true",
        help=(
            "Proceed even if the prompt preflight detects context-window "
            "overflow vs max_model_len (default: abort before spend)"
        ),
    )
    args = parser.parse_args(argv)

    cfg = load_matrix_config(args.config)
    matrix_spec = cfg.to_matrix_spec()
    total_cells = matrix_spec.total_cells()
    print(f"[run_matrix] sweep size: {total_cells} cells", flush=True)
    print(
        f"[run_matrix] topologies={cfg.topologies or 'ALL'} "
        f"models={cfg.models or 'ALL'} "
        f"rates_rps={cfg.rates_rps or 'ALL'} "
        f"mixes={cfg.mixes or 'ALL'}",
        flush=True,
    )

    base_url = os.environ.get(cfg.vllm_base_url_env)
    if not base_url:
        print(
            f"[run_matrix] ERROR: env {cfg.vllm_base_url_env} not set. "
            f"On RunPod: export RUNPOD_VLLM_BASE_URL=http://127.0.0.1:8000/v1",
            file=sys.stderr,
        )
        return 2

    print(f"[run_matrix] vLLM endpoint: {base_url}", flush=True)
    print(f"[run_matrix] pod_id: {cfg.pod_id}", flush=True)
    print(f"[run_matrix] output_dir: {cfg.output_dir}", flush=True)

    matrix = BenchMatrix(
        cells_dir=cfg.output_dir,
        cost_per_hour_usd=cfg.cost_per_hour_usd,
        pod_id=cfg.pod_id,
        client_factory=_build_client_factory(base_url, args.model_name, args.max_concurrent),
        replay_factory=_build_replay_factory(),
        thermal=NvidiaSmiThermalSource(),
        matrix_spec=matrix_spec,
    )

    # ---- preflight: cost + prompt/context, both before any request ----
    pending = matrix.all_cell_specs() if args.run_all else matrix.pending_cell_specs()
    cost_pf = build_cost_preflight(
        pending=pending,
        n_total_cells=total_cells,
        cost_per_hour_usd=cfg.cost_per_hour_usd,
        output_dir=str(cfg.output_dir),
        smoke=cfg.smoke,
    )
    print(format_cost_preflight(cost_pf), flush=True)
    if args.run_all:
        print(
            "[run_matrix] WARNING: --run-all set; existing cell JSONs in "
            f"{cfg.output_dir} will be overwritten.",
            flush=True,
        )

    if not pending:
        print("[run_matrix] nothing pending; output dir already complete.", flush=True)

    prompt_pf = build_prompt_preflight(pending, cfg.max_model_len)
    print(format_prompt_preflight(prompt_pf), flush=True)
    if not prompt_pf.ok and not args.allow_overflow:
        print(
            "[run_matrix] ERROR: context overflow predicted; aborting before "
            "spend. Re-run with --allow-overflow only if the overflow is "
            "intentional.",
            file=sys.stderr,
        )
        return 6

    # ---- spend gate: non-smoke runs need explicit approval ----
    if pending and not cfg.smoke and not spend_approved(args.approve_cost):
        print(
            f"[run_matrix] ERROR: paid run not approved. Estimated cost "
            f"${cost_pf.est_cost_usd:.2f} for {cost_pf.n_pending_cells} cells. "
            f"Re-run with --approve-cost or export {APPROVE_ENV}=yes.",
            file=sys.stderr,
        )
        return 5

    # Record what the run is about to do — the prompt-length distribution
    # is part of the run's evidence trail (see docs/GPU_COST_OPTIMIZATION.md).
    preflight_path = Path(cfg.output_dir) / "preflight.json"
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_text(
        json.dumps(
            {"cost": cost_pf.to_dict(), "prompt": prompt_pf.to_dict()}, indent=2
        )
    )
    print(f"[run_matrix] preflight recorded: {preflight_path}", flush=True)

    stop_on_unreconciled = not args.keep_going
    report = (
        matrix.run_all(stop_on_unreconciled=stop_on_unreconciled)
        if args.run_all
        else matrix.run_pending(stop_on_unreconciled=stop_on_unreconciled)
    )
    print(
        f"[run_matrix] done: {report.n_cells_completed} completed, "
        f"{report.n_cells_failed} failed, "
        f"duration={report.total_duration_s:.1f}s, "
        f"cost=${report.cost_usd:.4f}",
        flush=True,
    )

    if report.n_cells_completed == 0:
        print(
            "[run_matrix] ERROR: no cells completed; skipping summary.",
            file=sys.stderr,
        )
        return 3  # distinct exit code for "all failed / nothing to aggregate"

    try:
        summary_path = write_summary(cfg.output_dir, report, cfg.cost_per_hour_usd)
    except ValueError as exc:
        # write_summary() raises if no valid CellResult JSONs exist on disk.
        logger.error("summary write failed: %s", exc)
        return 4
    print(f"[run_matrix] summary: {summary_path}", flush=True)
    return 0 if report.n_cells_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())