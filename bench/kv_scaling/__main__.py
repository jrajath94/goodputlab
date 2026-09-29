"""CLI entry point for the KV-cache scaling study.

Runbook usage::

    python3 -m bench.kv_scaling \\
      --model Qwen/Qwen2.5-0.5B-Instruct \\
      --revision <pinned-hf-commit-sha> \\
      --seq-lens 1024,2048,4096,8192,16384,32768 \\
      --batches 1,2,4,8,16 \\
      --n-samples 5 \\
      --max-model-len 32768 --enforce-eager \\
      --out bench/results/kv_scaling \\
      --approve-cost

``--dry-run`` runs rung 0 (fake block manager, CPU only, no approval
needed). Without ``--approve-cost`` (or ``APPROVE_GPU_SPEND=yes``) a
multi-cell sweep refuses to start after printing the cost preflight;
single-cell smoke configs are exempt.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _parse_int_list(raw: str) -> list[int]:
    """Parse a comma-separated int list like ``"1024,2048"``."""
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not an int list: {raw!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"empty int list: {raw!r}")
    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError(f"values must be > 0: {raw!r}")
    return values


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (separate for testability)."""
    p = argparse.ArgumentParser(
        prog="bench.kv_scaling",
        description="KV-cache scaling study: block-manager memory sweep.",
    )
    p.add_argument("--model", default=None, help="model id (alias of --model-id)")
    p.add_argument(
        "--model-id",
        default=None,
        help="HF model id (default: $KV_SCALING_MODEL_ID or Qwen/Qwen2.5-0.5B-Instruct)",
    )
    p.add_argument(
        "--revision",
        default=None,
        help="pinned HF commit SHA (default: $KV_SCALING_REVISION or the TRD pin)",
    )
    p.add_argument(
        "--seq-lens",
        type=_parse_int_list,
        default=[512, 2048, 8192],
        help="comma-separated sequence lengths (default: 512,2048,8192)",
    )
    p.add_argument(
        "--batches",
        type=_parse_int_list,
        default=[1, 4, 16],
        help="comma-separated batch sizes (default: 1,4,16)",
    )
    p.add_argument("--n-samples", type=int, default=5, help="samples per cell")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--enforce-eager", dest="enforce_eager", action="store_true", default=True)
    p.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    p.add_argument("--out", type=Path, default=Path("bench/results/kv_scaling"))
    p.add_argument(
        "--approve-cost",
        action="store_true",
        help="approve the GPU spend shown in the preflight",
    )
    p.add_argument("--campaign-id", default="kv-scaling")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="rung 0: fake block manager on CPU, no GPU, no approval needed",
    )
    p.add_argument(
        "--usd-per-hour",
        type=float,
        default=None,
        help="GPU $/hr for the preflight (or $KV_SCALING_USD_PER_HR)",
    )
    p.add_argument(
        "--secs-per-cell",
        type=float,
        default=120.0,
        help="rough planning assumption for preflight wall time",
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--decode-tokens", type=int, default=4)
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI main. Returns the process exit code."""
    from bench.kv_scaling.measure import MODEL_ID, REVISION, run_vllm_sweep

    args = build_parser().parse_args(argv)
    model = args.model_id or args.model or MODEL_ID
    revision = args.revision or REVISION
    usd_per_hour = args.usd_per_hour
    if usd_per_hour is None:
        env_rate = os.environ.get("KV_SCALING_USD_PER_HR")
        if env_rate:
            try:
                usd_per_hour = float(env_rate)
            except ValueError:
                print(
                    f"warning: ignoring invalid KV_SCALING_USD_PER_HR={env_rate!r}",
                    file=sys.stderr,
                )
    try:
        run_vllm_sweep(
            model_id=model,
            revision=revision,
            seq_lens=args.seq_lens,
            batch_sizes=args.batches,
            n_samples=args.n_samples,
            campaign_id=args.campaign_id,
            out_dir=args.out,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            block_size=args.block_size,
            enforce_eager=args.enforce_eager,
            decode_tokens=args.decode_tokens,
            approve_cost=args.approve_cost,
            dry_run=args.dry_run,
            usd_per_hour=usd_per_hour,
            secs_per_cell=args.secs_per_cell,
        )
    except RuntimeError as exc:
        # Cost refusal and missing-CUDA refusal: do not start, exit 2.
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        # Gate failures (theory cross-check, block consistency, rung 0).
        print(f"gate failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
