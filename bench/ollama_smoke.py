"""Ollama smoke harness — prove the loadgen → reconciler pipeline runs locally.

Drives a small chat trace through a local ``ollama serve`` instance and
writes a single per-model JSON to ``bench/results/ollama/<model>.json``.

This is the LOCAL BASELINE path: it does not exercise the vLLM
topologies (colocated / chunked / disagg / disagg_tier) — those require
CUDA and remain measured on RunPod (commit ``c57ee66``).  Ollama is the
single-process baseline that proves the control plane plumbing works
end to end on the operator's laptop.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx

from bench.orchestrator import CampaignReport, Topology
from loadgen.client import VllmHttpClient
from loadgen.replay import ReplayRunner

OLLAMA_TOPOLOGY_LABEL = "ollama"

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_N_REQUESTS = 12
DEFAULT_RATE = 1.5
DEFAULT_PROMPT_TOKENS = 48
DEFAULT_OUTPUT_TOKENS = 24
DEFAULT_OUT = Path("bench/results/ollama")


def _trace(
    n: int,
    rate: float,
    prompt_tokens: int,
    output_tokens: int,
    seed: int,
) -> "Trace":  # type: ignore[name-defined]  # noqa: F821
    from core.trace import (
        ArrivalConfig,
        RequestSpec,
        SloClass,
        Trace,
        WorkloadType,
    )

    text = (
        "Summarize the role of a control plane in a disaggregated LLM "
        "serving stack in two sentences. "
    )
    pad = max(1, prompt_tokens // 16)
    return Trace(
        workload=WorkloadType.CHAT,
        seed=seed,
        duration_s=n / rate,
        arrival=ArrivalConfig(process="poisson", rate_per_sec=rate, seed=seed),
        requests=[
            RequestSpec(
                request_id=f"ollama-{i:04d}",
                slo_class=SloClass.INTERACTIVE,
                workload=WorkloadType.CHAT,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                prompt_text=(text * pad)[: prompt_tokens * 4],
            )
            for i in range(n)
        ],
    )


async def _wait_for_ollama(base_url: str, timeout_s: int = 60) -> bool:
    url = base_url.rstrip("/").rsplit("/v1", 1)[0] + "/api/tags"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(url)
                if r.status_code == 200:
                    return True
        except (httpx.HTTPError, OSError):
            pass
        await asyncio.sleep(1)
    return False


async def _run(args: argparse.Namespace) -> int:
    if not await _wait_for_ollama(args.base_url):
        print(f"[ollama-smoke] ERROR: no Ollama at {args.base_url} in 60s")
        return 1

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ollama-smoke] probing {args.base_url} model={args.model} n={args.n}")
    client = VllmHttpClient(
        base_url=args.base_url,
        model=args.model,
        max_concurrent=8,
        timeout_s=120.0,
    )
    runner = ReplayRunner(client)
    trace = _trace(args.n, args.rate, args.prompt_tokens, args.output_tokens, args.seed)

    telemetries = await runner.replay(trace)

    from bench.orchestrator import _summary

    summary = _summary(telemetries)
    report = CampaignReport(
        topology=Topology.COLOCATED,  # single-process baseline maps to colocated
        n_requests=int(summary["n_requests"]),
        success_rate=summary["success_rate"],
        mean_ttft_ms=summary["mean_ttft_ms"],
        p95_ttft_ms=summary["p95_ttft_ms"],
        mean_itl_ms=summary["mean_itl_ms"],
        cache_hit_rate=summary["cache_hit_rate"],
        reconcile_passes=summary["success_rate"] >= 0.95,
        notes=[
            OLLAMA_TOPOLOGY_LABEL,
            f"model={args.model}",
            f"base_url={args.base_url}",
            f"host=local-m1-max",
            f"platform={os.uname().sysname}",
            f"n={args.n}",
        ],
    )

    safe_model = args.model.replace("/", "_").replace(":", "_")
    out_file = out_dir / f"{safe_model}.json"
    out_file.write_text(report.model_dump_json(indent=2))
    print(f"[ollama-smoke] {safe_model}: success={report.success_rate:.2%} "
          f"mean_ttft={report.mean_ttft_ms:.1f}ms mean_itl={report.mean_itl_ms:.1f}ms "
          f"-> {out_file}")

    summary_blob = {
        "n_reports": 1,
        "host": "local-m1-max",
        "base_url": args.base_url,
        "model": args.model,
        "all_reconcile": report.reconcile_passes,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_blob, indent=2))
    return 0 if report.reconcile_passes else 2


def cli() -> None:
    parser = argparse.ArgumentParser(description="Ollama local smoke bench harness.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n", type=int, default=DEFAULT_N_REQUESTS)
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE)
    parser.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    cli()


__all__ = ["OLLAMA_TOPOLOGY_LABEL", "cli"]
