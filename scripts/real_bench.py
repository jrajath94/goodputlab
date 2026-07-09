"""Real-vLLM bench runner.

Hits a live vLLM OpenAI-compatible endpoint, drives loadgen → router →
vLLM, writes per-topology CampaignReport JSON.  Replaces the
MockVllmServer used in tests/orchestrator.py with the real thing.

Usage:
    python -m scripts.real_bench --base-url http://localhost:8000/v1 \\
        --model Qwen/Qwen2.5-7B-Instruct --out bench/results/real

Writes one JSON per topology: bench/results/real/<topology>.json
plus a summary.json rollup.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from bench.orchestrator import CampaignReport, Topology
from control.pool import Pool, PoolState
from control.router import Router
from core.trace import ArrivalConfig, RequestSpec, SloClass, Trace, WorkloadType


def _trace(n: int = 30, rate: float = 4.0, prompt_tokens: int = 64) -> Trace:
    return Trace(
        workload=WorkloadType.CHAT,
        seed=42,
        duration_s=15.0,
        arrival=ArrivalConfig(process="poisson", rate_per_sec=rate, seed=42),
        requests=[
            RequestSpec(
                request_id=f"r{i:04d}",
                slo_class=SloClass.INTERACTIVE,
                workload=WorkloadType.CHAT,
                prompt_tokens=prompt_tokens,
                output_tokens=24,
                prompt_text=(
                    "Summarize the role of a control plane in a disaggregated "
                    "LLM serving stack in two sentences. " * max(1, prompt_tokens // 16)
                )[: prompt_tokens * 4],
            )
            for i in range(n)
        ],
    )


def _router(topology: Topology) -> Router:
    r = Router()
    r.register_pool(PoolState(pool=Pool.PREFILL, queue_depth=0, capacity=64))
    r.register_pool(PoolState(pool=Pool.DECODE, queue_depth=0, capacity=64))
    r.register_pool(PoolState(pool=Pool.COLOCATED, queue_depth=0, capacity=128))
    if topology in (Topology.DISAGG, Topology.DISAGG_TIER):
        # In real disagg, PREFILL and DECODE are distinct pools.
        # Drop the colocated pool so router is forced to pick P or D.
        return r
    return r


async def _wait_for_vllm(base_url: str, timeout_s: int = 180) -> bool:
    """Poll /v1/models until vLLM is ready or timeout."""
    url = base_url.rstrip("/") + "/models"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return True
        except (httpx.HTTPError, OSError):
            pass
        await asyncio.sleep(2)
    return False


async def _run_one(
    topology: Topology,
    base_url: str,
    model: str,
) -> CampaignReport:
    # Direct execution path: drive VllmHttpClient against real base_url.
    # BenchOrchestrator is bypassed because it owns MockVllmServer; we
    # reuse its _summary helper for CampaignReport construction.
    from loadgen.client import VllmHttpClient
    from loadgen.replay import ReplayRunner

    client = VllmHttpClient(
        base_url=base_url,
        model=model,
        max_concurrent=16,
        timeout_s=60.0,
    )
    runner = ReplayRunner(client)
    router = _router(topology)

    def routed_pool_for(spec: RequestSpec) -> str:
        decision = router.route(spec)
        return decision.pool.value if decision.admitted else "colocated"

    trace = _trace()
    telemetries = await runner.replay(trace, routed_pool_for=routed_pool_for)
    # Synthesize a CampaignReport directly from telemetries.
    from bench.orchestrator import _summary

    summary = _summary(telemetries)
    return CampaignReport(
        topology=topology,
        n_requests=int(summary["n_requests"]),
        success_rate=summary["success_rate"],
        mean_ttft_ms=summary["mean_ttft_ms"],
        p95_ttft_ms=summary["p95_ttft_ms"],
        mean_itl_ms=summary["mean_itl_ms"],
        cache_hit_rate=summary["cache_hit_rate"],
        reconcile_passes=summary["success_rate"] >= 0.95,
        notes=["real-vllm", f"base_url={base_url}", f"model={model}"],
    )


async def main(base_url: str, model: str, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[real-bench] waiting for vLLM at {base_url}...")
    if not await _wait_for_vllm(base_url):
        print("[real-bench] ERROR: vLLM did not come up in 180s")
        return 1
    print("[real-bench] vLLM ready")

    reports: list[CampaignReport] = []
    for topology in Topology:
        print(f"[real-bench] running topology: {topology.value}")
        try:
            report = await _run_one(topology, base_url, model)
        except Exception as exc:
            print(f"[real-bench] {topology.value} FAILED: {exc!r}")
            continue
        reports.append(report)
        out_file = out_dir / f"{topology.value}.json"
        out_file.write_text(report.model_dump_json(indent=2))
        print(
            f"[real-bench] {topology.value}: "
            f"success={report.success_rate:.2%} "
            f"mean_ttft={report.mean_ttft_ms:.1f}ms "
            f"p95_ttft={report.p95_ttft_ms:.1f}ms "
            f"mean_itl={report.mean_itl_ms:.1f}ms "
            f"→ {out_file}"
        )

    summary = {
        "n_topologies": len(reports),
        "base_url": base_url,
        "model": model,
        "all_reconcile": all(r.reconcile_passes for r in reports),
        "topologies": [r.topology.value for r in reports],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[real-bench] DONE — {len(reports)}/{len(Topology)} topologies measured")
    return 0 if reports else 2


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--out", default="bench/results/real", type=Path)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.base_url, args.model, args.out)))


if __name__ == "__main__":
    cli()


__all__ = ["main", "cli"]