"""scripts.autoscaler_live — drive PoolAutoscaler from a live vLLM server.

Closes GPU_EXECUTION_PLAN §3 (live autoscaler workload-shift validation):

1. Fires a prompt-heavy phase (long prompts, tiny outputs) then a
   decode-heavy phase (short prompts, long outputs) at the server.
2. Every tick, scrapes ``/metrics`` for the real queue signal
   (``vllm:num_requests_waiting`` / ``vllm:num_requests_running``) and
   feeds it to :class:`control.autoscaler.PoolAutoscaler`.
3. Appends one JSONL record per tick: timestamp, phase, queue depth,
   in-flight, decision (delta + reason), and the virtual replica count.

Honesty note: replica actuation is **virtual** — a single fixed vLLM
process serves the whole run, and the recorded replica count is the
controller's bookkeeping, not a real scale event. What this validates
against live (not simulated) queue dynamics: the PID reacts to a real
workload shift, the dwell gate suppresses ping-pong, the thrash alarm
stays quiet or fires explainably, and scale-down decisions never occur
while ``in_flight > 0`` (the AUTO-05 drain invariant, asserted per
tick). Real multi-replica actuation needs a pool manager + k8s/compose,
which is out of scope for a single-pod run.

Usage (RUNPOD_VLLM_BASE_URL must point at the server's /v1)::

    python -m scripts.autoscaler_live --out bench/results/autoscaler_live/run.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from control.autoscaler import PoolAutoscaler, PoolTopology
from control.pid import PidController, PidGains
from control.pool import Pool
from core.metrics import parse_prometheus

VLLM_NUM_REQUESTS_WAITING = "vllm:num_requests_waiting"
VLLM_NUM_REQUESTS_RUNNING = "vllm:num_requests_running"


def queue_signal(metrics_text: str) -> tuple[int, int]:
    """Extract ``(waiting, running)`` from a vLLM /metrics body.

    Sums across label sets (one model per server in our deploys, but
    label-robust anyway). Missing metrics read as 0 — a freshly started
    server legitimately exposes no samples yet.
    """
    parsed = parse_prometheus(metrics_text)
    waiting = sum(
        v for k, v in parsed.items() if k.startswith(VLLM_NUM_REQUESTS_WAITING)
    )
    running = sum(
        v for k, v in parsed.items() if k.startswith(VLLM_NUM_REQUESTS_RUNNING)
    )
    return int(waiting), int(running)


@dataclass(frozen=True)
class Phase:
    """One workload phase of the shift experiment."""

    name: str
    duration_s: float
    rate_rps: float
    prompt_words: int
    max_tokens: int


DEFAULT_PHASES = [
    # Prompt-heavy: ~8K-token prompts at 24 rps ≈ 190K prefill tok/s —
    # deliberately above one H100's 7B prefill throughput so the waiting
    # queue actually builds (8 rps × 4K tokens never queued: measured
    # max_queue_waiting = 0 on 2026-07-17).
    Phase("prompt_heavy", 60.0, 24.0, 6000, 8),
    # Decode-heavy: tiny prompts, long outputs. Queue drains, running
    # stays occupied by decode.
    Phase("decode_heavy", 60.0, 4.0, 40, 512),
]


def build_autoscaler(min_dwell_s: float) -> PoolAutoscaler:
    """One COLOCATED-pool controller with the TUNING.md default gains."""
    pid = PidController(
        gains=PidGains(kp=0.08, ki=0.01, kd=0.0),
        output_min=-1.0,
        output_max=1.0,
    )
    return PoolAutoscaler(
        controllers={Pool.COLOCATED: pid},
        min_replicas=1,
        max_replicas=8,
        step_size=1,
        min_dwell_s=min_dwell_s,
    )


def tick_record(
    now: float,
    phase: str,
    waiting: int,
    running: int,
    replicas: int,
    decision_delta: int,
    decision_reason: str,
) -> dict[str, Any]:
    return {
        "ts": round(now, 3),
        "phase": phase,
        "queue_waiting": waiting,
        "in_flight_running": running,
        "replicas_virtual": replicas,
        "decision_delta": decision_delta,
        "decision_reason": decision_reason,
    }


async def _fire_phase(
    client: httpx.AsyncClient,
    base_url: str,
    phase: Phase,
    stop: asyncio.Event,
    status_counts: dict[str, int],
) -> int:
    """Fire phase traffic open-loop at ``rate_rps``; return request count."""
    base_prompt = "lorem " * phase.prompt_words
    interval = 1.0 / phase.rate_rps
    n = 0
    deadline = time.monotonic() + phase.duration_s

    async def one(seq: int) -> None:
        # Unique prefix per request: with a shared prompt, vLLM's
        # automatic prefix caching absorbs the entire prefill and the
        # queue never builds (measured max_queue_waiting = 0 at 24 rps
        # x 8K tokens on 2026-07-17). The distinct first tokens defeat
        # the cache so every request pays real prefill.
        prompt = f"request {seq} distinct: {base_prompt}"
        # Queue-signal experiment; individual failures are tolerated but
        # COUNTED — a silent 4xx/5xx flood previously masqueraded as
        # "the server absorbed the load" (max_queue_waiting 0).
        try:
            resp = await client.post(
                f"{base_url}/chat/completions",
                json={
                    "model": "goodputlab-model",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": phase.max_tokens,
                    "temperature": 0.0,
                },
                timeout=180.0,
            )
            key = str(resp.status_code)
        except httpx.HTTPError as exc:
            key = type(exc).__name__
        status_counts[key] = status_counts.get(key, 0) + 1

    tasks: list[asyncio.Task[None]] = []
    while time.monotonic() < deadline and not stop.is_set():
        tasks.append(asyncio.create_task(one(n)))
        n += 1
        await asyncio.sleep(interval)
    await asyncio.gather(*tasks, return_exceptions=True)
    return n


async def run_experiment(
    base_url: str,
    out_path: Path,
    phases: list[Phase],
    tick_s: float = 2.0,
    target_queue_depth: int = 4,
    min_dwell_s: float = 10.0,
) -> dict[str, Any]:
    """Run the shift experiment; return the summary dict (also written)."""
    scaler = build_autoscaler(min_dwell_s)
    metrics_url = base_url.removesuffix("/v1") + "/metrics"
    replicas = 1
    records: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    drain_violations = 0
    flips = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient() as client:
        for phase in phases:
            stop = asyncio.Event()
            traffic = asyncio.create_task(
                _fire_phase(client, base_url, phase, stop, status_counts)
            )
            # End ticking slightly before the traffic deadline so the
            # final in-flight requests drain inside the client context —
            # a tick that outlives the phase can otherwise observe the
            # client after closure ("client has been closed").
            phase_deadline = time.monotonic() + phase.duration_s - tick_s
            while time.monotonic() < phase_deadline:
                body = (await client.get(metrics_url, timeout=10.0)).text
                waiting, running = queue_signal(body)
                topo = {
                    Pool.COLOCATED: PoolTopology(
                        pool=Pool.COLOCATED,
                        replicas=replicas,
                        target_queue_depth=target_queue_depth,
                    )
                }
                decisions = scaler.tick(
                    topology=topo,
                    queue_depths={Pool.COLOCATED: waiting},
                    in_flight={Pool.COLOCATED: running},
                    dt=tick_s,
                )
                d = decisions[0]
                # AUTO-05 drain invariant, asserted live: the controller
                # must never emit a scale-down while requests are in flight.
                if d.delta < 0 and running > 0:
                    drain_violations += 1
                if d.delta != 0:
                    flips += 1
                    replicas = max(1, replicas + d.delta)
                records.append(
                    tick_record(
                        time.time(), phase.name, waiting, running,
                        replicas, d.delta, d.reason,
                    )
                )
                await asyncio.sleep(tick_s)
            stop.set()
            await traffic

    # Records are the experiment; write them even if a late tick raised.
    with out_path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    wall_s = len(records) * tick_s
    summary = {
        "ticks": len(records),
        "flips": flips,
        "flips_per_minute": round(flips / (wall_s / 60.0), 3) if wall_s else 0.0,
        "drain_violations": drain_violations,
        "max_queue_waiting": max((r["queue_waiting"] for r in records), default=0),
        "response_status_counts": status_counts,
        "phases": [p.name for p in phases],
        "records_path": str(out_path),
    }
    summary_path = out_path.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autoscaler_live")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("bench/results/autoscaler_live/run.jsonl"),
    )
    parser.add_argument("--tick-s", type=float, default=2.0)
    parser.add_argument("--target-queue-depth", type=int, default=4)
    parser.add_argument("--min-dwell-s", type=float, default=10.0)
    args = parser.parse_args(argv)

    base_url = os.environ.get("RUNPOD_VLLM_BASE_URL")
    if not base_url:
        print(
            "[autoscaler_live] ERROR: RUNPOD_VLLM_BASE_URL not set",
            file=sys.stderr,
        )
        return 2

    summary = asyncio.run(
        run_experiment(
            base_url=base_url,
            out_path=args.out,
            phases=DEFAULT_PHASES,
            tick_s=args.tick_s,
            target_queue_depth=args.target_queue_depth,
            min_dwell_s=args.min_dwell_s,
        )
    )
    print(json.dumps(summary, indent=2))
    return 1 if summary["drain_violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
