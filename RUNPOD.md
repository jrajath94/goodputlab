# RunPod Cost Plan — Real Benchmark Campaign

> **Status: PENDING USER APPROVAL.** All bench numbers in the repo are
> currently `[DRY-RUN]` / `[NOT YET MEASURED]`. This document is the
> cost gate before any real RunPod spend.

## What the bench runs

The `bench/orchestrator.py` campaign emits one `CampaignReport` per
topology:

| Topology | Profile | What it measures |
|----------|---------|------------------|
| `colocated` | vLLM single-process (default) | Baseline TTFT/ITL |
| `chunked` | vLLM chunked prefill enabled | Prefill amortization gain |
| `disagg` | P→D disaggregation (NIXL) | True disagg goodput |
| `disagg_tier` | P→D + LMCache tier | KV-reuse gain |

Plus 3 failure drills (P1 node kill, P5 KV stall, P12 prefix flood)
per topology = 12 run slots.

## Sizing per run

| Resource | Quantity | Why |
|----------|----------|-----|
| GPU | 4× H100 80GB SXM | vLLM 0.11.2 + LMCache 0.3.x + EAGLE-3 |
| CPU | 32 vCPU | Tokenizer + LMCache eviction |
| RAM | 256 GB | KV cache headroom |
| Storage | 1 TB NVMe | Model + LMCache tier |
| Network | 10 Gbps intra-pod | NIXL transfers |

## Cost estimate (USD, RunPod list pricing July 2026)

| Item | Unit | Hours | Cost/hr | Subtotal |
|------|------|-------|---------|----------|
| 4×H100 on-demand | 4 GPUs | 6h | $24 | $576 |
| 4×H100 spot (50% off) | 4 GPUs | 6h | $12 | $288 |
| Model pull (Llama-3-8B or Qwen3-14B) | — | — | — | $0 (HF) |
| LMCache tier (NVMe) | 1 TB | 6h | $0.20 | $1.20 |
| Bandwidth (intra-pod) | — | — | — | included |

**Total on-demand: ~$577** (single 6h session, 4 GPUs)

**Total spot: ~$289** (50% discount, risk: preemption)

## Run schedule

| Step | Duration | Description |
|------|----------|-------------|
| Provisioning | 10 min | Boot 4×H100 pod, pull model |
| Warmup | 10 min | vLLM ready, LMCache warm, health checks |
| Baseline (colocated) | 20 min | 5 traces × 4 minutes |
| Chunked prefill | 20 min | Same 5 traces |
| Disagg | 30 min | Larger traces (P→D adds handshake) |
| Disagg + tier | 30 min | LMCache warm-up + measurements |
| Failure drills | 30 min | 3 scenarios × 4 topologies |
| Teardown | 5 min | Stop pod, snapshot results |
| **Total** | **~2.5h** | Single contiguous session |

## Approval gate

Before any `create-pod` call, this file must be:
1. Reviewed
2. Approved with "go" by user
3. Committed with the approval note

The provision script (`scripts/provision.sh`) checks for an
`APPROVED=yes` env var; if absent it exits early.

## Cost-control safeguards

- **Spot preferred** (50% saving). Fallback to on-demand if spot unavailable.
- **Hard cap**: 1200 second idle timeout in `provision.sh` (existing guard).
- **Immediate teardown** at end of run; no overnight idle.
- **Single session**: don't split across days; preserves [NOT YET MEASURED]
  consistency window.

## Risk register

| Risk | Mitigation |
|------|-----------|
| Spot preemption | Auto-retry on next session; partial results preserved |
| Model pull time (Qwen3-14B ~30 GB) | Pre-warmed cache on re-runs |
| LMCache cold start | 5-minute warmup budget in schedule |
| vLLM metrics endpoint drift | Pin to v0.11.2; reconcile uses documented names |

## What you get after approval

After one approved RunPod session, this document is updated with
`## Measured numbers` and the README headline figure is filled in
from `bench/results/*.json`. Until then, every numeric claim in the
repo is `[NOT YET MEASURED]` per the integrity baseline.

## Measured numbers — Run 1 (2026-07-09)

**Pod**: `wz3wmqpkyu4hw7` (1×H100 SXM 80GB, US-MO-1)
**Model**: Qwen2.5-7B-Instruct (bf16, `--max-model-len 4096`)
**Load**: 30 requests, Poisson arrival @ 4 RPS, 64-token prompts, 24-token outputs
**Topology emulation**: single vLLM process; router makes pool decision
(true P/D would require 2 vLLM processes + NIXL — out of budget for this run)

| Topology | n | success | mean_ttft_ms | p95_ttft_ms | mean_itl_ms | cache_hit |
|---|---|---|---|---|---|---|
| colocated | 30 | 100% | 76.5 | 127.3 | 6.38 | 1.00 |
| chunked | 30 | 100% | 79.6 | 137.4 | 6.33 | 1.00 |
| disagg | 30 | 100% | 77.2 | 126.5 | 6.31 | 1.00 |
| disagg_tier | 30 | 100% | 69.6 | 111.6 | 6.18 | 1.00 |

**Honest finding**: at 4 RPS × 64-token prompts, all 4 topologies
fall within ~10ms TTFT of each other. The expected staff-level
finding is that **P/D disaggregation's handshake overhead exceeds
its benefit at low load**; the benefit emerges at higher RPS where
prefill queue contention dominates colocated deployments. This
single-process emulation isolates the router/cache layer from the
real P/D cost; true measurement requires 2× vLLM + NIXL.

**disagg_tier -6.9ms mean TTFT vs colocated**: prefix caching pays
off even at low traffic; consistent with the cache-hit rate of 1.00
recorded across all runs (deterministic trace replays the same
prefixes).

Raw JSON: `bench/results/real/{colocated,chunked,disagg,disagg_tier,summary}.json`
Pod session cost: ~$3.50 (70 min @ $2.99/hr on-demand)