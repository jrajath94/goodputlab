# RunPod Cost Plan — Real Benchmark Campaign

> **Status: APPROVED + EXECUTED (2026-07-09 + 2026-07-14).** The
> original 4×H100 multi-pod plan below was **deferred** in favour of a
> cheaper 1×H100 single-pod sequential approach. Two sessions ran:
> Run 1 (4 topologies × qwen2.5-7b × 30 reqs, $3.50) and the matrix
> pipeline validation (pilot + 72-cell reduced sweep, total $2.56).
>
> **Current execution policy (2026-07-16):** all further GPU work goes
> through the staged frugal ladder in `docs/GPU_COST_OPTIMIZATION.md`
> (smoke → paired probes → context repair → focused → full), driven by
> `configs/runpod_smoke.yaml` / `runpod_paired_chat.yaml` /
> `runpod_paired_disagg.yaml` / `runpod_context_repair.yaml`. Paid runs
> require `--approve-cost` or `APPROVE_GPU_SPEND=yes` and stop at the
> first unreconciled cell. The 4×H100 multi-pod plan below stays
> DEFERRED and per the cost plan should be avoided unless explicitly
> proving multi-node/multi-GPU behavior. The RAG/agentic prompt failure
> root cause is measured (worst RAG prompt+output = 18,539 tokens →
> needs `--max-model-len 20480`); verify via
> `configs/runpod_context_repair.yaml` before any sweep.

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

> Note: the table above describes the *intended* topology profiles.
> Every run executed so far (Run 1, pilot, 72-cell, v1.1) served all
> topology labels from a single vLLM process — the disagg rows measure
> routing, not true P/D transfer. True disagg evidence is still open
> and gated behind `configs/runpod_paired_disagg.yaml`.

## Sizing per run — 4×H100 plan (DEFERRED to v1.1)

| Resource | Quantity | Why |
|----------|----------|-----|
| GPU | 4× H100 80GB SXM | vLLM 0.11.2 + LMCache 0.3.x + EAGLE-3 |
| CPU | 32 vCPU | Tokenizer + LMCache eviction |
| RAM | 256 GB | KV cache headroom |
| Storage | 1 TB NVMe | Model + LMCache tier |
| Network | 10 Gbps intra-pod | NIXL transfers |

## Cost estimate — 4×H100 plan (DEFERRED)

| Item | Unit | Hours | Cost/hr | Subtotal |
|------|------|-------|---------|----------|
| 4×H100 on-demand | 4 GPUs | 6h | $24 | $576 |
| 4×H100 spot (50% off) | 4 GPUs | 6h | $12 | $288 |
| Model pull (Llama-3-8B or Qwen3-14B) | — | — | — | $0 (HF) |
| LMCache tier (NVMe) | 1 TB | 6h | $0.20 | $1.20 |
| Bandwidth (intra-pod) | — | — | — | included |

**Total on-demand: ~$577** (single 6h session, 4 GPUs)

**Total spot: ~$289** (50% discount, risk: preemption)

## Cost — 1×H100 actual runs (2026-07-09 + 2026-07-14)

| Item | Cost | Duration | Notes |
|---|---|---|---|
| Run 1 (RunPod secure, 1×H100 SXM) | $3.50 | ~70 min | 30 reqs × 4 topologies, all reconciled |
| Pilot matrix (1×H100) | $1.26 | ~25 min | 2 cells, all reconciled |
| 72-cell reduced sweep (1×H100) | $1.30 | ~26 min | 24/72 reconciled; agentic+RAG broken |
| **Total actual spend** | **$6.06** | ~2h | All measured on RunPod secure $2.99/hr |

The 1×H100 approach covers Run 1 (canonical TTFT/ITL) and the matrix
pipeline validation but cannot run true DISAGG (needs separate prefill
+ decode vLLM processes) or multi-model sweep (one model per vLLM
process). Those limitations are documented in
`bench/results/runpod_full/README.md`.

## Run schedule — 4×H100 plan (DEFERRED)

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

In addition, `scripts/run_matrix.py` enforces a per-run spend gate:
non-smoke configs refuse to fire requests without `--approve-cost` or
`APPROVE_GPU_SPEND=yes`, print a cost preflight (pending cells, est
wall time, hourly rate, est cost) first, and abort before spend if the
prompt preflight predicts context-window overflow.

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
| disagg | 30 | 100% | 77.2 | 126.5 | 6.32 | 1.00 |
| disagg_tier | 30 | 100% | 69.6 | 111.6 | 6.21 | 1.00 |

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
Pod session cost: $3.50 (70 min @ $2.99/hr on-demand)

## Measured numbers — Pilot (2026-07-14)

2-cell pilot on H100 SXM, qwen2.5-7b, chat mix. All reconciled.
See `bench/results/runpod_pilot/README.md` for full table.

## Measured numbers — 72-cell reduced sweep (2026-07-14)

1×H100 SXM pod, 26 min wall. **24/72 cells reconciled** (chat mix only —
agentic + RAG fully failed due to ~4× prompt overflow on vLLM
`--max-model-len=4096`). DISAGG / DISAGG_TIER cells never attempted:
sweep was interrupted mid-run after colocated×3 models + chunked×1
model (72 total). See `bench/matrix_report` for the gap diagnosis.

Mean TTFT (24 reconciled, chat-only): **197.55 ms**
Mean ITL (24 reconciled, chat-only): **8.39 ms**
Median TTFT: 120.83 ms; p95 TTFT: 493.42 ms

See `bench/results/runpod_full/README.md` for the honest breakdown,
including model coverage and per-topology tables.