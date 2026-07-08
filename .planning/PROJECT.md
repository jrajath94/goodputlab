# GoodputLab — SLO-Aware Disaggregated Inference Serving

## What This Is

GoodputLab is the **control plane** for prefill/decode (P/D) disaggregated LLM inference: an SLO-aware router, KV-cache tier, and P:D autoscaler that sit in front of vLLM pools. It benchmarks when disaggregation actually pays vs. when chunked-prefill wins — the honest frontier for inference optimization. Built as a portfolio flagship for Staff-level inference/performance roles targeting Anthropic, OpenAI, frontier labs.

## Core Value

**Goodput (throughput × SLO attainment) under realistic mixed workloads**, with verified, reproducible numbers and a public artifact trail — not marketing claims.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] All 4 topologies (colocated, chunked-prefill, disagg, disagg+tiering) deploy and serve end-to-end on a single GPU node
- [ ] Load generator emits reproducible chat / RAG / agentic traces with reconciled metrics (±2% of vLLM truth)
- [ ] Cache-aware router selects prefill/decode pools by prefix-hash affinity with load as tiebreaker
- [ ] SLO-class admission control sheds BATCH traffic to hold INTERACTIVE TTFT p95 attainment ≥99%
- [ ] LMCache KV tier (HBM → DRAM → disk) with documented break-even curve
- [ ] EAGLE-3 speculative decoding on decode pool with auto-disable above batch-size threshold
- [ ] P:D autoscaler (PID on queue-depth) with 120s minimum dwell and drain protocol
- [ ] Benchmark campaign: 4 topologies × 3 workloads × 6 load levels × 3 seeds → goodput curves, TTFT/ITL CDFs, cost/1M tokens
- [ ] Failure drills: kill decode mid-stream, KV-transfer stall, pathological mixes — written as production postmortem
- [ ] One-command `make bench` reproduces every README number from cold node in <20 minutes
- [ ] 3,000-word report: "When disaggregation pays: an SLO-aware study" with the honest chunked-prefill-vs-disagg finding
- [ ] ≥80% pytest coverage on core/control modules

### Out of Scope

- Training new model weights (this is serving, not pretraining)
- Frontend UI / chat product (we expose OpenAI-compatible HTTP endpoints only)
- Multi-tenant auth / billing (single-tenant benchmarking rig)
- Cloud-managed autoscaler integration (we ship our own PID controller)
- LMCache training/prewarming of specific prefix distributions (out of band)
- Cross-region failover (single-region bench)
- Production-grade observability stack hardening (basic Prometheus/Grafana only)

## Context

**Why this exists:** P/D disaggregation is the industry default for serving large LLMs (vLLM, SGLang, TensorRT-LLM, DeepSeek, Moonshot all ship it). The frontier has moved from "can we disagg?" to **"when does it pay, and what routing/scheduling policy extracts the most goodput?"** This project sits exactly on that frontier.

**Audience:** Inference/performance engineers evaluating disagg ROI, hiring committees at frontier labs, the open-source community that needs honest numbers.

**Prior art / inspiration:** vLLM v1 disagg (NIXL-based KV transfer), LMCache (LMSYS), Moonshot Kimi K2 architecture, DeepSeek V3/R1 serving stack, Anyscale/SkyPilot patterns.

**Hardware plan:** Start on 2× A100/H100 spot (~$3-6/hr dev). Stretch to 4-8× H100 for the full benchmark campaign (W10). All H100 spot.

## Constraints

- **GPU budget:** $600-1200 total. Spot-only. No debugging on bench rig.
- **Timeline:** 8-10 weekends (W1-W10 from CLAUDE.md). Tier-1 flagship delivery before Aug 2026.
- **Engine:** vLLM v1 (latest stable) — SGLang noted as alternative if vLLM flags change mid-phase.
- **Stack lock-in:** NIXL for KV transfer, LMCache for tiering, EAGLE-3 for spec decode. Each must justify its cost or get cut.
- **Reproducibility:** Every number traceable to a commit + seed + hardware record.
- **Cost discipline:** Cold-to-serving <20 min; `make bench` idempotent; parquet/S3 capture immediate.
- **No fabrication:** All numbers from runs or marked `[NOT YET MEASURED]`. All docs cite upstream versions.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Goodput = primary metric, not throughput | SLOs are the constraint; frontier labs optimize goodput; honest measurement requires the constraint | — Pending |
| Queued tokens, not requests, as load signal | Prefill cost ∝ prompt length; request count underweights long prompts | — Pending |
| Cache-first, load-second routing policy | Cache hit saves more than load rebalance in steady state; tiebreaker = load | — Pending |
| Honest boundary findings required | Chunked-prefill beats disagg on some workloads; report this (staff move) | — Pending |
| 120s autoscaler dwell, not per-request | Avoid role-flip thrash; documented in Phase 9 | — Pending |
| vLLM as primary engine | Production-grade, P/D disagg in v1 release line; SGLang kept as fallback | — Pending |
| Single-node first, multi-node stretch | Simpler cause isolation; spec requires single-node cold-to-20min | — Pending |

## Architecture

```
Requests (tagged SLO class)
       ↓
┌──────────────────────────────────────┐
│  GoodputLab Router (SLO-aware)      │
│   - SLO classifier                   │
│   - Cache-aware (prefix hash 256-tok)│
│   - P/D pool selection               │
│   - Admission control                │
└──────────────────────────────────────┘
         ↓                      ↓
   ┌────────────────┐    ┌──────────────────┐
   │ PREFILL Pool   │ KV │ DECODE Pool      │
   │ × 2 vLLM       │Xfer│ × 2 vLLM         │
   │ chunked PF     │NIXL│ continuous batch │
   │ prefix $       │    │ EAGLE-3 spec     │
   └────────────────┘    └──────────────────┘
         ↓                      ↓
      ┌──────────────────────────┐
      │ KV Tier: LMCache         │
      │ HBM → DRAM → disk        │
      └──────────────────────────┘

+ Autoscaler: P:D ratio (PID on queue depth, 120s dwell)
+ Prometheus/Grafana metrics
+ Load generator: chat / RAG / agentic traces
```

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-07-08 after initialization*