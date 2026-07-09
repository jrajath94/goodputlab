# Architecture — SLO-Aware Disaggregated LLM Serving

**Project:** GoodputLab | **Researched:** 2026-07-08
**Confidence:** HIGH (vLLM v1 NIXL, LMCache, EAGLE-3) | MEDIUM (PID tuning, Moonshot K2 specifics)

## System Diagram

```
  Clients (chat/RAG/agentic, SLO-tagged) → ROUTER (FastAPI: SLO classify, prefix-hash LRU, cache-first/load-tiebreak, admission BATCH-shed)
       │prefill                              │decode + KV addr
       ▼                                     ▼
  ┌──────────┐  NIXL UCX async send/recv  ┌──────────┐
  │ PREFILL×2│◀──────────────────────────▶│ DECODE×2 │ vLLM cont-bat + EAGLE-3
  │ vLLM ch  │                            └────┬─────┘
  └────┬─────┘                                 │
       ▼                                       ▼
  ┌──────────────────────────────────────────────┐
  │ KV TIER: LMCache HBM→DRAM→NVMe, per-SLO ns, LRU │
  └──────────────────────────────────────────────┘
  + AUTOSCALER (PID 1s, 120s dwell, drain coord)
  + LOAD GEN (Poisson+bursts, seed-pinned) | OBS (Prom/Grafana, per-30s reconcile)
```

## Components (7)

| # | Component | Tech | Responsibility |
|---|-----------|------|----------------|
| 1 | Router | FastAPI/asyncio, LRU+bloom, SHA-256/128 + per-pool salt | SLO classify, prefix-hash, pool select, admission |
| 2 | Prefill Pool | vLLM v1 chunked-prefill, NixlConnector UCX | Prompt→KV, emit, transfer |
| 3 | Decode Pool | vLLM v1 cont-batch, EAGLE-3 (gated) | Pull KV, generate, spec-decode |
| 4 | KV Tier | LMCache per-SLO namespace | Multi-tier cache, eviction, prewarm |
| 5 | Autoscaler | asyncio; 2-tier gate; 120s dwell; anti-windup | P:D ratio via PID + drain coord |
| 6 | Load Gen | Custom harness, parquet, seed-pinned | Reproducible mixed traces |
| 7 | Observability | Prom/Grafana, per-30s reconcile | Reconcile client↔engine; bench capture |

## Data Flow

- **Request:** Client → Router (parse+classify+prefix-lookup+admit) → P-pool prefill → NIXL `send` UCX async → D-pool `recv` → populate KV → stream tokens. 1% sampled sentinel-token validity test (P1).
- **KV-transfer:** P emits block hashes+RDMA handles → async send → D populates paged KV → first-token.
- **Metrics:** vLLM `/metrics` 5s scrape → reconcile by `request_id` → `router_overhead_ms`, `clock_skew_ms`; Grafana: TTFT/ITL CDF, goodput, `kv_transfer_inflight`, `role_flip_count`, `prefix_index_size_bytes`.

## Build Order (W1-W8)

1. **W1 TOPO** — docker-compose × 4 topologies; `make provision` cold→serving <20 min; sentinel health check; vLLM ≥0.7.2; UCX backend.
2. **W2 LOAD** — chat/RAG/agentic mixer; per-30s reconcile ≤2% drift; parquet.
3. **W3 RTR** — prefix-hash LRU w/ hard cap; per-pool salt; BATCH shedding.
4. **W4 RTR-verify** — A/B vs round-robin; **cold+warm regimes reported separately**; load-bal fallback on no-history.
5. **W5 KV** — LMCache per-SLO namespaces; prewarm hot prefixes; per-workload policy.
6. **W6 SPEC** — EAGLE-3 w/ **SPEC-04 gate**: acceptance vs colocated ≥0.85× else abort; ITL-vs-batch crossover; auto-disable breaker.
7. **W7 AUTO** — PID + 2-tier gate; explicit drain coord; 120s dwell; anti-windup; AUTO-05 = zero drops.
8. **W8 BENCH** — 4 topo × 3 wl × 6 load × 3 seed; goodput w/ chunked-prefill on same axes; failure drills (P5, P12); 3K-word report.

## Reference Architectures

- **vLLM v1 disagg (NIXL async send/recv, UCX):** primary P→D path; verify `--kv-transfer-config` day-1 each phase.
- **Moonshot Kimi K2 (KVCache-centric):** KV as first-class persistent resource → per-SLO LMCache budget.
- **SGLang RadixAttention:** tree prefix sharing → informs router keying + LMCache namespaces (pattern only; vLLM primary).
- **KEDA/HPA stabilization:** validates 120s dwell; logic inspired, not adopted.

## Anti-Patterns (explicit avoid)

| Anti-pattern | Why bad | Prevention |
|--------------|---------|-----------|
| Per-request autoscaling | Flip = 15-30s drain + warm repop; thrash (P6) | 2-tier gate; 120s actuator dwell |
| Unbounded prefix index | 50MB/hr @ 10k QPS; lookup stalls (P8) | Cap+TTL; SHA-256/128 (P2); first 4KB only |
| Naive round-robin | 0% prefix hit on agentic; defeats LMCache | Cache-first, load tiebreaker (queued tokens) |
| Trust vLLM TTFT as truth | Engine clock ≠ router clock; batching (P10) | Common boundary; `router_overhead_ms`; CDF reconcile per 30s |
| LIBFABRIC NIXL default | Silent KV corruption vLLM 0.11.0 (P1) | Pin UCX; sentinel before LIBFABRIC |
| Default SIGTERM on flip | In-flight KV hard-cut; client reset (P5) | Explicit drain: stop→inflight==0→flip→rejoin |
| "Disagg wins" w/o chunked axes | Wrong portfolio claim (P4) | Plot both; crossover = headline |

## Gaps to Validate Phase-by-Phase

1. vLLM `--kv-transfer-config` syntax v0.11.x (verify day-1).
2. LMCache per-SLO eviction/budget API (may need upstream PR).
3. PID gain methodology (no canonical GPU-inference reference).

---
*Sources: vLLM v0.11+ NixlConnector docs, LMCache architecture docs, Mooncake FAST 2025, DistServe/Splitwise/Sarathi, KEDA HPA stabilization.*
