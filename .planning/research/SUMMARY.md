# Project Research Summary — GoodputLab

**Project:** GoodputLab — SLO-Aware Disaggregated LLM Inference Serving
**Domain:** LLM inference serving (P/D disagg, SLO routing, KV tier, autoscaler, spec decode)
**Researched:** 2026-07-08
**Confidence:** HIGH

---

## Executive Summary

P/D disaggregation is the industry default in 2026 across vLLM, SGLang, TensorRT-LLM, Moonshot, DeepSeek. Frontier-lab posture treats **goodput** (throughput × SLO attainment), not raw throughput, as the primary metric. GoodputLab is the control plane on top of vLLM v1 disagg: cache-aware FastAPI router + LMCache KV tier + P:D PID autoscaler + EAGLE-3 spec decode, benchmarked honestly against chunked-prefill.

**Stack:** vLLM v0.11.x pinned, NIXL 0.6.x over **UCX** (not LIBFABRIC — silent corruption vllm #27055), LMCache 0.3.x per-SLO namespaces, EAGLE-3 head from HF, FastAPI/Pydantic v2, prometheus-client. Single-node first; multi-node = v2.

**Top risks (all critical/high):** NIXL silent garbage (P1, UCX + sentinel validity); CVE-2025-25183 prefix-cache hash collision (P2, SHA-256 + per-pool salt + vLLM ≥0.11.x); spec×disagg draft KV uninit (P3, acceptance-vs-colocated gate); chunked-vs-disagg crossover overclaim (P4, same-axes goodput curves); drain-protocol in-flight loss (P5, explicit coord, 0 drops); thrashing (P6, 2-tier gate + 120s dwell). Capstone = honest chunked-prefill wins-some-cells finding.

---

## Key Findings

### Recommended Stack

- **vLLM v0.11.x** — pinned (≥0.11.x fixes CVE-2025-25183; NixlConnector in v1 disagg path)
- **NIXL 0.6.x / UCX 1.15+** — UCX backend mandatory (LIBFABRIC silent garbage, vllm #27055)
- **LMCache 0.3.x** — per-SLO namespaces, HBM→DRAM→disk tier, `enable_pd` + `transfer_channel:"nixl"`
- **EAGLE-3** — Llama-3.3-70B verified head (`nvidia/Llama-3.3-70B-Instruct-Eagle3`); Qwen3-32B community head TBD
- **FastAPI 0.115+ / Pydantic v2 / uvicorn** — router HTTP front door
- **prometheus-client 0.20+ / Grafana** — `/metrics` endpoint + dashboard JSON
- **2-4× H100 80GB SXM spot** — Phase 1 dev; 4-8× for Phase 8 bench campaign

### Expected Features

**Table stakes (must have):** OpenAI-compat HTTP+SSE, `/metrics`, PagedAttention/continuous batching, per-token timestamps, `make provision` <20min, HW/seed/version metadata.

**Differentiators (GoodputLab-specific):** cache-aware prefix routing (256-tok block hash, SHA-256/128 + per-pool salt), SLO admission, LMCache per-SLO tier, P:D PID autoscaler, EAGLE-3 auto-disable, goodput-as-primary, cold/warm regime split, chunked-vs-disagg crossover, pathological-mix failure drill.

**Anti-features (avoid):** per-request autoscaling (thrashing), unbounded prefix index (RSS blowup P8), Python `hash()` (CVE-2025-25183), LIBFABRIC NIXL default (silent garbage P1), naive round-robin under heterogeneous prefix load, static spec-decode batch cap (cross-batch threshold moves P11), in-flight drops during role flip (no drain handshake P5).

### Architecture Approach

Router (SLO classify, prefix-hash LRU, cache-first/load-tiebreak, BATCH-shed admission) → P-pool×2 (vLLM chunked-prefill) + D-pool×2 (vLLM cont-batch + EAGLE-3) connected via NIXL UCX async send/recv → LMCache tier (HBM→DRAM→NVMe, per-SLO namespaces, LRU) + Autoscaler (PID 1s tick, 120s dwell, drain coord) + Load Gen + Obs (Prom/Grafana, per-30s reconcile). 1% sentinel-token validity test every disagg hop (mitigates P1).

### Critical Pitfalls (top 6 of 12)

1. **P1 NIXL silent garbage** (CRITICAL) — UCX backend + sentinel-token validity test
2. **P2 CVE-2025-25183** (CRITICAL multi-tenant; MEDIUM here) — vLLM ≥0.11.x + SHA-256 + per-pool salt
3. **P3 Spec×disagg draft KV uninit** (HIGH) — acceptance ≥0.85× of colocated gate
4. **P4 Chunked crossover unmeasured** (HIGH) — same-axes goodput curves
5. **P5 Drain in-flight loss** (HIGH) — explicit coord, 0 drops
6. **P6 Autoscaler thrashing** (HIGH) — 2-tier gate, 120s dwell, over-damped PID

---

## Implications for Roadmap

### Phase 1: Topologies (TOPO)

**Rationale:** Need all 4 topologies serving before any measurement is possible.
**Delivers:** docker-compose for colocated / chunked-prefill / disagg / disagg+tiering; health checks.
**Acceptance gates:** cold→serving <20min, P→D KV confirmed (sentinel test, not just count increment), vLLM ≥0.11.x pinned, UCX backend.

### Phase 2: Load + Metrics (LOAD)

**Rationale:** Bench numbers are meaningless without reconciled truth.
**Delivers:** chat/RAG/agentic traces, per-request logs, Prom/Grafana setup.
**Acceptance gates:** ≤2% reconciliation per-30s (not per-run), single TTFT def, separate `router_overhead_ms`.

### Phase 3: Router + Admission (RTR)

**Rationale:** SLO-aware control plane = the artifact's differentiator.
**Delivers:** cache-aware router, SLO classifier, admission control.
**Acceptance gates:** SHA-256/128 prefix hash + per-pool salt, TTL 1hr + LRU size cap.

### Phase 4: Router Verification (RTR-verify)

**Rationale:** Cache-aware router false confidence pitfall (P7).
**Delivers:** A/B vs round-robin, cold/warm regime split, regime isolation.

### Phase 5: KV Tiering (KV)

**Rationale:** LMCache tier breaks long-context RAG economics.
**Delivers:** LMCache integration, prewarming, eviction policy per workload.
**Acceptance gates:** per-SLO namespaces, prewarm hot prefixes, per-class budget, eviction rate metric.

### Phase 6: Speculative Decoding (SPEC)

**Rationale:** Spec × disagg is poorly charted territory; gate required.
**Delivers:** EAGLE-3 on decode pool, ITL-vs-batch curve, auto-disable.
**Acceptance gates:** acceptance ≥0.85× of colocated, auto-disable breaker at crossover.

### Phase 7: Autoscaler (AUTO)

**Rationale:** P:D ratio dynamics are the frontier's hardest control problem.
**Delivers:** over-damped PID + 2-tier gate + 120s dwell + drain coord.
**Acceptance gates:** 0 in-flight drops during forced role flip, flip-count-per-minute <0.5.

### Phase 8: Benchmark Campaign + Report (BENCH)

**Rationale:** The artifact's value = the honest findings.
**Delivers:** 4 topo × 3 wl × 6 load × 3 seed matrix, goodput curves, CDFs, cost table, failure drills, 3K-word report.
**Acceptance gates:** chunked-prefill on same axes as disagg, 3 failure drills, `make bench` reproducible.

---

## Phase Ordering Rationale

- TOPO → LOAD: load gen needs running topologies (P1 health check gates P2 measurement)
- LOAD → RTR: router needs realistic traffic to test routing decisions
- RTR → RTR-verify: verify A/B claim with regime isolation
- RTR → KV: KV tier breaks long-context economics (RAG workload)
- KV → SPEC: spec decode assumes KV layout matches target (P3 gate before enabling)
- SPEC → AUTO: autoscaler requires stable per-pool behavior first
- AUTO → BENCH: bench campaign measures whole-system goodput

**Grouping rationale:** Each phase delivers end-to-end testable capability (MVP-style) rather than horizontal layer; vertical slices make failure visible early.

**Pitfall avoidance:** Sentinel test in TOPO gates LOAD. SHA-256 in RTR gates Router Verify. Per-SLO LMCache gates SPEC. Drain coord in AUTO gates BENCH. Acceptance gates inherited from PITFALLS.md.

---

## Research Flags

**Phases needing deeper research during planning:**
- **Phase 5 (KV):** LMCache eviction semantics under agentic workload — empirical measurement, no canonical ref
- **Phase 7 (AUTO):** PID gain tuning methodology — first-principles, no canonical GPU-inference ref
- **Phase 6 (SPEC):** EAGLE-3 × disagg pairwise interaction — 200-line reproducer experiment before production

**Phases with standard patterns (skip research-phase):**
- **Phase 1 (TOPO):** Well-documented docker-compose patterns
- **Phase 3 (RTR):** FastAPI + prefix-hash indexing = established patterns

---

## Confidence Assessment

| Area | Confidence | Reason |
|------|------------|--------|
| Stack | HIGH | Context7-verified vLLM v0.11.x + NIXL + LMCache + EAGLE-3 versions |
| Features | HIGH | Direct 1:1 mapping to 50 v1 requirements |
| Architecture | HIGH | Component boundaries from NIXLConnector + LMCache + KEDA precedent |
| Pitfalls (P1/P2/P3/P4/P6/P10/P11) | HIGH | First-party issues + control theory consensus |
| Pitfalls (P5/P7/P8/P9/P12) | MEDIUM | Logical consequences; magnitudes vary |
| Roadmap | HIGH | Dependencies traced; ordering enforced |

**Overall: HIGH.** Honest crossover mandate makes chunked-vs-disagg the load-bearing claim; instrumentation designed to prove or refute it.

---

## Gaps to Address

1. **LMCache eviction semantics under agentic workload** — Phase 5 empirical (P9, P12 mitigation requires measurement)
2. **PID gain tuning methodology** — Phase 7 first-principles (no canonical GPU-inference ref; derive from queue-depth transfer function)
3. **Per-30s reconciliation tooling** — Phase 2 (Prometheus query + statistical test design)
4. **EAGLE-3 × disagg pairwise interaction** — Phase 6 200-line reproducer before production enablement
5. **Pure cold-cache TTFT on cache-aware router** — Phase 4 cold-regime measurement design

---

## Sources

### HIGH (first-party docs / issues / CVEs)
- vLLM NixlConnector docs (Context7) — async send/recv, UCX + GDS backends
- LMCache architecture docs + #649 (P0 eviction bug)
- vLLM #27055 (NIXL LIBFABRIC silent garbage) + #24885 (graceful shutdown RFC)
- CVE-2025-25183 (prefix-cache hash collision, fix in 0.7.2)
- sglang #19796 (Eagle V2 NaN from draft KV uninit)
- TensorRT-LLM spec decode docs (pre-scheduling KV reconciliation pattern)
- EAGLE-3 HF model: `nvidia/Llama-3.3-70B-Instruct-Eagle3`
- Mooncake FAST 2025 paper — early-rejection load-aware admission
- KEDA + HPA stabilization patterns (120s dwell precedent)

### MEDIUM (cross-validated)
- Splitwise (ICML 2024), DistServe (OSDI 2024), Sarathi-Serve (EuroSys 2025) — chunked vs disagg crossover
- Hydragen, TetriInfer, Infinite-LLM, SpotServe — P/D disagg alternatives
- BARS (Microsoft/PKU) — Poisson vs bursty: p99 7.5× higher
- K8s RDMA gotchas writeup (Medium)
- NVIDIA KV cache compression research — eviction policy regress vs LRU

### LOW (validate in execution)
- LMCache stampede behavior — Momento blog only
- PID gain tuning — no canonical ref

---
*Research completed: 2026-07-08*
*Ready for roadmap: yes*