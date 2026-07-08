# Roadmap: GoodputLab

## Overview

Build the SLO-aware control plane for prefill/decode disaggregated LLM inference on vLLM v1: a cache-aware router, LMCache KV tier, and P:D autoscaler sitting in front of vLLM pools, benchmarked honestly against chunked-prefill to surface the workload-shape crossover where each architecture wins. Eight MVP phases ship vertical slices — each ends with a verifiable capability (deploys, traces, A/B-verified routing, tier-integrated serving, autoscaler demo, full bench matrix) — so failures surface at the smallest possible scope.

## Phases

- [ ] **Phase 1: Topologies (TOPO)** - 4 serving topologies deploy and serve end-to-end with P→D KV transfer validated
- [ ] **Phase 2: Load + Metrics (LOAD)** - Reproducible trace load generator with metrics reconciled to vLLM truth within ±2%
- [ ] **Phase 3: Router + Admission (RTR)** - Cache-aware SLO router with BATCH-shed admission control
- [ ] **Phase 4: Router Verification (RTR-verify)** - Cache-aware routing claim verified with cold/warm regime isolation
- [ ] **Phase 5: KV Tiering (KV)** - LMCache tier integrated with per-workload eviction policy and break-even chart
- [ ] **Phase 6: Spec Decode (SPEC)** - EAGLE-3 on decode pool with auto-disable at batch-size crossover
- [ ] **Phase 7: Autoscaler (AUTO)** - PID P:D controller with 120s dwell and zero-drop drain protocol
- [ ] **Phase 8: Benchmark Campaign + Report (BENCH)** - Full matrix, goodput curves, CDFs, cost table, failure drills (capstone)

## Phase Details

### Phase 1: Topologies (TOPO)

**Goal**: All 4 serving topologies (colocated, chunked-prefill, disagg, disagg+tiering) deploy via docker-compose and serve OpenAI-compatible HTTP end-to-end on a single GPU node, with P→D KV transfer validated by sentinel-token test (not just count increment).
**Depends on**: Nothing (first phase)
**Mode**: mvp
**Requirements**: TOPO-01, TOPO-02, TOPO-03, TOPO-04, TOPO-05, TOPO-06, TOPO-07, REPRO-01, REPRO-02
**Success Criteria** (what must be TRUE):

  1. User can run `make up-colocated`, `make up-chunked`, `make up-disagg`, `make up-disagg-tier` and each serves OpenAI-compatible HTTP on cold node
  2. `make health` confirms P→D flow on disagg topologies via sentinel-token validity test (decode of known sentinel produces expected first-token logits); not gated solely on `kv_transfer_complete_count` increment
  3. Cold-node-to-serving takes <20 min for any single topology (`make provision` idempotent)
  4. vLLM pinned ≥0.11.x in `make provision` to mitigate CVE-2025-25183 (P2); NIXL backend pinned to UCX (not LIBFABRIC)
  5. All 4 topologies share common OpenAI-compatible request schema and `/metrics` endpoint

**Pitfalls Prevented**: P1 (NIXL LIBFABRIC silent garbage via UCX pin + sentinel test), P2 (CVE-2025-25183 via vLLM ≥0.11.x pin)
**Plans**: 7 plans
Plans:
**Wave 1**

- [ ] 01-01-PLAN.md — Project skeleton, Python tooling, Makefile command surface

**Wave 2** *(blocked on Wave 1 completion)*

- [ ] 01-03-PLAN.md — Single docker-compose file with four topology profiles and UCX/LMCache configs
- [ ] 01-04-PLAN.md — OpenAI-compatible disaggregated proxy for P/D profiles
- [ ] 01-05-PLAN.md — Three-layer sentinel-token validity validator and daemon

**Wave 3** *(blocked on Wave 2 completion)*

- [ ] 01-06-PLAN.md — Health gate for endpoint readiness, P→D metrics, and sentinel checks
- [ ] 01-07-PLAN.md — README quickstart plus runtime topology/schema tests

**Wave 4** *(blocked on Wave 3 completion)*

- [ ] 01-02-PLAN.md — Idempotent RunPod provisioning and model compatibility probe

### Phase 2: Load + Metrics (LOAD)

**Goal**: Load generator emits reproducible chat/RAG/agentic traces with per-request telemetry reconciled against vLLM's `/metrics` endpoint within ±2% per-30s window, plus Prometheus scrape and single TTFT metric definition.
**Depends on**: Phase 1
**Mode**: mvp
**Requirements**: LOAD-01, LOAD-02, LOAD-03, LOAD-04, LOAD-05, LOAD-06, LOAD-07, OBS-01
**Success Criteria** (what must be TRUE):

  1. User can run seed-controlled chat (multi-turn, 0.5-2K in / 50-500 out), RAG (8-32K in, ~80% prefix overlap), and agentic (bursty, high prefix overlap) trace generators with Poisson + ON/OFF open-loop arrivals
  2. Per-request log captures `enqueue_ts`, `ttft_ms`, `per_token_ts[]`, `completion_ts`, `status_code` with byte-identical replay given same seed
  3. Logged latencies reconcile with vLLM `/metrics` within ±2% on CDF @ p50/p95/p99, evaluated per 30s window (not per-run); `router_overhead_ms` reported as separate gap metric
  4. Single TTFT definition adopted: `engine_request_arrival_ts → first_token_emitted_ts`; client clock synced via NTP, `clock_skew_ms` recorded continuously
  5. Prometheus scrape endpoint live on router, prefill pool, decode pool

**Pitfalls Prevented**: P10 (metric reconciliation drift — per-30s reconciliation, not per-run)
**Plans**: TBD

### Phase 3: Router + Admission (RTR)

**Goal**: Cache-aware SLO router (SHA-256 prefix hash + per-pool salt, cache-first / load-tiebreak) with BATCH-shed admission control that holds INTERACTIVE TTFT p95 attainment ≥99% under 2× overload without dropping requests.
**Depends on**: Phase 2
**Mode**: mvp
**Requirements**: RTR-01, RTR-02, RTR-03, RTR-04, RTR-05, RTR-06, RTR-07
**Success Criteria** (what must be TRUE):

  1. Router exposes FastAPI HTTP front door; SLO classifier maps each request to INTERACTIVE / BATCH class from metadata or default heuristic
  2. Cache-aware routing policy: (1) cache affinity first via SHA-256 truncated 128-bit prefix hash per 256-token block with per-pool salt; (2) queued-token load as tiebreaker
  3. Admission control sheds BATCH when INTERACTIVE TTFT p95 attainment <99% over 30s window; INTERACTIVE SLO held under 2× overload; BATCH degrades gracefully; zero drops
  4. Prefix index hard-capped (TTL 1hr + LRU size cap); `prefix_index_size_bytes` metric exposed; alert > 1GB or > 10% router RSS
  5. No request drops under admission shedding (RTR-06); routing decision in-process O(log N), no network hop on hot path

**Pitfalls Prevented**: P2 (per-pool salt mitigation), P8 (prefix-index memory blowup via TTL + size cap)
**Plans**: TBD

### Phase 4: Router Verification (RTR-verify)

**Goal**: Cache-aware router claim A/B-verified with explicit cold-cache vs warm-cache regime isolation, preventing false confidence from cold-phase measurement bias.
**Depends on**: Phase 3
**Mode**: mvp
**Requirements**: RTR-04, RTR-07
**Success Criteria** (what must be TRUE):

  1. A/B test (cache-aware vs round-robin) at fixed load shows TTFT improvement on agentic trace in **steady-state warm regime** (post-prefill)
  2. **Cold-cache regime** (first-N requests, first 50 reqs) reported separately from warm; cache-aware router falls back to load-balancing when prefix hash has zero history (no matching prefill in last K min)
  3. `cache_aware_router_looked_up_no_history` counter distinguishes cold-cache from cache-miss
  4. INTERACTIVE TTFT p95 attainment ≥99% confirmed under 2× overload; zero drops

**Pitfalls Prevented**: P7 (cold-cache false confidence — dual-regime reporting)
**Plans**: TBD

### Phase 5: KV Tiering (KV)

**Goal**: LMCache KV tier (HBM → DRAM → disk) integrated with workload-appropriate eviction policy, per-SLO namespaces, and break-even chart documenting tier overhead vs benefit.
**Depends on**: Phase 3
**Mode**: mvp
**Requirements**: KV-01, KV-02, KV-03, KV-04, KV-05, KV-06
**Success Criteria** (what must be TRUE):

  1. LMCache integrated as shared KV tier (HBM → DRAM → disk) with per-SLO namespaces; `enable_pd` + `transfer_channel="nixl"` config verified against current LMCache docs
  2. Prefill outputs flow into LMCache; decode pulls from LMCache on cache miss; round-trip KV lookup confirmed for `make up-disagg-tier`
  3. Eviction policy measured per workload (LRU vs LFU); chosen policy + rationale documented; per-class LMCache budget enforced; hot prefixes prewarmed out of band (P9)
  4. Break-even chart plotted: benefit vs prefix-reuse rate AND HBM pressure; tiering overhead ≤5% TTFT when unpressured (cached lookup = direct)
  5. KV-stall failure drill (kill LMCache backend mid-request) written up as postmortem with recovery steps

**Pitfalls Prevented**: P9 (LMCache eviction policy mismatch — per-workload measurement + prewarm + per-class budget)
**Plans**: TBD

### Phase 6: Spec Decode (SPEC)

**Goal**: EAGLE-3 speculative decoding on decode pool with auto-disable circuit breaker at batch-size crossover, and spec × disagg KV interaction documented.
**Depends on**: Phase 5
**Mode**: mvp
**Requirements**: SPEC-01, SPEC-02, SPEC-03, SPEC-04, SPEC-05
**Success Criteria** (what must be TRUE):

  1. EAGLE-3 head loaded on decode pool (pre-trained, from HuggingFace); acceptance rate measured per workload
  2. Acceptance on disagg decode pool ≥0.85× of colocated same-model baseline (P3 gate); divergence >10% triggers abort / scope-out
  3. ITL vs batch-size curve plotted (SPEC-02); crossover point identified; auto-disable circuit breaker engages when ITL(spec) crosses below non-spec baseline (SPEC-03)
  4. Spec-decode × disagg KV interaction issues documented as commit notes (SPEC-04 = gate, not deliverable)
  5. Rejection-rate delta vs non-spec baseline logged per workload; auto-disable alarm fires if disabled >30% time

**Pitfalls Prevented**: P3 (spec × disagg draft KV uninit via acceptance-vs-colocated gate), P11 (spec acceptance collapse via ITL-vs-batch curve + auto-disable)
**Plans**: TBD

### Phase 7: Autoscaler (AUTO)

**Goal**: P:D autoscaler (PID on queue-depth pressure differential, two-tier gate, 120s minimum dwell, explicit drain protocol) holds SLO attainment during prompt-heavy→decode-heavy shifts without in-flight drops or thrashing.
**Depends on**: Phase 6
**Mode**: mvp
**Requirements**: AUTO-01, AUTO-02, AUTO-03, AUTO-04, AUTO-05, AUTO-06, AUTO-07
**Success Criteria** (what must be TRUE):

  1. PID-style controller on `(w1 × prefill_queue_pressure − w2 × decode_queue_pressure)` with two-tier gate: controller computes desired ratio each tick (1-5s); actuator only fires when desired ratio diverges > threshold AND last actuation ≥120s ago
  2. Explicit drain protocol on role flip: `accepting=false` → wait `inflight_count == 0` (or deadline) → coordinate peer pool drain handshake so remaining P→D KV transfer lands → signal worker to rejoin (P5)
  3. 120s minimum dwell (anti-thrash); `flip_count_per_minute` <0.5 sustained; `controller_thrash_detected` alarm fires on 2 flips within 240s
  4. SLO attainment with autoscaler ≥ static best-of-both-fixed-ratios on prompt-heavy → generation-heavy shift (AUTO-04); zero dropped in-flight requests during forced role flip (AUTO-05)
  5. Role-transition events logged to Prometheus (AUTO-06); rationale document committed explaining why NOT reactive per-request (AUTO-07); PID gains over-damped not under-damped; integral term anti-windup

**Pitfalls Prevented**: P5 (drain in-flight loss via explicit coord + 0-drop metric), P6 (autoscaler thrashing via two-tier gate + 120s dwell)
**Plans**: TBD

### Phase 8: Benchmark Campaign + Report (BENCH)

**Goal**: Capstone — full bench matrix produces goodput curves (chunked-prefill and disagg on same axes), TTFT/ITL CDFs, cost-per-million-tokens table, and three failure drills, with one-command reproduction and verified headline numbers. Merges OBS + REPRO.
**Depends on**: Phase 7
**Mode**: mvp
**Requirements**: BENCH-01, BENCH-02, BENCH-03, BENCH-04, BENCH-05, BENCH-06, BENCH-07, BENCH-08, BENCH-09, OBS-01, OBS-02, OBS-03, REPRO-01, REPRO-02, REPRO-03, REPRO-04, REPRO-05, REPRO-06
**Success Criteria** (what must be TRUE):

  1. `make bench` runs 4 topologies × 3 workloads × 6 load levels × 3 seeds = 216 cells from cold node in <20 min; idempotent
  2. Goodput curves plotted for each (topology × workload) with chunked-prefill and disagg on **same axes** at ≥2 cells straddling expected crossover (long-context RAG @ high load; short chat @ low load); crossover location = headline finding (P4)
  3. TTFT/ITL CDFs at the knee (point where SLO attainment <99%) committed; cost per million tokens table committed; hardware record (GPU model, vRAM, driver, CUDA, engine version, model + quant, seed, date) attached to every result file
  4. Failure appendix written as production postmortem: (1) kill decode mid-stream, (2) KV-transfer stall, (3) pathological RAG-burst-over-chat mix (P12); include recovery steps and prevention notes
  5. ≥3,000-word report "When disaggregation pays: an SLO-aware study" published in repo with honest chunked-vs-disagg finding; README headline table traceable to specific bench commit + seed; Grafana dashboard JSON committed (goodput, TTFT p95, ITL p95, queue depth per pool, KV-tier hit rate, spec acceptance rate, role-flip count, drain duration); ≥80% pytest coverage on `core/` and `control/` modules

**Pitfalls Prevented**: P4 (chunked crossover unmeasured — same-axes curves), P12 (pathological mix — Drill 3 in failure appendix)
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Topologies (TOPO) | 0/TBD | Not started | - |
| 2. Load + Metrics (LOAD) | 0/TBD | Not started | - |
| 3. Router + Admission (RTR) | 0/TBD | Not started | - |
| 4. Router Verification (RTR-verify) | 0/TBD | Not started | - |
| 5. KV Tiering (KV) | 0/TBD | Not started | - |
| 6. Spec Decode (SPEC) | 0/TBD | Not started | - |
| 7. Autoscaler (AUTO) | 0/TBD | Not started | - |
| 8. Benchmark Campaign (BENCH) | 0/TBD | Not started | - |

**Coverage:**

- v1 requirements: 50 total
- Mapped to phases: 50
- Unmapped: 0 ✓
- Pitfalls covered: 12 / 12 (P1–P12, all mapped to phases with prevention + verification)

---
*Roadmap created: 2026-07-08*
*Granularity: standard | Mode: mvp | Parallel: true*
