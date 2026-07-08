# Requirements: GoodputLab

**Defined:** 2026-07-08
**Core Value:** Goodput (throughput × SLO attainment) under realistic mixed workloads, with verified reproducible numbers and a public artifact trail.

## v1 Requirements

### Topologies (TOPO)

- [ ] **TOPO-01**: Colocated serving (single vLLM pool, no disagg) deploys via `make up-colocated` and serves OpenAI-compatible HTTP
- [ ] **TOPO-02**: Chunked-prefill topology (single vLLM pool with `--enable-chunked-prefill`) deploys via `make up-chunked` and serves requests
- [ ] **TOPO-03**: P/D disaggregated topology (separate prefill + decode pools, NIXL KV transfer) deploys via `make up-disagg` and P→D KV transfer is confirmed via vLLM metrics
- [ ] **TOPO-04**: Disaggregated + LMCache tiering topology deploys via `make up-disagg-tier` and round-trip KV lookup hits the tier
- [ ] **TOPO-05**: Health-check script (`make health`) confirms P→D flow on disagg topologies and decode never runs prefill (per vLLM metrics)
- [ ] **TOPO-06**: Cold-node-to-serving <20 min for any single topology
- [ ] **TOPO-07**: All 4 topologies share a common OpenAI-compatible request schema and metrics endpoint

### Load Generation (LOAD)

- [ ] **LOAD-01**: Chat trace generator (multi-turn, shared system prompt, 0.5-2K input tokens, 50-500 output tokens, seedable)
- [ ] **LOAD-02**: RAG trace generator (8-32K input tokens, short output, ~80% prefix-overlap)
- [ ] **LOAD-03**: Agentic trace generator (bursty, high prefix overlap across calls)
- [ ] **LOAD-04**: Open-loop arrival process (Poisson + ON/OFF bursts) per trace
- [ ] **LOAD-05**: Per-request log: enqueue_ts, ttft_ms, per_token_ts[], completion_ts, status_code
- [ ] **LOAD-06**: Logged latencies reconcile with vLLM's `/metrics` endpoint within ±2%
- [ ] **LOAD-07**: Trace replay is byte-identical given same seed

### Routing (RTR)

- [ ] **RTR-01**: SLO classifier maps each request to INTERACTIVE / BATCH class from request metadata or default heuristic
- [ ] **RTR-02**: Prefix index: rolling hash per 256-token block, per-worker matched-block lookup table
- [ ] **RTR-03**: Routing policy: (1) cache affinity first; (2) queued-token load as tiebreaker
- [ ] **RTR-04**: Cache-aware router improves TTFT p95 vs round-robin on agentic trace at fixed load (A/B verified)
- [ ] **RTR-05**: Admission control: sheds BATCH when INTERACTIVE TTFT p95 attainment <99% over 30s window
- [ ] **RTR-06**: Router exposes FastAPI HTTP front door; no request drops under admission shedding
- [ ] **RTR-07**: Hold INTERACTIVE SLO under 2× overload; BATCH degrades gracefully

### KV Tiering (KV)

- [ ] **KV-01**: LMCache integrated as shared KV tier (HBM → DRAM → disk)
- [ ] **KV-02**: Prefill outputs flow into LMCache; decode pulls from LMCache on cache miss
- [ ] **KV-03**: Eviction policy tested (LRU vs LFU) and documented
- [ ] **KV-04**: Break-even chart plotted: benefit vs prefix-reuse rate and HBM pressure
- [ ] **KV-05**: Tiering overhead measured ≤5% TTFT when unpressured (cached lookup = direct)
- [ ] **KV-06**: KV-stall failure drill written up as postmortem

### Speculative Decoding (SPEC)

- [ ] **SPEC-01**: EAGLE-3 speculative decoding head loaded on decode pool (pre-trained, from HuggingFace)
- [ ] **SPEC-02**: Spec decode ITL vs batch-size curve plotted; crossover point identified
- [ ] **SPEC-03**: Auto-disable spec decode above the batch-size threshold
- [ ] **SPEC-04**: Spec-decode × disagg KV interaction issues documented (commit notes)
- [ ] **SPEC-05**: Acceptance rate measured per workload; rejection-rate delta vs non-spec logged

### Autoscaling (AUTO)

- [ ] **AUTO-01**: P:D controller (PID-style) on (w1×prefill_queue_pressure − w2×decode_queue_pressure)
- [ ] **AUTO-02**: Flex worker role-flipping with drain protocol: stop admissions → finish in-flight → rejoin
- [ ] **AUTO-03**: 120s minimum dwell on any role assignment (anti-thrash)
- [ ] **AUTO-04**: SLO attainment with autoscaler ≥ static best-of-both-fixed-ratios on prompt-heavy→generation-heavy shift
- [ ] **AUTO-05**: Zero dropped in-flight requests during role flips
- [ ] **AUTO-06**: Role-transition events logged to Prometheus
- [ ] **AUTO-07**: Documented: why not reactive per-request (thrashing analysis)

### Benchmarking (BENCH)

- [ ] **BENCH-01**: `make bench` runs full matrix: 4 topologies × 3 workloads × 6 load levels × 3 seeds = 216 cells
- [ ] **BENCH-02**: Metrics captured: goodput (req/s with ≥99% SLO), TTFT p50/p95/p99, ITL p50/p95/p99, cost/1M tokens
- [ ] **BENCH-03**: Goodput curves plotted for each (topology × workload); figures committed
- [ ] **BENCH-04**: TTFT/ITL CDFs at the knee (point where SLO attainment falls below 99%) committed
- [ ] **BENCH-05**: Cost per million tokens table committed
- [ ] **BENCH-06**: Failure appendix: kill decode mid-stream, KV-transfer stall, pathological mix (committed as postmortem)
- [ ] **BENCH-07**: Report (≥3,000 words) "When disaggregation pays: an SLO-aware study" published in repo
- [ ] **BENCH-08**: One-command `make bench` reproduces every README number from cold node in <20 min
- [ ] **BENCH-09**: Hardware record (GPU model, vRAM, driver, CUDA, engine version, model + quant, seed, date) attached to every result file

### Observability (OBS)

- [ ] **OBS-01**: Prometheus scrape endpoint on router, prefill pool, decode pool
- [ ] **OBS-02**: Grafana dashboard JSON committed: goodput, TTFT p95, ITL p95, queue depth per pool, KV-tier hit rate
- [ ] **OBS-03**: Spec-decode acceptance rate, role-flip count, drain duration in dashboard

### Reproducibility (REPRO)

- [ ] **REPRO-01**: docker-compose files for all 4 topologies committed
- [ ] **REPRO-02**: `make provision` provisions a bare GPU node → healthy serving in <20 min
- [ ] **REPRO-03**: All bench results stored as parquet + metadata JSON (HW, seed, version)
- [ ] **REPRO-04**: ≥80% pytest coverage on `core/` and `control/` modules (router, autoscaler, admission, load gen)
- [ ] **REPRO-05**: pyproject.toml + ruff + mypy config committed; `make lint` passes
- [ ] **REPRO-06**: README with measured-headline table traceable to specific bench commit + seed

## v2 Requirements

Deferred to post-v1 if milestone scope expands.

### Multi-node

- **MULTI-01**: Multi-node P/D pools (prefill and decode on separate physical nodes) with NIXL cross-node
- **MULTI-02**: Topology-aware allocator (GPU-NVLink vs IB vs Ethernet)

### Advanced Routing

- **ADV-01**: Multi-model routing (route by model name across heterogeneous pools)
- **ADV-02**: Cost-aware admission (optimize $/request rather than SLO attainment)

### Learning

- **LRN-01**: Online prefix-tree learning (auto-promote hot prefixes to dedicated prefill workers)
- **LRN-02**: Workload shift detection (auto-retrain trace mix)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Training new model weights | This is serving, not pretraining |
| Frontend chat UI | We expose OpenAI-compatible HTTP only; no UI work |
| Multi-tenant auth / billing | Single-tenant benchmarking rig; not a product |
| Cloud-managed autoscaler (K8s HPA, etc.) | We ship our own PID controller as the artifact |
| LMCache prefix-prewarming | Out of band; we measure cold and warm tiers |
| Cross-region failover | Single-region bench; multi-region is operational, not algorithmic |
| Production hardening of observability stack | Basic Prometheus/Grafana only |
| Mobile / edge inference | Far from the frontier we're studying |
| SGLang deep integration | vLLM v1 is primary; SGLang noted as fallback only |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| TOPO-01..07 | Phase 1 (Topologies) | Pending |
| LOAD-01..07 | Phase 2 (Load + Metrics) | Pending |
| RTR-01..07 | Phase 3 (Router + Admission) | Pending |
| RTR-04, RTR-07 | Phase 4 (Router Verification) | Pending |
| KV-01..06 | Phase 5 (KV Tiering) | Pending |
| SPEC-01..05 | Phase 6 (Spec Decode) | Pending |
| AUTO-01..07 | Phase 7 (Autoscaler) | Pending |
| BENCH-01..09 | Phase 8 (Benchmark Campaign) | Pending |
| OBS-01..03 | Phase 8 (Benchmark Campaign) | Pending |
| REPRO-01..06 | Phase 1 + Phase 8 (cross-cutting) | Pending |

**Coverage:**
- v1 requirements: 50 total
- Mapped to phases: 50
- Unmapped: 0 ✓

---
*Requirements defined: 2026-07-08*
*Last updated: 2026-07-08 after initialization*