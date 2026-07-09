# Feature Landscape — GoodputLab

**Domain:** SLO-aware P/D disaggregated LLM inference serving
**Researched:** 2026-07-08
**Confidence:** HIGH (table stakes/differentiators from REQUIREMENTS.md + vLLM v1/LMCache/EAGLE-3 public docs); MEDIUM (anti-feature severity from PITFALLS.md empirical reproduction)

---

## Table Stakes (any LLM serving system; missing = product is broken)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| OpenAI-compat `/v1/chat/completions` + `/v1/completions` | All clients, evals, tools assume it | Low | TOPO-07 |
| SSE streaming responses | TTFT/ITL only measurable via stream | Low | LOAD-05 |
| `/metrics` Prometheus endpoint | Standard scrape target | Low | OBS-01 |
| Health/readiness probes | K8s/orchestrator gating | Low | TOPO-05 |
| Continuous batching (vLLM-native) | Default vLLM behavior, not optional | Low | PagedAttention scheduler |
| PagedAttention | KV block paging; required for any modern engine | Low | vLLM core |
| Per-token timestamp logging | TTFT/ITL reconciliation | Med | LOAD-05 |
| Load generator (per-token timestamps) | Self-test + bench driver | Med | LOAD-01..04 |
| TTFT/ITL measurement | The SLO primitives | Med | BENCH-02 |
| `make provision` cold-to-serving <20min | Reproducibility contract | Med | TOPO-06, REPRO-02 |
| Hardware/seed/version metadata per result | Audit trail | Low | BENCH-09 |

## Differentiators (GoodputLab-specific; staff-portfolio competitive edge)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Cache-aware prefix routing (256-tok block hash, per-pool, load tiebreaker) | TTFT win on agentic/RAG traces | Med | RTR-02..04; routes by prefix affinity, load as tiebreaker |
| SLO-class admission control (INTERACTIVE vs BATCH shedding) | Holds interactive SLO @ 2× overload | Med | RTR-05..07 |
| LMCache KV tier (HBM→DRAM→disk) w/ prewarming | Long-prefix cache amplification | High | KV-01..06 |
| P:D autoscaler (PID on queue-depth, 120s dwell, drain handshake) | Adapts to prompt-vs-decode phase shift | High | AUTO-01..07 |
| EAGLE-3 spec decode w/ auto-disable threshold | ITL speedup @ low batch | High | SPEC-01..05 |
| **Goodput as primary metric** (throughput × SLO attainment) | The thesis; honest frontier metric | Low | DECISIONS table row 1 |
| Per-30s metric reconciliation (not per-run) | Detects drift at the bench knee | Med | LOAD-06 + P10 |
| Cold-cache vs warm-cache regime separation | Honest router A/B; no false confidence | Low | P7, RTR-04 split |
| Chunked-prefill vs disagg crossover measurement | Honest boundary finding (staff move) | High | P4, BENCH-03 |
| 4-topology matrix (colocated/chunked/disagg/disagg-tier) | Apples-to-apples frontier map | High | TOPO-01..04 |
| Pathological-mix failure drill (RAG-burst-over-chat) | Documents LMCache eviction cascade | Med | P12, BENCH-06 |

## Anti-Features (commonly requested, dangerous in this domain)

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Per-request autoscaling | Thrashing; loses 15-30s capacity per flip | Two-tier gate (controller + rate-limited actuator), 120s dwell (P6, AUTO-03) |
| Unbounded prefix index | RSS blowup at 24hr long-tail chat (P8) | TTL 1hr + LRU size cap + SHA-256 keys |
| Python `hash()` for prefix keys | CVE-2025-25183 cross-pool cache collision / leakage | SHA-256 truncated 128 bits, per-pool salt |
| LIBFABRIC NIXL backend default | Silent garbage output (vLLM #27055, P1) | Pin UCX; sentinel-token validity test |
| Naive round-robin under heterogeneous prefix load | Wastes cache affinity on agentic/RAG | Cache-first, load-second (RTR-03) |
| Static spec-decode batch cap | Crossover moves with workload (P11) | Auto-disable on ITL-vs-baseline inversion (SPEC-03) |
| In-flight drops during role flip | SIGTERM hard-cuts P→D KV edge (P5) | Explicit drain handshake (AUTO-02, AUTO-05 gate) |
| Engine-clock TTFT as sole ground truth | Hides router overhead, load-dependent drift (P10) | Reconcile vs client wall-clock w/ `router_overhead_ms` metric |
| Default LMCache LRU all workloads | Evicts hot system prompts under churn (P9) | Pick eviction per workload; prewarm; per-class budget |
| Blending cold + warm TTFT in A/B | False "cache-aware wins" claim (P7) | Split cold (first-N) and warm regimes; report separately |

## Feature Dependencies

```
prefix-hash (RTR-02) → cache-aware routing (RTR-03) → admission control (RTR-05) → autoscaler (AUTO-01)
LMCache tier (KV-01) → break-even chart (KV-04) → eviction policy choice (KV-03)
EAGLE-3 (SPEC-01) → ITL-vs-batch curve (SPEC-02) → auto-disable (SPEC-03)
load gen (LOAD-01..04) → reconciliation (LOAD-06) → goodput curves (BENCH-03)
4 topologies (TOPO-01..04) → benchmark matrix (BENCH-01) → chunked-vs-disagg crossover (P4)
```

Each layer's output is the next layer's gate; skipping order = silent failures (P1, P5, P10).

## MVP Definition (v1 = W1-W8 deliverables per REQUIREMENTS.md)

**Ship (v1 must-have):**
- TOPO-01..07 (4 topologies, health checks, <20min cold-start)
- LOAD-01..07 (3 traces, open-loop, per-30s reconciliation)
- RTR-01..07 (SLO classifier, prefix hash, cache-aware, admission, FastAPI)
- KV-01..06 (LMCache, break-even curve, eviction policy, stall drill)
- SPEC-01..05 (EAGLE-3, ITL curve, auto-disable, disagg interaction notes)
- AUTO-01..07 (PID + drain + 120s dwell, zero in-flight drops)
- BENCH-01..09 (full matrix, CDFs, cost/1M tokens, failure appendix)
- OBS-01..03 (Prometheus + Grafana)
- REPRO-01..06 (docker-compose, make provision, ≥80% coverage on core/control)
- ≥3,000-word report "When disaggregation pays: an SLO-aware study"

**Defer to v2 (post-v1 stretch):**
- MULTI-01..02 (multi-node P/D pools, topology-aware allocator)
- ADV-01..02 (multi-model routing, cost-aware admission)
- LRN-01..02 (online prefix-tree learning, workload shift detection)
- One upstream PR to vLLM/LMCache (stretch deliverable)

**Explicitly out of scope (per PROJECT.md):** training weights, frontend UI, multi-tenant auth, K8s HPA, LMCache prefix-prewarming of specific distributions, cross-region failover, production observability hardening, SGLang deep integration, mobile/edge.

---

## Sources

- REQUIREMENTS.md (v1 requirements 50 total, all mapped to phases)
- PROJECT.md (architecture, key decisions, evolution policy)
- PITFALLS.md P1-P12 (anti-feature justification, severity ranking)
- vLLM v0.11+ NixlConnector docs (cache-aware routing + KV transfer primitives)
- LMCache architecture docs + issue #649 (KV tiering, eviction policy gap)
- EAGLE-3 (sglang #19796 reproducer; spec × disagg KV layout)
- Mooncake FAST 2025 paper, DistServe OSDI 2024, Splitwise ICML 2024, Sarathi-Serve EuroSys 2025 (chunked-vs-disagg crossover)
- CVE-2025-25183 (Miggo; prefix hash collision)