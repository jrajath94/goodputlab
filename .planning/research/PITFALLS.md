# Pitfalls Research — GoodputLab

**Domain:** LLM inference serving — P/D disaggregation, SLO-aware routing, KV tiering, autoscaling, spec decode
**Researched:** 2026-07-08
**Confidence:** MEDIUM-HIGH (vLLM/LMCache/Mooncake ecosystem); MEDIUM (autoscaler/LMCache remaining bugs)

---

## Executive Summary

P/D disagg = right control plane to study. Sits on 4 fragile layers: (1) vLLM async engine + KVConnector handshakes, (2) NIXL transfer backends (silent corruption = known production hazard), (3) LMCache eviction/tiering (semantics change between releases), (4) EAGLE-3 spec decode (draft KV not first-class in disagg path). Most dangerous failures silent: garbage output, data leakage via cache collision, metric drift masking overload. Honest chunked-vs-disagg crossover finding IS itself a pitfall — naive benches hide it.

12 pitfalls ordered by fail-severity. Each maps to phase + detection signal + prevention.

---

## Critical Pitfalls

### P1: NIXL KV-transfer silent garbage output (LIBFABRIC backend)

**Wrong:** Disagg prefill → garbage output. Logs clean. KV dump = empty. No signal until users complain.

**Why:** vLLM #27055 — confirmed bug on LIBFABRIC backend w/ vLLM 0.11.0 + NIXL 0.6.1. UCX unaffected. Standalone NIXL LIBFABRIC tests pass; only vLLM connector path fails. KVBuffer negotiation mismatches silently. K8s RDMA adds 2nd failure mode (NIC isolation, MR registration race).

**Avoid:**
- Pin `NixlConnector` config to `backends=["UCX"]` for phase-1; switch to LIBFABRIC only after checksum test passes.
- Post-transfer validity check: send known-token sentinel, verify decode produces expected first-token logits (L2 < ε).
- Hash incoming KV block hashes vs prefill-emitted block hashes (acceptance test).
- `--kv-transfer-timeout` w/ abort-and-retry on timeout, not silent continuation.
- Verify vLLM docs Day-1 of each topology session — NIXL semantics change between minors (0.5 → 0.6 broke path).

**Warning:** Identical-prefix requests → divergent outputs. `kv_transfer_complete_count` ↑ but `kv_transfer_inflight` does not ↓. Output logprob entropy unusually high. Token count matches, content incoherent.

**Phase:** Phase 1 (TOPO). Severity: CRITICAL — silent corruption.

### P2: CVE-2025-25183 prefix-cache hash collision → cross-user data leakage

**Wrong:** Multitenant prefix cache reuses blocks from other tenants via crafted collision. Py 3.12+ `hash(None)` predictable → attacker constructs colliding prompts.

**Why:** Miggo CVE-2025-25183 — vulnerable: `content_hash`, `hash_block_tokens`, `ComputedBlocksTracker._update_seq_hashes`. Computed `None`'s hash directly. Patched in vLLM 0.7.2 → `hash('None')`. Single-tenant GoodputLab lower-risk, but cache-aware router REPLICATES hazard across pools (poisoned prefill-A prefix matched against diff request at decode-B).

**Avoid:**
- vLLM ≥ 0.7.2 minimum; pin v0.11.x+ primary.
- Document single-tenant assumption in router reqs; do NOT ship router as multitenant w/o (a) per-tenant prefix-hash salt + (b) crypto hash (SHA-256) not Python `hash()`.
- Unit test: prefix-block-hash for benign prefix ≠ deliberately colliding prefix.

**Warning:** `cache_hit=True` for prefix never seen. Identical-prefix decode-side requests → content disagrees w/ original prefill replay.

**Phase:** Phase 1 (vLLM version pin) + Phase 3 (router salt per pool). Severity: CRITICAL multi-tenant; MEDIUM for GoodputLab — teaching point in failure appendix.

### P3: Spec-decode × disagg KV interaction (draft model KV uninitialized)

**Wrong:** EAGLE-3 on decode pool of disagg → draft model's KV cache slots (should be populated by transferred P→D blocks) = uninitialized memory in draft model. NaN logits, crashes, or silently degraded draft quality → fails verification more than offline-acceptance predicts.

**Why:** sglang #19796 root cause: target populates positions, draft's parallel cache lane sees raw allocated memory. EAGLE assumes draft shares structure w/ target. Disagg transfer writes only target's KV layout; draft's parallel structure must be (a) separately transferred, (b) reconstructed by draft-side prefill, or (c) bypassed via shared RadixAttention (SGLang model, not vLLM-native). TensorRT-LLM solves w/ explicit pre-scheduling "stage" reconciling KV layout across draft+target.

**Avoid:**
- Do NOT naively enable EAGLE-3 on disagg decode pool w/o controlled experiment measuring acceptance vs non-spec baseline.
- Pin acceptance-rate measurement per workload (SPEC-05). Divergence from offline = leading indicator.
- Auto-disable threshold (SPEC-03) + human review before re-enable after crash.
- EAGLE-3 acceptance differs >10% between colocated baseline + disagg decode pool → abort Phase 6, either (a) wire draft into KV transfer path or (b) scope EAGLE-3 out.

**Warning:** EAGLE-3 acceptance (disagg) < acceptance (colocated same-model baseline). NaN in draft logits in vLLM logs (search `nan` in `EngineCore`). Spec-decode ITL slower than non-spec at low batch (spec point inverted).

**Phase:** Phase 6 (SPEC). SPEC-04 documentation = gate, not deliverable. Severity: HIGH.

### P4: Chunked-prefill vs disagg crossover not measured → wrong headline

**Wrong:** "Disagg wins everywhere" or "chunked always wins" overclaimed. Honest boundary = workload-shape dependent. Chunked-prefill wins short contexts + latency-sensitive; disagg wins long contexts + decode-heavy. Reporting one w/o measuring other → wrong staff-portfolio claim.

**Why:** DistServe, Splitwise, Mooncake, Sarathi-Serve all show chunked vs disagg crossover at different prompt-length/batch-size cells. Most prod cherry-picks load shape favoring their arch. GoodputLab honest-finding mandate requires both at crossover.

**Avoid:**
- Bench matrix extends to ≥2 cells straddling expected crossover (long-context RAG @ high load, short chat @ low load).
- Plot *goodput curves w/ disagg + chunked-prefill on same axes*; crossover = headline.
- If chunked prefill wins all measured cells → that's honest finding. Report as-is. Do not retroactively tune disagg.

**Warning:** Disagg goodput curve has no measured points where chunked-prefill above. "Disagg wins" @ low context (<2k tokens) — known to favor chunked. No SMALL batches @ LONG context — disagg overhead dominates.

**Phase:** Phase 8 (BENCH). Severity: HIGH (report credibility); MEDIUM (correctness).

### P5: P:D autoscaler drain-protocol request loss

**Wrong:** Role flip → in-flight requests attached to re-purposed worker dropped. Clients see connection-reset, partial tokens, 503s. "Drain protocol" (stop admissions → finish in-flight → rejoin) broken because KV transfers in flight at SIGTERM time abandoned vs allowed to complete.

**Why:** vLLM graceful-shutdown RFC #24885 — "for disagg deployments (NIXL prefill/decode), KVConnector.shutdown() should allow deferring shutdown until pending KV transfers complete or time out." Specified behavior, not impl. Actual KVConnector.shutdown() in current vLLM does not always block on in-flight KV transfers. W/o explicit drain coord, SIGTERM during role flip hard-cuts prefill→decode edge.

**Avoid:**
- Explicit `drain(role_flip=True)` controller:
  1. `accepting=false` on router (no new reqs to this pool).
  2. Wait for `inflight_count == 0` OR deadline.
  3. Coordinate peer pool's drain handshake so remaining P→D KV transfer lands.
  4. After deadline, signal worker to rejoin w/ new role.
- Metric `role_flip_inflight_dropped` (must = 0 for AUTO-05).
- 120s min dwell (AUTO-03) bounds flip freq.
- Failure drill (BENCH-06): force flip mid-flight, document recovery.

**Warning:** `connection reset by peer` correlated w/ role-flip events. `kv_transfer_inflight` does not return to 0 after flip. Decode pool shows `decode_inflight_count > 0` after `accepting=false` set on flipped pool.

**Phase:** Phase 7 (AUTO). AUTO-05 = gate. Severity: HIGH.

### P6: Autoscaler thrashing from per-request control loop

**Wrong:** PID per-request oscillates "scale up P" / "scale down P" faster than reqs complete. Each flip loses 1 worker capacity (15-30s drain + warm-cache repopulation). Net throughput < static best ratio.

**Why:** Standard PID controllers have no rate limit on actuator. K8s HPA + KEDA expose `stabilizationWindowSeconds`/`cooldownPeriod` (default 300s) because reactive per-request scaling = known failure mode. 120s min dwell (AUTO-03) correct ONLY if actuator also rate-limits. W/o rate limit, dwell check per-attempt not per-actuation.

**Avoid:**
- PID controller w/ two-tier gate: (1) controller computes desired ratio each tick (1-5s); (2) actuator only fires if desired ratio diverges from current > threshold AND last actuation ≥120s ago.
- Document why 120s (not 60s/300s) — model warm-cache repop + drain time on smallest target worker.
- Tune PID gains to over-damp not under-damp (slow controller that occasionally misses transition beats fast one that thrashes).
- `controller_thrash_detected` boolean: 2 flips within 240s → log + force extra 120s dwell on next flip.

**Warning:** `role_flip_count_per_minute` > 0.5 sustained. SLO attainment w/ autoscaler < static best-of-both-fixed-ratios (fails AUTO-04). Worker `cache_hit_rate` drops correlated w/ flips (cache lost during role flip).

**Phase:** Phase 7 (AUTO). AUTO-04 acceptance = gate. Severity: HIGH.

### P7: Cache-aware routing false confidence on cold cache

**Wrong:** A/B cache-aware vs round-robin shows "improvement" measured during warm-cache steady state, but cold-cache TTFT (cache_hit_rate = 0%) is *worse* for cache-aware (each request's prefix-hash lookup adds latency — Redis hop, hash compute — before routing). Cold-cache measurement window biases comparison.

**Why:** vLLM prefix cache fill incremental — first req primes, later hit. Standard A/B starts after warm-up, but router w/ sub-ms lookup (Redis hop or in-process) still adds to TTFT p50 in cold phase. "Cache-aware wins" claim valid in steady state only. RTR-04 says improvement at fixed load = steady-state, cold phase reported separately.

**Avoid:**
- Two regimes: cold-cache (first-N reqs) and warm-cache (post-prefill). Report both.
- Router prefix-hash lookup = in-process O(log N) or bounded — no network hops on hot path.
- Cache-aware router falls back to load-balancing when prefix hash has zero history (no matching prefill in last K min).
- `cache_aware_router_looked_up_no_history` counter distinguishes cold from miss.

**Warning:** A/B first 50 reqs shows cache-aware > round-robin (impossible; cache empty). Cold-cache TTFT p95 > warm-cache TTFT p95 by >10% in cache-aware but not round-robin.

**Phase:** Phase 4 (Router Verification). RTR-04 measured steady state; cold phase reported separately. Severity: MEDIUM.

### P8: Prefix-index memory blowup on long-tail traffic

**Wrong:** Router prefix index grows unboundedly on chat-like workloads w/ long-tail unique prefixes. Each req consumes ~32B (block hash) + lookup entry. @10k QPS w/ 0.5% unique prefixes → index grows ~50MB/hr, never reclaimed. After 1 day → disk-swap territory, lookup latency explodes.

**Why:** Bounded eviction requires (a) TTL or (b) LRU w/ size cap. Naive impl uses `dict`, never evicts. Even w/ LRU, bucket structure can fragment.

**Avoid:**
- Hard cap: TTL 1hr on prefix index entries; LRU w/ size cap.
- Compress block-hash keys w/ crypto hash to fixed-size (SHA-256 truncated 128 bits), not Python `hash()` (CVE-2025-25183).
- Bound parse+match work per req (e.g., only first 4KB tokens for routing).
- Metric `prefix_index_size_bytes`; alert > 1GB or > 10% router RSS.

**Warning:** Router RSS grows monotonically over 24hrs. TTFT p95 climbs w/ RSS.

**Phase:** Phase 3 (RTR). Severity: MEDIUM — slow failure, 24hr drift.

### P9: LMCache eviction policy wrong for workload shape

**Wrong:** Default LRU under high churn evicts hot prefixes (early arrivals evicted before late arrivals reuse) → cache stampede / thundering herd. Agentic w/ hot system prompts + many one-shot suffix prefixes → LRU keeps evicting hot system prompt.

**Why:** LMCache defaults to LRU. LMCache eviction issue #649 (P0, stale) flags remote-server eviction buggy. Open-source LMCache does not yet ship LFU or ARC; KVCache compression research notes most "smart" eviction regresses in practice vs LRU when working set > capacity.

**Avoid:**
- Measure prefix-reuse distribution per workload FIRST, then pick eviction (LRU high-churn, LFU hot-system-prompt).
- Pre-warm known hot prefixes out of band (LMCache prewarming); do not rely on cold-fill under burst.
- `cache_eviction_count_per_minute` metric; correlate w/ TTFT spikes.
- GoodputLab bench: document chosen policy, reproduce LMCache #649 if persists → clean upstream PR.

**Warning:** Hot prefixes (system prompts 90% reuse) get evicted under burst. TTFT p99 spikes when prefill queue depth > threshold (cache miss cascade). LMCache remote-server eviction logs show race / lost updates (#649).

**Phase:** Phase 5 (KV). KV-03 = gate. Severity: MEDIUM.

### P10: Metric reconciliation drift — client TTFT ≠ vLLM `/metrics`

**Wrong:** Load gen logs TTFT=350ms; vLLM Prometheus `time_to_first_token_seconds{quantile="0.5"} = 0.320`. 9% drift — fails LOAD-06 (≤2%). Root causes: (a) client-side timer starts before req sent to engine (POST /v1/chat/completions parse latency included), (b) vLLM "TTFT" measured from `request_arrival` in engine core, not router arrival, (c) streaming first-byte vs first-token boundary, (d) batching effect — 2 reqs same batch get same engine time but diff wall-clock TTFT.

**Why:** Client-side measurement always includes network + queue-wait + parse overhead; engine-side may exclude queue-wait (engine-internal clock starts @ scheduling). Reconciling requires (a) common clock reference (NTP / carrier-grade sync) and (b) explicit accounting for what each timer includes.

**Avoid:**
- SINGLE metric definition: **TTFT = `engine_request_arrival_ts → first_token_emitted_ts`, measured both client and server w/ same boundary**; report *gap* as separate metric (`router_overhead_ms`).
- Prometheus histogram `vllm:request_success_total{...}` + `vllm:e2e_request_latency_seconds`; reconcile vs per-request logs by request_id.
- Client clock synced w/ engine clock via PTP/NTP; record `clock_skew_ms` continuously.
- Reconcile by histogram CDF, not point estimate — ≤2% deviation in CDF @ p50, p95, p99.

**Warning:** Client and server TTFT p95 diverge >2% sustained. Reconciliation passes low load, fails high load (load-dependent bias).

**Phase:** Phase 2 (LOAD). LOAD-06 = gate. Severity: MEDIUM — every bench result depends on this.

### P11: Spec-decode acceptance collapse above batch-size threshold

**Wrong:** EAGLE-3 → 2.5× speedup @ batch=4. @ batch=64, acceptance collapses (draft proposals diverge across seqs; verifier rejects more; effective wall-clock speedup < 1×). Net: spec decode = overhead, not acceleration.

**Why:** EAGLE draft single-sequence-trained; multi-seq verification in batch drops joint acceptance rate as seqs diverge. Crossover model-specific (training data + draft head quality), typically 16-64 batch.

**Avoid:**
- Plot **ITL vs batch-size** curve (SPEC-02) explicitly. Identify crossover.
- Auto-disable (SPEC-03): circuit breaker that disables EAGLE-3 the moment ITL crosses below non-spec baseline. Log role transition.
- Do NOT paper over crossover w/ static batch cap — changes w/ workload.

**Warning:** ITL(spec) > ITL(non-spec) at some batch — disqualifying. Acceptance rate < 0.4 at high batch.

**Phase:** Phase 6. SPEC-02 = crossover plot; SPEC-03 = auto-disable. Severity: MEDIUM.

### P12: Pathological workload mixes (multiturn chat + RAG burst collision)

**Wrong:** Router prefix-hash index assumes stable prefix distribution. RAG burst (8-32k long context, ~80% prefix overlap) lands on existing chat workload (0.5-2k context). RAG burst *trashes chat prefixes from LMCache* (cache eviction race), then chat TTFT spikes for next 30-60s while chat prefixes re-cached.

**Why:** LMCache capacity shared; eviction workload-agnostic. RAG tokens arrive en masse → evict chat prefixes regardless of frequency. Combined w/ chunked-prefill or disagg, eviction cascades: prefill workers lose chat-prefix cache; decode workers request KV transfer; KV transfer queue fills; prefill stalls.

**Avoid:**
- Budget LMCache per SLO class (fixed ratio disk cache to chat, RAG, agentic).
- Per-SLO-class eviction priority in LMCache config (if available) or simulate via prefix-hash-keyed namespaces.
- Test in Phase 8 failure drills (BENCH-06) — third drill listed.

**Warning:** After RAG burst, chat TTFT p99 climbs >30s. LMCache eviction logs skew toward chat prefixes during RAG bursts.

**Phase:** Phase 8 (BENCH + drills). BENCH-06 explicit. Severity: MEDIUM — well-defined but rarely tested; failure appendix material.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Skip post-KV-transfer validity check (send token, decode, compare) | Faster deploy | Silent garbage ships to "successful" benches | Never — 30-line sanity test |
| Python `hash()` for prefix indexing | No dep, fast | CVE-2025-25183 class risk; collisions yield cross-pool cache reuse | Never; use SHA-256 truncated 128 bits |
| Auto-disable spec decode via static batch cap | Simple | Cap becomes too high (overhead) or too low (no benefit some cells) | Never prod; MVP-only |
| Run role flips w/o drain handshake | Simulator faster | In-flight loss makes AUTO-05 unverifiable | Never |
| Per-request PID setpoint updates | Tighter control | Thrashing; drains 30s capacity per flip | Never — accept slower controller |
| Skip prewarming LMCache on cold bench | Fast first boot | First-N reqs 5-10× steady-state TTFT; first-N metrics noise | Acceptable cold-test runs only |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| NIXL ↔ vLLM | Default LIBFABRIC backend | Pin UCX phase-1; checksum-validate before LIBFABRIC |
| NIXL ↔ K8s | Assume RDMA works in container | Verify `device-plugin`, `CNIs` (Multus), `--cap-add=IPC_LOCK` per K8s RDMA writeup |
| LMCache ↔ vLLM | Default LRU all workloads | Pick per workload; prewarm hot prefixes; per-class budget |
| EAGLE-3 ↔ vLLM decode pool | Naively enable w/o disagg test | Measure acceptance on decode pool vs colocated baseline; SPEC-04 gate |
| vLLM ↔ Prometheus | Trust engine TTFT as ground truth | Reconcile vs client wall-clock w/ explicit gap metric |
| vLLM graceful shutdown ↔ role flip | Use vLLM default SIGTERM for role flips | Custom drain coordinator blocks until KV transfers complete |
| Prefix cache ↔ Router | Hash w/ Python `hash()` | Crypto hash w/ per-pool salt |
| LMCache ↔ Mooncake transfer engine | Treat as interchangeable | Verify schema/version compat — LMCache own store; Mooncake separate system |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Disagg KV transfer bottleneck | `kv_transfer_inflight` saturates; prefill outpaces decode | Pin `--kv-buffer-size`; measure transfer bandwidth @ expected prompt size | Prompt >16k tokens AND >2 inflight/worker |
| LMCache disk-tier latency | First-miss 100ms+ from disk | Pin hot working set to DRAM; document disk = cold cache | Working set > DRAM tier |
| Spec-decode draft = bottleneck | Acceptance high but ITL doesn't drop | Re-batch draft gen; check draft tokens not serialized w/ target verifier | Decode batch >32 |
| Router prefix-lookup = bottleneck | Each lookup = Redis hop | In-process LRU + bloom filter; Redis only for cross-instance cache sharing | >5k QPS/router instance |
| LMCache eviction → cache stampede | Many reqs miss simultaneously when hot key TTLs | Lua-style early refresh; per-class budget; jitter TTL | All hot keys share TTL |
| PID controller oscillates | SLO attainment swings 30% peak-peak | Two-tier gate (controller + rate-limited actuator); 120s dwell at actuator | Control interval < drain+warm time |

---

## Security Mistakes (single-tenant context)

| Mistake | Risk | Prevention |
|---------|------|------------|
| Prefix cache enabled in multitenant w/o per-tenant salt | Cross-tenant prompt injection / leakage (CVE-2025-25183) | Salt per tenant + crypto hash |
| Expose `/metrics` w/o auth | Internal cluster topology fingerprintable; query volume hints | Bind internal interface or auth proxy |
| Log full req prompts in shared log volume | Sensitive data at rest | Hash + truncate in logs; redact system prompt segments |

---

## "Looks Done But Isn't" Checklist

- [ ] **NIXL connector healthy**: Often missing **validity check** (sentinel token test) — verify KV hashes match after every transfer, not just count incremented
- [ ] **Spec decode enabled**: Often missing **acceptance-rate measurement in disagg** vs colocated baseline — verify decode-pool EAGLE-3 acceptance ≥ offline-trained × 0.85
- [ ] **Autoscaler live**: Often missing **drain handshake** — verify zero in-flight drops during forced role flip
- [ ] **Cache-aware router wins**: Often missing **cold vs warm regime split** — verify both regimes reported separately, not blended
- [ ] **LMCache integrated**: Often missing **eviction policy measurement** — verify eviction rate under workload's specific prefix distribution
- [ ] **Metrics reconciled**: Often missing **per-second reconciliation, not per-run** — verify ±2% on every 30s window, not just run-end
- [ ] **P:D ratio tested**: Often missing **prompt-burst-then-decode-shift** — verify autoscaler handles single phase transition within 120s w/o drops
- [ ] **PID gains "tuned"**: Often missing **anti-windup** — verify integral term doesn't grow past actuator max

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| NIXL silent garbage | LOW | Backend → UCX; sentinel-check fails; abort + redeploy UCX |
| CVE-2025-25183 collision | LOW | Upgrade vLLM ≥0.7.2; flush all prefix caches |
| EAGLE-3 NaN on disagg | MEDIUM | Disable spec decode (loss 2× ITL); investigate draft/target KV layout |
| Autoscaler thrashing | LOW | Force-dwell next flip; if recurring, raise min-dwell to 240s; review PID gains |
| LMCache evicts hot prefix | MEDIUM | Pre-warm out of band; switch to per-class budget; re-bench |
| KV transfer timeout/stall | LOW | Fail req, return 503, router retries on diff worker |
| Drain protocol loses in-flight | HIGH (SLO breach) | Resume static roles; investigate KVConnector.shutdown blocking; custom drain protocol; bench → spec rerun |
| Metrics drift >2% | LOW | Identify diverging segment (parse/queue/engine); fix reconciliation script |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| P1 (NIXL silent corruption) | Phase 1 (TOPO) | TOPO-05 health check includes sentinel-token validity test, not just `kv_transfer_complete_count` increment |
| P2 (CVE-2025-25183) | Phase 1 (vLLM version pin) + Phase 3 (RTR) | vLLM ≥0.7.2 in `make provision`; salt per pool in `core.routing` |
| P3 (spec × disagg KV) | Phase 6 (SPEC) | SPEC-04 documented; SPEC-05 acceptance measured per workload vs colocated baseline |
| P4 (chunked vs disagg crossover) | Phase 8 (BENCH) | BENCH-03 goodput curves include both topologies on same axes; explicit crossover region |
| P5 (drain protocol) | Phase 7 (AUTO) | AUTO-05 verified in failure drill: zero in-flight drops during forced role flip |
| P6 (autoscaler thrashing) | Phase 7 (AUTO) | AUTO-04 verifier includes thrash check: 30-min trace, flip-count-per-minute <0.5 |
| P7 (cold-cache false confidence) | Phase 4 (Router Verification) | RTR-04 measurement split into cold (first 50 reqs) and warm; both reported |
| P8 (prefix-index blowup) | Phase 3 (RTR) | `prefix_index_size_bytes` capped; 24h soak test Phase 8 |
| P9 (LMCache eviction) | Phase 5 (KV) | KV-03 documents chosen policy per workload; KV-04 break-even curve includes eviction-rate overlay |
| P10 (metric drift) | Phase 2 (LOAD) | LOAD-06 reconciliation script runs per-30s window, not per-run |
| P11 (spec acceptance collapse) | Phase 6 (SPEC) | SPEC-02 ITL curve; SPEC-03 auto-disable + alarm if disabled >30% time |
| P12 (pathological mix) | Phase 8 (Failure drills) | BENCH-06 includes RAG-burst-over-chat as Drill 3 |

---

## Phase-Specific Warnings

| Phase | Likely Pitfall | Mitigation |
|-------|---------------|------------|
| Phase 1 (TOPO) | NIXL silent corruption (P1); vLLM <0.7.2 (P2) | Sentinel validity test; version pin in `make provision` |
| Phase 2 (LOAD) | Metric reconciliation drift (P10) | Per-30s reconciliation, not per-run |
| Phase 3 (RTR) | Prefix-index blowup (P8); cache-aware cold false confidence (P7) | Hard cap on index size; dual-regime reporting |
| Phase 4 (Router verify) | Cold vs warm regime confusion (P7) | Split measurement; cold phase reported separately |
| Phase 5 (KV) | LMCache eviction policy mismatch (P9) | Pick per workload; prewarm; per-class budget |
| Phase 6 (SPEC) | EAGLE × disagg KV (P3); acceptance collapse (P11) | Acceptance vs colocated baseline; ITL-vs-batch curve; auto-disable |
| Phase 7 (AUTO) | Drain protocol bugs (P5); thrashing (P6) | Explicit drain coord; rate-limited actuator; AUTO-04 verifier |
| Phase 8 (BENCH) | Pathological mix (P12); crossover not measured (P4) | Failure drills; both topologies same axes |

---

## Sources

### HIGH confidence (first-party issues / official docs)
- [CVE-2025-25183 — Miggo](https://www.miggo.io/vulnerability-database/cve/CVE-2025-25183)
- [vLLM #27055 — NIXL LIBFABRIC garbage](https://github.com/vllm-project/vllm/issues/27055)
- [vLLM #24885 — graceful shutdown RFC](https://github.com/vllm-project/vllm/issues/24885)
- [LMCache #649 — eviction remote server (P0)](https://github.com/LMCache/LMCache/issues/649)
- [LMCache architecture docs](https://docs.lmcache.ai/developer_guide/architecture.html)
- [Mooncake paper (FAST 2025)](https://www.usenix.org/system/files/fast25-qin.pdf)
- [vLLM NixlConnector docs](https://docs.vllm.ai/en/stable/features/nixl_connector_usage/)
- [TensorRT-LLM spec decode docs](https://nvidia.github.io/TensorRT-LLM/1.2.0rc6/features/speculative-decoding.html)

### MEDIUM confidence (cross-validated)
- [Splitwise (ICML 2024)](https://arxiv.org/abs/2311.18677)
- [DistServe (OSDI 2024)](https://arxiv.org/abs/2401.09670)
- [Sarathi-Serve (EuroSys 2025)](https://arxiv.org/abs/2403.02310)
- [KEDA + HPA stabilization](https://oneuptime.com/blog/post/2026-02-09-hpa-stabilization-window-prevent-thrashing/view)
- [Nearform K8s autoscaling](https://nearform.com/digital-community/the-hidden-complexities-of-kubernetes-autoscaling-beyond-the-basics/)
- [Why RDMA KV Cache Transfer Broke in K8s](https://medium.com/@owumifestus/why-rdma-kv-cache-transfer-broke-in-kubernetes-dd31fd66fe9a)
- [sglang #19796 — Eagle V2 NaN](https://github.com/sgl-project/sglang/issues/19796)
- [Hydragen](https://arxiv.org/html/2402.05099v2)
- [NVIDIA KV cache compression research](https://research.nvidia.com/labs/eai/blogs/kv-cache-compression-and-its-infra-problems/)

### LOW confidence (single source, validate)
- LMCache stampede behavior — only attested in [Momento blog](https://www.gomomento.com/blog/reduce-ttft-by-50-with-lmcache-momento-accelerator/)
- PID gain tuning for GPU inference — no canonical reference

### Background papers
- TetriInfer (2024), Infinite-LLM (2024), SpotServe (2023), Flying Serving (Feb 2026), BARS (Microsoft/PKU)

---

## Confidence Assessment

| Area | Confidence | Reason |
|------|------------|--------|
| NIXL/UCX/LIBFABRIC (P1) | HIGH | Direct GH bug reproduction |
| CVE-2025-25183 (P2) | HIGH | CVE w/ affected versions + patches |
| Spec × disagg (P3) | HIGH | Multiple papers + sglang reproducer + TRT-LLM reference |
| Chunked vs disagg crossover (P4) | HIGH | Four canonical papers agree direction; magnitudes vary |
| Drain protocol (P5) | MEDIUM-HIGH | RFC specified, actual KVConnector.shutdown() not audited |
| Autoscaler thrashing (P6) | HIGH | Control theory + K8s HPA/KEDA precedent |
| Cold-cache false confidence (P7) | MEDIUM | Logical consequence; magnitude varies |
| Prefix-index blowup (P8) | MEDIUM | Common pattern; workload-dependent |
| LMCache eviction (P9) | MEDIUM | Issue #649 exists but body empty; default LRU confirmed |
| Metric reconciliation (P10) | HIGH | Industry-wide; BARS paper quantifies |
| Spec acceptance collapse (P11) | HIGH | Direct EAGLE behavior + empirics |
| Pathological mix (P12) | MEDIUM | Emergent LMCache + workload interaction |

---

## Gaps to Address Later

1. Exact LMCache eviction semantics under agentic workload — empirical Phase 5
2. PID gain tuning methodology — first-principles Phase 7
3. Per-30s reconciliation methodology — Phase 2
4. EAGLE-3 vs disagg pairwise replication — Phase 6 200-line reproducer
5. Pure cold-cache TTFT on cache-aware router — Phase 4

---
*Pitfalls research for GoodputLab — P/D Disaggregated LLM Serving. Source review performed vs vLLM v0.11+ docs (NIXLConnector), LMCache current docs + issue tracker (2026-07-08), EAGLE-3 refs, Mooncake FAST 2025, CVEs through 2025-25183.*