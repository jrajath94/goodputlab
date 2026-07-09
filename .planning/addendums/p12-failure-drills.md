# Phase 8 Addendum — Failure Drills (Postmortem Templates)

**Status:** design sketch (per `suggestions/feedback.md` priority #5)
**Owner:** Phase 8 BENCH planner must run all three drills and emit
postmortem-style appendix entries.
**Refs:**
- `.planning/research/PITFALLS.md` P5 (drain in-flight loss), P1 (NIXL
  silent garbage), P12 (pathological mix)

---

## Why Failure Drills Are Hiring Signal #4

Bench numbers describe the **happy path**. Hiring committees at frontier
labs care whether you can debug the unhappy path. The 3K-word report caps
with three postmortem-style drill writeups — each follows the same
template: **Detection signal**, **Recovery steps**, **Instrumentation gap**.

---

## Drill 1: Decode Node Dies Mid-Stream (P5)

### Setup

- Cluster: 2 prefill pools + 2 decode pools, control plane live.
- Workload: 32 interactive chat requests at 8 RPS.
- Mid-run (5 s in): SIGKILL one decode pool. No graceful shutdown.

### Detection Signal

- `goodputlab:decode_pool_heartbeat` counter for the killed pool stops
  incrementing within 5 s of the kill.
- `vllm:e2e_request_latency_seconds_count` delta for the killed pool falls
  to zero while the surviving pool's count continues.
- The router sees the pool health go `UNHEALTHY` after a 3-retry
  handshake policy.

### Recovery Steps

1. Router removes pool from the eligible set (live, no config reload).
2. In-flight requests on the dead pool surface `503` to clients — clients
   must retry. The router's `role_flip_inflight_dropped` counter MUST read
   **zero** because the pool flip should not cancel streams in the
   surviving pool.
3. Pool supervisor restarts the killed vLLM; readiness probe waits for
   `/v1/models` to return `goodputlab-model` and for
   `vllm:model_loaded` to flip `1`.
4. Pool rejoins eligible set after `decode_pool_join_dwell_seconds=60`
   (prevents thrash).

### Instrumentation Required

- `goodputlab:decode_pool_heartbeat{role="decode",pool_id="..."}` counter,
  incrementing every 1 s while alive.
- `goodputlab:role_flip_inflight_dropped_total{from_pool=...,to_pool=...}`
  counter — **must remain zero during the drill**.
- `goodputlab:pool_rejoin_dwell_seconds` gauge — confirms 60-s dwell.
- vLLM `/metrics` for the killed pool: `vllm:num_requests_swapped`
  should spike during the disconnect.

---

## Drill 2: KV-Transfer Stall (P1)

### Setup

- Disagg topology only. NIXL backend = UCX.
- Workload: 64 RAG requests at 16 RPS, 8K prompt tokens each (heavy
  prefill, the failure-prone regime).
- Mid-run (10 s in): drop UDP traffic to the NIXL side-channel port
  (e.g. `iptables -A OUTPUT -p udp --dport 8210 -j DROP`). All NIXL
  transfers stall but the underlying vLLM engine is still "alive".

### Detection Signal

- **Primary:** sentinel probe (the 60-s daemon) reports
  `sentinel_drift=1` — the decode path produces a token mismatch against
  the recorded fixture.
- **Secondary (real NIXL metrics, per D-03):**
  - `vllm:nixl_bytes_transferred_sum` ceases to increment
  - `vllm:nixl_num_failed_transfers_total` rises (the connection drops are
    recorded as failures, not stalls)
- Tertiary: prefill-p99 TTFT climbs above 3× the rolling baseline.

### Recovery Steps

1. Sentinel probe pages via `goodputlab:sentinel_drift` gauge hitting 1.
2. Control plane routes subsequent requests to the colocated fallback
   pool; KV-dependent requests get a `503 Retry-After` header.
3. Operator restarts the NIXL side channel or reroutes UCX to a different
   fabric interface.
4. Sentinel probe clears once the new fixture matches decode output
   (re-record on first successful post-recovery serving).

### Instrumentation Required

- `goodputlab:sentinel_drift{topology="disagg"}` gauge, 0/1.
- `vllm:nixl_xfer_time_seconds_count` per pool — must increase during
  healthy operation.
- `vllm:nixl_num_failed_transfers_total` counter — baseline expectation is
  **zero**; nonzero indicates corruption.
- Histogram: `vllm:nixl_xfer_time_seconds_bucket` per pool, per transfer
  size bucket.

### What This Drill Does NOT Use

`vllm:kv_transfer_complete_count`. Per D-03, that counter increments on
the metadata-layer completion event and **cannot detect corrupt bytes**.
The probe-based sentinel check + the NIXL metrics are the canonical signal.

---

## Drill 3: Pathological Mix (P12)

### Setup

- Mixed workload: 4 RAG-burst-of-32K requests landing in the same
  200-ms window, on top of a steady 8-RPS interactive chat background.
- Topology under test: all four topologies run independently; report
  includes the worst cells per topology.

### Detection Signal

- Pre-fill queue depth (`vllm:num_prefill_slots`) saturates within the
  first 200 ms.
- Interactive TTFT p95 climbs from baseline 200 ms to >1500 ms.
- Admission control sheds BATCH traffic as designed; INTERACTIVE traffic
  remains within SLO.

### Recovery Steps

1. Router's SLO classifier routes incoming RAG bursts to the chunked
   topology OR the disagg prefill pool (not the colocated-decode pool).
2. Batch-tier traffic queuing delay grows but does not starve interactive
   traffic — the admission-control logic gates on
   `interactive_queue_depth / batch_queue_depth` ratio, not absolute
   depth.
3. After the 32 RAG requests drain, TTFT p95 returns to baseline within
   5 s (recovery SLO).

### Instrumentation Required

- `goodputlab:slo_attainment{class="interactive",window="30s"}` histogram
  — the gold metric of this drill.
- `goodputlab:routed_to{topology="...",class="..."}` counter.
- `vllm:num_prefill_slots{role="prefill"}` gauge for each prefill pool.

---

## Cross-Drill Findings to Catalog

Phase 8 plan should capture, per drill:
1. Detection latency wall-clock seconds.
2. Recovery wall-clock seconds.
3. Any metric not exposed at the time that we wished we had.
4. Any "good behavior" the system exhibited that we did not anticipate.

---

*Drafted 2026-07-09. Phase 8 planner to incorporate.*
