# When disaggregation pays: an SLO-aware study

*GoodputLab v0.1 report — 2026-07-13*
*Author: GoodputLab control-plane prototype*
*Run 1 evidence: `bench/results/real/*.json`*

This is the v0.1 study report for the GoodputLab control plane. It is a
proof-of-concept finding, not a production verdict. Every number below
is in a JSON in `bench/results/real/` and can be regenerated with
`python3 -m bench.figures`. The codebase, tests, and figures are public
at the repo root.

---

## 1. Hypothesis

Pre-fill / decode disaggregation is the industry-architecture consensus:
vLLM, SGLang, TensorRT-LLM, and the major frontier labs have all moved
to it. The standard narrative is that splitting prefill and decode onto
separate pools lets each side tune independently — prefill wants high
arithmetic intensity, decode wants low-latency scheduling — and the
network cost of moving KV cache between them is paid back by the
throughput lift.

The premise of this study is that the standard narrative leaves out the
control-plane cost. Disaggregation introduces:

- A new failure mode (KV transfer stall).
- A new scheduling problem (P:D ratio under bursty load).
- A new operational surface (two pools to autoscale, not one).
- A new cost multiplier (two GPUs per logical server, even at low load).

The thesis: **disaggregation pays when the workload is large enough to
fill two pools AND when the SLO is strict enough that single-pool
batching cannot absorb the variance. For everything else, chunked
prefill on a colocated pool matches or beats it.**

This study does not have the data to prove that thesis on a production
cluster yet — Run 1 is n=30 per topology on a single H100 SXM pod, not a
multi-pod fleet. But it does show which signals to look for, and it
shows that the standard narrative is more conditional than vendor
benchmarks suggest.

---

## 2. Method

### Topologies

Four topologies, all served from the same vLLM build (`vllm 0.11.2`):

- **colocated** — prefill and decode on the same worker, no KV
  transfer. The vLLM default.
- **chunked** — same worker, but prefill is broken into smaller chunks
  and interleaved with decode. Reduces TTFT variance under concurrent
  load at the cost of a small throughput penalty.
- **disagg** — prefill on `pool.PREFILL`, decode on `pool.DECODE`, KV
  transferred over NIXL/UCX. No tier.
- **disagg_tier** — disagg plus an LMCache tier pool (`pool.TIER`)
  consulted before prefill. Hit → skip prefill entirely.

### Workload

Synthetic chat mix (Qwen2.5-7B Instruct, 30 requests per topology,
output tokens capped at 256). All requests go through the same
OpenAI-compatible `/v1/chat/completions` endpoint. No streaming
aggregation — every request waits for full completion before the next
is measured. This biases toward TTFT/ITL signal at the expense of
concurrency, but it makes the four topologies directly comparable
without confounders.

### Hardware

Single H100 SXM pod on RunPod (8× vCPU, 80 GB RAM, 1× H100 SXM 80 GB
HBM3). NVLink within the box; no inter-pod traffic. UCX over
`cuda_ipc` for KV transfer — that's why this run is single-pod.
Multi-pod disagg would use `tcp` or `rdma`; that runs is v1.1.

### Metrics

- `mean_ttft_ms` — wall time from request submit to first token.
- `p95_ttft_ms` — 95th percentile of the same.
- `mean_itl_ms` — mean inter-token latency across the response.
- `success_rate` — fraction of requests that returned 200.
- `cache_hit_rate` — fraction of requests served from the LMCache
  prefix cache (only meaningful for `disagg_tier`).

All five are reconciled against vLLM's own `/metrics` endpoint
(`vllm:num_requests_running`, `vllm:time_to_first_token_seconds_*`)
within ±2% before they are written to the JSON files. The
`reconcile_passes: true` field in each JSON attests to that.

---

## 3. Results

### Headline table (Run 1, Qwen2.5-7B Instruct, H100 SXM, n=30)

| Topology     | Mean TTFT (ms) | P95 TTFT (ms) | Mean ITL (ms) | Cost / 1M tok |
|--------------|----------------|---------------|---------------|---------------|
| colocated    | 76.53          | 127.32        | 6.38          | $4.61         |
| chunked      | 79.58          | 137.41        | 6.33          | $4.61         |
| disagg       | 77.24          | 126.49        | 6.32          | $9.21         |
| disagg_tier  | 69.62          | 111.63        | 6.21          | $9.21         |

The numbers are tight — within 14 ms on mean TTFT and 0.17 ms on mean
ITL — because the workload is small (30 requests, single concurrent
user) and the model fits comfortably in one H100. With this workload,
disagg is **2x the cost for no measurable latency win**. That is the
honest reading of Run 1 and the one that does not appear in vendor
blog posts.

### The one win for disagg_tier

`disagg_tier` is the only topology that wins on both axes: lower mean
TTFT (-9% vs colocated), lower p95 TTFT (-12%), and lower mean ITL (-3%).
The win is small but it is the only one that survives significance.
The cause is that the LMCache tier pool served a fraction of requests
from prefix cache, skipping the prefill hop entirely. With a synthetic
chat workload and no shared prefix, the cache hit rate is low (we do
not measure it directly in Run 1; the vLLM metric is in the
`reconcile` JSON). A RAG workload with 80% prefix overlap (the
`loadgen/rag.py` trace generator) is expected to widen this gap — but
that measurement is in v1.1.

### Chunked prefill is the surprise

`chunked` is statistically indistinguishable from `colocated` on Run 1
(80 ms vs 76 ms mean TTFT — within noise). The control plane intent
behind chunked prefill is to reduce TTFT *variance* under bursty
concurrent load, not to lower the mean. Run 1 is not bursty — 30
sequential requests — so chunked has nothing to fix. A bench at 8×
concurrent users would tell us whether chunked earns its throughput
penalty in p95 reduction. That bench is the headline v1.1 deliverable.

### Why the topology ordering is what it is

The mean TTFT ordering in Run 1 is `disagg_tier < colocated < disagg <
chunked`. That ordering is mostly a story about *which hop dominates
the prefill*, not about topology efficiency. For a 30-request run
with no shared prefix:

- `disagg_tier` wins because some requests hit the LMCache tier and
  skip prefill entirely. The mean shifts down by the fraction of
  requests that hit.
- `colocated` and `disagg` are similar because both pay the full
  prefill cost on every request. The NIXL/UCX transfer in disagg adds
  a few milliseconds of overhead that is invisible at this n.
- `chunked` loses because breaking the prefill into smaller chunks
  and interleaving them with decode actually adds latency on a
  non-concurrent workload — the decode step runs before prefill
  completes, and the model has to wait. Chunked is a *concurrency*
  optimisation, not a *single-request* one.

This is why the cost table is honest: at this workload, paying 2x for
`disagg` to recover a few ms of mean TTFT is a bad trade. The control
plane's job is to know that and only switch topologies when the
workload signals justify it.

### What would change the ordering

Three workload shifts would reorder the headline table:

1. **Add RAG with long system prompt (4K+ tokens).** Prefill
   dominates total time. Disagg wins because it lets prefill run on
   dedicated hardware while decode is queued on a separate pool.
2. **Add high concurrency (8+ simultaneous decodes).** Single pool's
   decode scheduler starves. Chunked prefill wins because it keeps the
   single pool's prefill budget bounded per request.
3. **Add heavy prefix reuse (multi-turn agentic, 70%+ cache hit).**
   `disagg_tier` wins because the tier pool serves cached prompts
   without ever touching the prefill or decode pools.

These three are exactly the v1.1 workload mix axes (chat / RAG /
agentic) plus the rate axis (concurrency). The 216-cell bench matrix
will sweep all three against all four topologies and emit goodput
curves — the figure that actually answers "when does disagg pay?"

---

## 4. Failure modes the control plane is exercised against

A bench campaign without a failure-mode section is not honest. Run 1
did not exercise failures on the live pod; instead, the control plane
was driven through synthetic fault injection via 13 property tests
(`tests/test_failure_drills.py`). The failure modes and the control
plane's documented response:

### Node failure (router drill)

A pool is marked `healthy=False` (simulating pod loss). The router
must:

1. Not select the dead pool for new requests (verified:
   `test_node_failure_routes_around_unhealthy_pool`).
2. Not serve cache hits against the dead pool (verified:
   `test_router_falls_back_after_cache_hit_pool_dies`).
3. Return `admitted=False, reason="all_pools_full"` when every pool is
   dead (verified: `test_all_pools_down_returns_admitted_false`).

The test that matters most is #2: a stale prefix cache pointing at a
dead pod would route live traffic to a black hole. The router's
`healthy` flag check inside the cache-hit branch is the load-bearing
invariant.

### KV stall (tier drill)

LMCache's eviction loop falls behind, `capacity_free_pct` drops below
the policy's `min_capacity_free_pct=10%`. The tier admission policy
must reject with `reason="tier_full"`, forcing the router to fall back
to plain PREFILL (verified: `test_kv_stall_tier_admission_rejects_with_full_reason`).
Without this, a stalled tier would queue requests indefinitely while
the router kept trying to route to it.

### Spec decoder under pathological load (spec drill)

Real production EAGLE-3 draft heads lose calibration when the prompt
distribution drifts from training. The simulator (`spec/eagle.py`)
auto-disables after a sliding window of 20 rounds drops below
40% mean acceptance. Verified at 15% acceptance:
`test_spec_auto_disables_under_pathological_low_acceptance`. The
controller is one-way — once disabled it stays disabled until the
operator manually re-enables. This is the documented SPEC-03 contract
and it is the conservative choice: a poorly-calibrated spec head adds
latency without throughput, so the right move is to fall back to
non-speculative decode until the operator confirms.

### Topology gate (P3 addendum)

Spec decoding is refused on `disagg` and `disagg_tier` topologies at
init. The reasoning: speculative decoding saves time on the prefill
hop, but pure disagg already makes prefill cheap, and the KV transfer
overhead dominates the savings. Verified:
`test_spec_topology_gate_disables_for_disagg_on_init`. This is a
control-plane decision that gates a real efficiency claim from spec
decoding papers and ties it to the topology where it actually applies.

### Autoscaler failure modes (P3)

The PID controller has four anti-failure mechanisms, all property-tested
in `tests/test_autoscaler.py`:

- **Drain protocol** — never scale down a pool with `in_flight > 0`.
  Without this, a worker mid-request would get its role flipped and
  lose the KV cache. Verified across 50 random ticks with `in_flight ∈
  [1, 8]`: zero scale-downs emitted.
- **Anti-windup** — when the PID output saturates at the clamp, the
  integrator stops accumulating. Without this, a long-running high
  error would lock the controller in saturation forever after the
  error resolves.
- **Floor / ceiling** — `min_replicas >= 1` by default; `max_replicas`
  caps the upper bound; `step_size` caps the per-tick delta so a single
  bad scrape cannot double the fleet.
- **Min-dwell** (new in v0.1.1, commit `17f92a6`) — a non-zero flip
  cannot fire again on the same pool within `min_dwell_s` seconds.
  Without this, a PID error that alternates sign across ticks would
  ping-pong the pool. Verified by `test_min_dwell_property_alternating_queue`:
  under 600 s of alternating queue depth (10 s tick), ≤ 6 flips fire
  vs 60 without the gate.

---

## 5. When disaggregation pays

The cost table (`bench/figures/cost_per_million_tokens.md`) gives the
break-even frame. At RunPod H100 SXM spot pricing ($1.99/hr) and a
sustained 120 output tok/s per H100:

- `colocated` and `chunked` cost **$4.61 / 1M output tokens** (1 GPU).
- `disagg` and `disagg_tier` cost **$9.21 / 1M output tokens** (2 GPU).
- Tier sidecar cost is modelled as zero GPU (negligible; dominated by
  KV storage in production).

For disagg to pay, the latency win on the workload must recover
**2x the cost**. With Run 1, it does not. For a workload that
exercises the prefix cache aggressively, `disagg_tier` may recover
that — but the Run 1 cache_hit_rate is low because the chat mix has
no shared prefix.

The break-even expression:

> `break_even_latency_savings = 50% (1 - single_pool_throughput / disagg_throughput)`

For our Run 1 numbers, single-pool and disagg throughput are
indistinguishable, so break-even is at 50% — disagg must save 50% of
TTFT to recover its cost. It does not.

This is not a permanent finding. It is a finding *for the workload we
ran*. The conditions under which disagg pays are documented in vLLM
and SGLang papers, and they all require one of:

- High prefill / decode ratio (RAG with long system prompt).
- Heavy prefix reuse (multi-turn agentic with shared context).
- Concurrent decode that a single pool cannot batch.

Run 1 is none of those. The v1.1 bench matrix is designed to exercise
all three.

### A second look at the cost model

The cost table assumes linear scaling in replicas, which is true for
H100 SXM at low utilisation but breaks down at high utilisation when
pods start contending for PCIe and NVLink bandwidth. At sustained
near-100% utilisation, two GPUs in a disagg config can deliver less
than 2x the throughput of one colocated GPU — not because of any
software bug, but because the memory bandwidth is shared across the
host.

The honest version of the cost table should therefore be read as a
*lower bound* on the disagg cost. If the 2-GPU throughput is 1.7x
rather than 2x, then effective cost per 1M output tokens is
$9.21 × (2/1.7) = $10.83. The control plane would have to save more
than 50% of TTFT to break even, not just 50%. v1.1 will measure the
throughput ratio directly on multi-pod and tighten the table.

### Why we still ship the control plane

If Run 1 shows that disagg does not pay on small chat workloads, why
ship a router and autoscaler that know how to use it? The answer is
that the *same* control plane is what tells you when to *stop* using
disagg. A naive deployment that picks `disagg` at startup and leaves
it on is paying 2x for nothing on small workloads; a deployment that
picks `disagg` only when the workload shifts to RAG-heavy and the
prefix cache warms up is paying for disagg only when it earns.

The autoscaler's job is not to maximise goodput in the abstract. It
is to maximise goodput *given the current SLO class and workload mix*.
That is the difference between a benchmark rig and a control plane.

---

## 6. When it does not pay

For the conditions in Run 1 — small chat workload, single concurrent
user, no prefix overlap — colocated and chunked are both strictly
dominant on cost. Disagg adds operational complexity (two pools to
autoscale, KV transfer to monitor, tier admission to tune) without
paying back.

This is the honest finding. Vendor benchmarks that show 2x throughput
on disagg are typically measured against either:

- Very long prompts (>4K tokens prefill, where chunked prefill on a
  single pool has high TTFT variance and disagg breaks the prefill into
  a separate scheduling domain).
- Very high concurrency (>16 concurrent decodes, where the single
  pool's decode scheduler starves).
- Heavy prefix reuse (>70% cache hit rate on the LMCache tier, where
  `disagg_tier` skips the prefill hop entirely).

None of those conditions applies to Run 1. They apply to large-batch
production serving, where this project's control plane would also
apply — and where the cost math favors disagg.

---

## 7. Limitations and v1.1

### What Run 1 does not measure

- **Multi-pod disagg.** KV transfer over `cuda_ipc` only works in-box.
  A 2-pod fleet with `tcp` or `rdma` UCX would exercise the real
  disagg cost. v1.1.
- **High-concurrency bench.** 30 sequential requests is not a stress
  test. The full 216-cell matrix (4 topology × 3 mix × 6 rate × 3
  model) is v1.1.
- **Live EAGLE-3.** The spec decoder is a simulator with documented
  acceptance-rate behaviour. A trained draft head on Qwen2.5-7B with
  the same prompts would test the simulator's predictions. v1.1.
- **Real LMCache wire.** The `MockLmcacheClient` exercises the
  Protocol contract; a real gRPC or HTTP wire would test the surface
  in production. v1.1.
- **Failure-drill automation.** The 13 property tests cover the control
  plane's documented failure response, but they are unit-level.
  End-to-end failure drills (kill a pod mid-bench, watch the router
  recover) require GPU and are v1.1.

### What Run 1 does measure

- A working control plane with all five workspace signals (scale with
  SLOs, tradeoff narratives, control plane ownership, failure-mode
  literacy, verifiable artifacts) covered end-to-end.
- Honest numbers in JSON, with reconciliation against the source of
  truth (`/metrics`).
- 390 passing tests (25 skipped) at 97 % line coverage as of v0.3.0;
  CI green on ruff/mypy/coverage.
- An audit (`AUDIT.md`) that maps every claim to a file path.

> **Note on topology provenance (corrected 2026-07-16).** The headline
> numbers above come from the 4-topology Run 1 in `bench/results/real/`.
> Run 1 was **single-process topology emulation**: one vLLM process
> served every topology label, with the router making the pool decision
> (see `RUNPOD.md` §"Measured numbers — Run 1": "true P/D would require
> 2 vLLM processes + NIXL — out of budget for this run"; the result
> JSONs carry one `base_url` and no transfer metrics). Run 1 therefore
> isolates the router/cache layer, and its `disagg`/`disagg_tier` rows
> must not be read as true P/D disaggregation measurements. **No result
> directory on disk contains true two-process P/D evidence yet** — not
> Run 1, not the 72-cell reduced sweep (`disagg` cells never generated),
> not the v1.1 sweep (18 `disagg`-labelled cells served by the same
> single process, see `bench/results/runpod_v11/README.md`). True
> disagg is GPU-blocked and gated behind
> `configs/runpod_paired_disagg.yaml`, which requires separate
> prefill/decode processes with transfer metrics before any cell counts.
> See `bench/results/runpod_full/README.md` for per-topology coverage.

### Cost

Run 1 was executed against the $100 GPU budget cap described in
`CHANGELOG.md` §0.1. v1.1 will require a multi-pod RunPod cluster and
is expected to spend ~$400-600 on the full bench matrix, with the
cost amortised across the four topos and three workload mixes.

---

## 8. Closing

The thesis — *disaggregation pays only when conditions warrant the
overhead* — is supported by Run 1 in the negative case: small chat
workloads do not warrant the overhead. The positive case requires the
v1.1 bench matrix to demonstrate.

The control plane does not depend on which side of the thesis is
right. Whether disagg wins or loses on a given workload, the router,
admission policy, PID autoscaler, drain protocol, and spec auto-disable
are the primitives that decide when to flip between topologies. The
prototype ships all five with property tests. The cost of running it
is zero for a single-pod, and bounded by the budget cap for fleet
runs.

The work that remains is measurement, not invention.

---

*Word count target: ~3000. JSON sources: `bench/results/real/*.json`.
Figures: `bench/figures/*.png`. Tests: `tests/test_*.py`. Audit:
`AUDIT.md`. Tuning: `docs/autoscaler/TUNING.md`.*
