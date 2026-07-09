# Phase 2 Addendum — P10 Loadgen-vs-vLLM Metrics Reconciliation

**Status:** design sketch (per `suggestions/feedback.md` priority #4)
**Owner:** Phase 2 Load + Metrics planner must convert this sketch into a PLAN.
**Refs:** `.planning/research/PITFALLS.md` P10 (per-run averages hide
compensating drift)

---

## Why Per-Run Averages Lie

A natural-but-wrong baseline: aggregate every request, take means, claim
numbers. This silently hides scenarios where TTFT is +50% in the first half
of the bench and -50% in the second half — the average reads zero drift
while the SLO was violated half the time. The right unit is the **30-second
window**, reconciling loadgen-observed per-request latency against
vLLM-reported histogram counters over the same window.

## Window Definition

- **Window length:** 30 s wall clock (sliding, advancing every 30 s).
- **Per-window aggregation:**
  - `loadgen_ttft_p50`, `loadgen_ttft_p95` (per-window, computed from
    `request_finished_at - request_received_at`).
  - `vllm_ttft_p50`, `vllm_ttft_p95` (per-window, taken from
    `vllm:time_to_first_token_seconds_bucket`).
  - `loadgen_itl_p50`, `loadgen_itl_p95` (inter-token latency from
    streaming responses).
  - `vllm_itl_p50`, `vllm_itl_p95` (from `vllm:inter_token_latency_seconds`).
- **Reconciliation stat:** Wasserstein-1 distance per window per metric;
  window passes if Wasserstein-1 ≤ 0.02 × loadgen metric median (i.e.
  deviation ≤ 2%).

## Prometheus Scrape Schema (Phase 2 ships)

```prom
# Reconciliation observability (per 30s window)
goodputlab_recon_wasserstein_ttft{service="vllm-colocated"}     <float>
goodputlab_recon_wasserstein_itl{service="vllm-colocated"}      <float>
goodputlab_recon_window_pass{service="vllm-colocated"}          0|1
goodputlab_recon_gap_seconds{service="vllm-colocated"}          <int>
```

A scrape is "missing" if the loadgen observed a request in the window but
vLLM produced no histogram sample (likely client-side clock skew). The
`goodputlab_recon_gap_seconds` counter holds total seconds spent in such
gaps. CI gate: `goodputlab_recon_window_pass == 0` triggers a Phase 2 test
failure.

## Workload Trace Sketches

Phase 2 ships three trace templates under `loadgen/traces/`:

| Trace | Profile | Length / Cadence | Notes |
|---|---|---|---|
| `chat.jsonl` | Multi-turn chat | 512 tokens / 4 turns | Steady arrival at 16 RPS |
| `rag.jsonl` | RAG with long context | 8K–32K tokens / 1 turn | Burst 32 → 0 RPS over 60 s |
| `agentic.jsonl` | Tool-calling loop | 1–4K tokens / variable turns | Geometric inter-request gap |

Each trace declares its target SLO class (`interactive | batch`) so the
router can exercise admission control later (Phase 3). All traces are
reproducible with a `--seed` flag.

## Reconciliation Test Plan

Static:
- `tests/test_reconcile.py::test_wasserstein_under_2pct` constructs two
  matched-sample sets (loadgen vs vLLM) with 1.5% injected Wasserstein
  drift and asserts the stat returns within 2% on synthetic data.
- `tests/test_reconcile.py::test_window_gap_counter` asserts gap counter
  increments when a deliberate "missing" sample is injected.

Live (gated by `GOODPUTLAB_RUN_LIVE=1`):
- Spin colocated, run a 60-s trace, assert every 30-s window passes.

## Phase Plan Required Sections (Phase 2)

1. Implement `loadgen/` runner emitting JSONL of `(request_id,
   received_at, ttft, itl_p50)`.
2. Implement the 30-s reconciliation harness at
   `control/reconcile.py` (Wasserstein-1, gap detection).
3. Export Prometheus metrics from the harness.
4. Bake synthetic-data unit tests for the recon math.

## Sample Test Sketch (early)

```python
import numpy as np
from scipy.stats import wasserstein_distance

def _tw(samples_a, samples_b):
    return wasserstein_distance(samples_a, samples_b) / np.median(samples_a)

def test_wasserstein_under_2pct():
    base = np.random.gamma(0.08, scale=0.005, size=500)
    drift = base + np.random.normal(0, 0.001, size=500)
    assert _tw(base, drift) < 0.02
```

`scipy` is acceptable here (small extra dependency; can downgrade to a
plain-histogram implementation if scipy is too heavy).

---

*Drafted 2026-07-09. Phase 2 planner to incorporate.*
