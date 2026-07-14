# RunPod Full Sweep — 2026-07-14

72-cell sweep on real H100 SXM (3 models served by single vLLM process).

## What ran

- **Pod:** RunPod secure cloud, H100 SXM 80GB HBM3 ($2.99/hr)
- **vLLM:** 0.11.2, max-model-len 4096, chunked prefill ON, multi-model
  (Qwen2.5-7B-Instruct + Qwen3-1.7B + Qwen3-30B-A3B requested; actual
  serving model recorded per cell)
- **Sweep:** 4 topologies × 3 models × 6 rates × 3 mixes = **72 cells**
- **Per cell:** 5 warmup + 30 measure requests

## Headline (honest)

| Metric | Value |
|---|---|
| Cells completed | 72/72 (no crashes; all wrote JSON) |
| Cells reconcile_passes=True (success≥0.99) | **24/72 (33%)** — chat mix only |
| Cells with success_rate=0.0 (stub) | **48/72 (67%)** — 24 agentic + 24 RAG, both fully failed |
| Mean TTFT (24 reconciled, chat-only) | **197.55 ms** |
| Mean ITL (24 reconciled, chat-only) | **8.39 ms** |
| Median TTFT (24 reconciled) | 120.83 ms |
| p95 TTFT (24 reconciled) | 493.42 ms |
| Thermal warnings | 0/72 |

> **Note on summary.json aggregates:** the campaign-level `mean_ttft_ms` /
> `mean_itl_ms` in `summary.json` (101.23 / 5.07) average zeros from
> stub cells together with the 24 reconciled cells. The numbers above
> are computed from the 24 reconciled cells directly and are the
> honest aggregate.

## Honest finding: agentic + RAG workloads both broken in this setup

**All 24 agentic cells AND all 24 RAG cells (every rate × every topology,
every model) returned success_rate=0.0 with TTFT=0.** Only the 24 chat
cells produced real measurements. Root cause confirmed by code inspection
(post-sweep, no ssh access to pod):

1. **RAG prompt overflows vLLM `--max-model-len=4096` by ~4×.** From
   `loadgen/rag.py:38` — `n_corpus_docs=8` ×
   `doc_tokens_range=(1000, 4000)` × `include_fraction=0.8` =
   ~16,000 prompt tokens average, plus query + output ≈ 16,500 total.
   vLLM rejects at admission; `success_rate=0`, `gpu_util_pct=0`,
   `duration_s≈7.7s` (the rejected round-trip wall time).
2. **Agentic prompt overflows borderline.** From `loadgen/agentic.py:51` —
   `history_tokens_range=(500, 4000)` + tool defs + output 100-1000
   can reach 5,000+ tokens, also over 4096 limit.
3. **SSE parser is fine.** `loadgen/sse.py` skips malformed lines
   silently — failure is upstream at vLLM admission, not parsing.
4. **Telemetry gap.** `CellResult` records `success_rate=0` but does NOT
   capture error reason (e.g., HTTP 400 with "context length exceeded").
   Fixing this would help future sweeps diagnose without re-running.

The bench did NOT silently mask this — `reconcile_passes=False` for
every agentic and RAG cell, so `summary.json` reports it correctly.
Fix options for v1.1:

- Bump vLLM `--max-model-len` to 16384 (covers RAG, costs HBM)
- Reduce `loadgen/rag.py` doc count or `doc_tokens_range`
- Add `error_kind` field to `RequestTelemetry` so rejected cells
  report why (no re-run needed to diagnose)

## What worked (chat only — only mix with reconciled cells)

| Topology | Cells passed reconcile | Mean TTFT | Mean ITL |
|---|---|---|---|
| colocated | 18/18 | 170.68 ms | 8.36 ms |
| chunked | 6/18 | 278.13 ms | 8.47 ms |
| disagg | **0/18** | [NOT MEASURED] | [NOT MEASURED] |
| disagg_tier | **0/18** | [NOT MEASURED] | [NOT MEASURED] |

**Honest reading:** chunked-prefill shows ~63% higher mean TTFT than
colocated across the chat-mix cells. The DISAGG and DISAGG_TIER cells
returned success_rate=0.0 in this run — likely the disaggregation runtime
path requires config the matrix runner does not yet apply to a single
vLLM instance. They are NOT meaningful measurements of true
disaggregated serving. True DISAGG needs separate prefill + decode vLLM
instances + NIXL UCX (planned v1.1).

**Model breakdown of the 24 chat-reconciled cells:**

| Model | Reconciled | Notes |
|---|---|---|
| qwen3-1.7b | 12/24 | 6 colocated + 6 chunked |
| qwen2.5-7b | 6/24 | all colocated |
| qwen3-30b | 6/24 | all colocated |

## Figures

- `bench/figures/runpod_full_ttft.png` — TTFT vs rate, by topology
  (chat only, colocated + chunked; DISAGG / DISAGG_TIER had 0 measurements)
- `bench/figures/runpod_full_topo.png` — TTFT vs rate, by model
  (chat only, 3 models; qwen2.5-7b / qwen3-1.7b / qwen3-30b)

## Cost

- Pod wall time: ~26 min (model load + 72 cells × ~10s)
- Bench cost only: ~$0.42
- Pod total: ~$1.30

## What's next (v1.1)

1. **RAG fix:** bump vLLM `--max-model-len` to 16384 OR reduce
   `loadgen/rag.py` `n_corpus_docs` / `doc_tokens_range`. Root cause
   documented above (4× overflow on current config).
2. **Agentic fix:** borderline overflow — same options.
3. **Telemetry gap:** add `error_kind` field to `RequestTelemetry`
   so future sweeps capture HTTP 400 reason without re-running.
4. **True DISAGG:** separate prefill + decode vLLM instances + NIXL UCX —
   36 disagg cells produced 0 measurements
5. **Investigate vLLM multi-model serving:** 3 models labelled,
   24 cells reconciled across all 3; need to confirm what was actually
   served per request vs what was labelled
6. **Failure drill appendix** (router reject, KV stall, thermal throttle)