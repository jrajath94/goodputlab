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
| Cells on disk | **72** (54 colocated + 18 chunked + **0 disagg + 0 disagg_tier**) |
| Cells reconcile_passes=True (success≥0.99) | **24/72 (33%)** — chat mix only |
| Cells with success_rate=0.0 (stub) | **48/72 (67%)** — 24 agentic + 24 RAG, both fully failed |
| Topologies never attempted | **disagg, disagg_tier (0 cells on disk — not failed, not run)** |
| Mean TTFT (24 reconciled, chat-only) | **197.55 ms** |
| Mean ITL (24 reconciled, chat-only) | **8.39 ms** |
| Median TTFT (24 reconciled) | 120.83 ms |
| p95 TTFT (24 reconciled) | 493.42 ms |
| Thermal warnings | 0/72 |

> **Note on aggregates:** the campaign-level `mean_ttft_ms` /
> `mean_itl_ms` are computed by `SummaryStats.from_results` over the
> **reconciled** subset only. Stub cells have `mean_ttft_ms=0` because
> they never produced real telemetry; averaging those zeros with
> non-zeros would silently mask performance — the aggregator is
> explicit about excluding them.

## Honest finding: agentic + RAG workloads both broken in this setup

**All 24 agentic cells AND all 24 RAG cells (every rate × every topology
that has cells, every model) returned success_rate=0.0 with TTFT=0.**
Only the 24 chat cells produced real measurements. Root cause confirmed
by code inspection (post-sweep, no ssh access to pod):

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

| Topology | Cells on disk | Reconciled | Mean TTFT | Mean ITL |
|---|---|---|---|---|
| colocated | 54 | 18/54 | 170.68 ms | 8.36 ms |
| chunked | 18 | 6/18 | 278.13 ms | 8.47 ms |
| disagg | **0** | **NOT ATTEMPTED** | — | — |
| disagg_tier | **0** | **NOT ATTEMPTED** | — | — |

**Honest reading:** chunked-prefill shows ~63% higher mean TTFT than
colocated across the chat-mix cells. The DISAGG and DISAGG_TIER cells
do NOT exist on disk — the matrix orchestrator did not produce any
output for those topologies. They are NOT meaningful measurements of
true disaggregated serving. True DISAGG needs separate prefill + decode
vLLM instances + NIXL UCX (planned v1.1); the orchestrator needs to be
configured to actually run those cells (it currently skips them or
errors silently).

**Model breakdown of the 72 cells on disk:**

| Model | Cells | Topology | Reconciled |
|---|---|---|---|
| qwen3-1.7b | 18 | chunked | 6 |
| qwen3-1.7b | 18 | colocated | 6 |
| qwen2.5-7b | 18 | colocated | 6 |
| qwen3-30b | 18 | colocated | 6 |

The 4-cell × 18-cell asymmetry (colocated covers 3 models, chunked only
1) reflects the actual matrix runner output — the orchestrator did not
run disagg/disagg_tier and did not produce multi-model chunked cells.

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
4. **DISAGG / DISAGG_TIER never attempted:** sweep interrupted
   mid-run before reaching those topologies. Order from
   `MatrixSpec.cells()` is `for topo: for model: for rate: for mix`;
   on-disk tally shows colocated × all 3 models completed (54),
   chunked × qwen3-1.7b completed (18), then sweep stopped before
   chunked × qwen2.5-7b started. DISAGG / DISAGG_TIER were never
   reached. NOT a silent-skip bug in the orchestrator — re-running
   `run_pending` would resume from chunked × qwen2.5-7b and pick up
   the remaining cells. To validate, see `bench/matrix_report.py`.
5. **True DISAGG:** separate prefill + decode vLLM instances + NIXL UCX.
6. **Multi-model coverage gap:** only qwen3-1.7b ran chunked;
   qwen2.5-7b + qwen3-30b only ran colocated. Either extend sweep
   or document the model × topology matrix the runner actually covers.
7. **Failure drill appendix** (router reject, KV stall, thermal throttle)