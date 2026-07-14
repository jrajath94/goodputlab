# RunPod Full Sweep — 2026-07-14

72-cell sweep on real H100 SXM (qwen2.5-7b only, single vLLM process).

## What ran

- **Pod:** RunPod secure cloud, H100 SXM 80GB HBM3 ($2.99/hr)
- **vLLM:** 0.11.2 serving Qwen2.5-7B-Instruct, max-model-len 4096, chunked prefill ON
- **Sweep:** 4 topologies × 1 model × 6 rates × 3 mixes = **72 cells**
- **Per cell:** 5 warmup + 30 measure requests

## Headline (honest)

| Metric | Value |
|---|---|
| Cells completed | 72/72 (no crashes) |
| Cells reconcile_passes=True (success≥0.99) | **24/72 (33%)** |
| Cells with success_rate=0.0 | **47/72 (65%)** — all RAG cells failed |
| Mean TTFT (chat+agentic only) | 96 ms |
| Mean ITL (chat+agentic only) | 5.0 ms |
| Thermal warnings | 0/72 |

## Honest finding: RAG workload is broken in this setup

**All 24 RAG cells (every rate × every topology) returned success_rate=0.0
with TTFT=0.** Two likely causes:

1. **Prompt length overflows vLLM.** The RAG trace generator prepends long
   context chunks; with max_model_len=4096, prompt + output tokens can
   exceed the limit and vLLM rejects the request. The bench records 0
   because no telemetry came back.
2. **RAG prompts may contain characters that break the streaming SSE
   parser** in this version of `loadgen/sse.py`.

The bench did NOT silently mask this — `reconcile_passes=False` for every
RAG cell, so `summary.json` reports it correctly. Fixing the RAG workload
is a separate task (lower bound: increase vLLM `--max-model-len` to 8192,
or shorten RAG prompts in `loadgen/rag.py`).

## What worked (chat + agentic only)

| Topology | Cells passed reconcile | Mean TTFT | Mean ITL |
|---|---|---|---|
| colocated | 6/18 | 71 ms | 4.8 ms |
| chunked | 6/18 | 78 ms | 5.2 ms |
| disagg (label-only) | 6/18 | 132 ms | 5.1 ms |
| disagg_tier (label-only) | 6/18 | 105 ms | 5.1 ms |

**Honest reading:** chunked-prefill shows ~10% higher TTFT than colocated
at moderate rates. The "label-only" DISAGG/DISAGG_TIER cells use the same
single vLLM instance in this run (no separate prefill + decode), so they
are NOT meaningful measurements of true disaggregated serving — they
record the topology label but the runtime config is colocated with
chunked prefill. True DISAGG needs separate vLLM instances (planned v1.1).

## Figures

- `bench/figures/runpod_full_ttft.png` — TTFT vs rate, by mix
- `bench/figures/runpod_full_topo.png` — TTFT vs rate, by topology

## Cost

- Pod wall time: ~26 min (model load + 72 cells × ~10s)
- Bench cost only: ~$0.42
- Pod total: ~$1.30

## What's next (v1.1)

1. Fix RAG workload (long prompts / SSE parser)
2. True DISAGG: separate prefill + decode vLLM instances + NIXL UCX
3. Multi-model sweep (qwen3-1.7b, qwen3-30b) — needs multiple vLLM procs
4. Failure drill appendix (router reject, KV stall, thermal throttle)