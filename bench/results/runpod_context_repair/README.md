# Context repair probe — 2026-07-17 — RAG/agentic overflow fixed

The 72-cell (2026-07-14) and v1.1 (2026-07-16) sweeps lost every RAG
cell to HTTP 400 context overflow (`--max-model-len` 4096, then 16384).
The local prompt preflight measured the RAG worst case at 18,539
prompt+output tokens; this probe verifies the fix — vLLM launched with
`--max-model-len 20480` — with exactly 2 cheap cells before any larger
spend.

## Results

| Cell | Success | Mean TTFT | Mean ITL | Reconciles |
|---|---|---|---|---|
| colocated @ 4 rps, rag | 1.00 | 1,472 ms | 8.3 ms | yes |
| colocated @ 4 rps, agentic | 1.00 | 513 ms | 7.0 ms | yes |

Zero HTTP 400s. The preflight-predicted budget (worst RAG request
18,533 tokens on this seed set, see `preflight.json`) fits 20480 with
~2 K headroom. RAG and agentic mixes are now eligible for any future
focused sweep at `max_model_len: 20480`.

## Run log

| Field | Value |
|---|---|
| Date | 2026-07-17 |
| Pod | RunPod secure, 1× H100 SXM 80GB (`ewzxmo3mcttjm8`), $2.99/hr |
| vLLM | 0.11.2, `--max-model-len 20480`, chunked prefill ON |
| Model | Qwen/Qwen2.5-7B-Instruct |
| Bench wall | 9.3 s |
