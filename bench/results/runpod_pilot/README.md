# RunPod Pilot Sweep — 2026-07-14

First end-to-end bench on real H100 SXM. 2 cells, ~$0.01 GPU cost.

## What ran

- **Pod:** RunPod secure cloud, H100 SXM 80GB HBM3 ($2.99/hr)
- **Image:** runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
- **vLLM:** 0.11.2 serving Qwen/Qwen2.5-7B-Instruct, max-model-len 4096
- **Topology:** colocated (single pool, no P/D split)
- **Model:** qwen2.5-7b
- **Rates:** 4 rps, 8 rps
- **Mix:** chat (loadgen/chat.py)
- **Per cell:** 5 warmup + 30 measure requests, deterministic seed

## Measured numbers

| Cell | rate_rps | mean_ttft_ms | p95_ttft_ms | mean_itl_ms | success_rate | duration_s | gpu_temp_c | gpu_util_pct |
|---|---|---|---|---|---|---|---|---|
| colocated__qwen2.5-7b__rate-4__chat  | 4 | 141.10 | 309.22 |  7.74 | 1.00 | 11.32 | 37 | 89 |
| colocated__qwen2.5-7b__rate-8__chat  | 8 | 136.24 | 214.46 |  9.02 | 1.00 |  4.41 | 39 | 90 |

- All cells: `reconcile_passes=true` (success_rate ≥ 0.99 gate)
- Thermal: no warnings (THRESHOLD = 80°C, measured 37-39°C at 89-90% util)
- Cache hit rate: 0.0 (correct — colocated topology, no KV-tier)

## Wall clock + cost

- Per-cell wall: 4-11s (warmup + measure)
- Total sweep wall: 15.6s
- Bench cost: $0.0130 (15.6s × $2.99/hr ÷ 3600) — was incorrectly stated as $0.0079 in earlier draft
- Pod total (incl. vLLM install + model load): ~25 min, ~$1.25
- **Effective bench cost per cell: ~$0.004**

## What this proves

1. **Pipeline E2E works:** YAML → MatrixSpec → BenchMatrix → VllmHttpClient →
   vLLM server → RequestTelemetry → CellResult JSON → summary.json.
2. **Reconcile gate works:** all 2 cells passed (success_rate=1.0 ≥ 0.99).
3. **Thermal source works:** nvidia-smi reports realistic 37-39°C @ 89-90% util.
4. **Resume-safety works:** idempotent JSON write + corrupt-self-heal.

## What this does NOT prove (still need full sweep)

1. **DISAGG topology** (P/D split, NIXL UCX KV transfer) — pilot only ran colocated.
2. **DISAGG_TIER topology** (KV-tier with LMCache) — needs LMCache setup.
3. ~~**Multi-model** (qwen3-1.7b, qwen3-30b) — pilot only ran qwen2.5-7b.~~ *Done in full sweep — see runpod_full/README.md (12 qwen3-1.7b + 6 qwen2.5-7b + 6 qwen3-30b reconciled).*
4. **RAG + agentic mixes** — pilot only ran chat. *Full sweep also failed both — both at 0% success, see runpod_full README "Honest finding" section.*
5. **Steady-state TTFT at 16+ rps** — pilot capped at 8 rps. *Full sweep covered rates up to 32 rps.*
6. **Failure modes** (router reject, KV stall, nvidia-smi stall).

## Next step

Full sweep ran (reduced to 72 cells, 3 models): see
`bench/results/runpod_full/README.md`. 24/72 cells reconciled
(chat-only mix; agentic + RAG both fully failed).
Total cost: **$2.56** ($1.26 pilot + $1.30 full sweep).

True DISAGG (separate prefill + decode vLLM + NIXL UCX) and a
failure drill appendix remain the largest v1.1 follow-ups.