# Smoke cell — 2026-07-17 — ladder rung 2

Single gate-exempt health cell proving pod + vLLM + client + reconciler
before any paid rung (`configs/runpod_smoke.yaml`, `smoke: true`).

| Cell | Success | Mean TTFT | Mean ITL | Reconciles |
|---|---|---|---|---|
| colocated @ 4 rps, chat | 1.00 | 731 ms | 7.9 ms | yes |

Pod: RunPod secure 1× H100 SXM 80GB (`ewzxmo3mcttjm8`), vLLM 0.11.2,
`--max-model-len 20480`, Qwen/Qwen2.5-7B-Instruct. Bench wall 5.3 s.
