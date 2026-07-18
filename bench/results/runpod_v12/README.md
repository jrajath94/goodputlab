# v1.2 sweep — 2026-07-17 — 54/54 reconciled, dedicated-GPU disagg

First sweep where **every topology label maps to a genuinely different
serving configuration**, run as three passes with server swaps between
them (`--topologies` filter):

| Pass | Serving config |
|---|---|
| colocated | single vLLM 0.11.2, `--no-enable-chunked-prefill`, 1× H100 |
| chunked | single vLLM, `--enable-chunked-prefill` (restart), 1× H100 |
| disagg | prefill vLLM on GPU0 + decode vLLM on GPU1 (dedicated), NixlConnector, `scripts/disagg_proxy.py` front door |

`disagg_tier` is excluded: the real LMCache gRPC wire is v1.1-deferred,
and a tier label without a tier would be label-only data.

## Headline (all 54 cells reconcile; success 1.00 everywhere)

3 topologies × 6 rates (1..32 rps) × 3 mixes (chat/rag/agentic),
Qwen2.5-7B, `--max-model-len 20480`, 5 warmup + 30 measure per cell.

| Topology | Cells | Mean TTFT | Mean ITL |
|---|---|---|---|
| colocated | 18/18 | 900 ms | 8.8 ms |
| chunked | 18/18 | 974 ms | 8.7 ms |
| **disagg (true, 2 GPU)** | 18/18 | 1,034 ms | **8.3 ms** |

## NIXL transfer evidence (decode engine, end of disagg pass)

`nixl_after_disagg_pass.txt`:

| Counter | Value |
|---|---|
| transfers | 617 |
| bytes | 30.9 GB |
| failed | 0 |
| mean transfer time | 7.7 ms |

## Honest findings

1. **Dedicated hardware fixes the collapse.** The 2026-07-17 single-GPU
   probe collapsed at 16 rps (0.73 success, 9 s TTFT). With one GPU per
   stage, every rate through 32 rps reconciles at 1.00 success — the
   contention explanation holds.
2. **Disagg still pays +134 ms mean TTFT vs colocated** (proxy hop +
   handshake + transfer) at these loads with this 7B model — a single
   H100 is simply not saturated enough for stage separation to win.
   Disagg shows the best ITL (8.3 vs 8.8 ms): decode never shares its
   GPU with prefill bursts. This is the interference-isolation benefit,
   visible but small at 7B scale.
3. **This is 2× the hardware for worse TTFT** at every measured load —
   the honest cost framing. The regime where disagg wins (long-prompt
   interference at saturation, strict ITL SLOs, bigger models) is
   documented as future work, not claimed.

## Reproduce

Pod: RunPod secure 2× H100 SXM 80GB (`u1n8efij0owar2`), $5.98/hr.

```bash
# pod: bash deploy/pod/launch_vllm.sh colocated   # then chunked, disagg-2gpu
# local, per pass (proxy port 8000 for colocated/chunked, 9100 for disagg):
export RUNPOD_VLLM_BASE_URL=https://<pod>-<port>.proxy.runpod.net/v1
python -m scripts.run_matrix --config configs/runpod_matrix_full.yaml \
  --approve-cost --topologies <topology>
```

## Run log

| Field | Value |
|---|---|
| Date | 2026-07-17 |
| Pod | 2× H100 SXM 80GB secure (`u1n8efij0owar2`, AP-IN-1), $5.98/hr |
| vLLM | 0.11.2, `--max-model-len 20480` |
| Model | Qwen/Qwen2.5-7B-Instruct |
| Bench wall | 260 s + 257 s + 248 s (three passes) |
| Bench-attributed cost | $0.64 |
| Reconciled | **54/54 (100 %)** |
