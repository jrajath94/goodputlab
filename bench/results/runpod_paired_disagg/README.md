# Paired disagg probe — 2026-07-17 — FIRST TRUE P/D EVIDENCE

First run in this repository where disagg-labeled cells were served by
**separate prefill and decode vLLM processes with real NIXL KV transfer**.
Every prior disagg row (Run 1, runpod_full, runpod_v11) was label-only
(single process); see `docs/GAP_REPORT.md` and the per-directory READMEs.

## Topology (single pod, single H100)

- **colocated**: one vLLM 0.11.2 process on `:8000`,
  `--no-enable-chunked-prefill`, `--max-model-len 20480`.
- **disagg**: two vLLM 0.11.2 processes sharing the one H100 —
  prefill `:8100` (`kv_role=kv_producer`) + decode `:8200`
  (`kv_role=kv_consumer`), both `--gpu-memory-utilization 0.42`,
  `--max-model-len 8192`, `kv_connector=NixlConnector`, distinct
  `VLLM_NIXL_SIDE_CHANNEL_PORT` (5601/5602) — this avoids the
  2026-07-16 single-pod ZMQ collision. Front door:
  `scripts/disagg_proxy.py` on `:9100` speaking the vLLM
  `kv_transfer_params` protocol (commit `284ef6b`).

## Transfer evidence (decode engine `/metrics`, this run)

Captured in `nixl_before.txt` / `nixl_after.txt`:

| Counter | End of run |
|---|---|
| `nixl_xfer_time_seconds_count` | 30 transfers |
| `nixl_bytes_transferred_sum` | 2.11 GB |
| `nixl_num_failed_transfers_total` | 0 |
| mean transfer time | 16.9 ms |

## Results

| Cell | Success | Mean TTFT | Mean ITL | Reconciles |
|---|---|---|---|---|
| colocated @ 4 rps | 1.00 | 735 ms | 8.4 ms | yes |
| **disagg @ 4 rps** | 1.00 | 919 ms | 7.8 ms | **yes** |
| colocated @ 16 rps | 1.00 | 775 ms | 7.2 ms | yes |
| disagg @ 16 rps | 0.73 | 8,979 ms | 330 ms | **no** (kept as evidence) |

## Honest findings

1. **At 4 rps, true P/D costs +25 % TTFT vs colocated** (919 vs 735 ms)
   on a single shared GPU: the proxy hop plus the NIXL handshake are pure
   overhead when one process could have served the request. ITL is at
   parity (7.8 vs 8.4 ms) — decode is unaffected once KV lands.
2. **At 16 rps, single-GPU P/D collapses** (73 % success, 9 s mean TTFT,
   client timeouts). NIXL itself did not fail (0 failed transfers,
   16.9 ms mean); the two processes time-slice one H100's SMs, so
   prefill and decode contend instead of pipelining. Disaggregation
   needs dedicated hardware per stage — on shared hardware it is
   strictly worse than colocation at load. This is the core
   "when disaggregation does NOT pay" datapoint for `docs/REPORT.md`.
3. The unreconciled 16 rps disagg cell is retained on disk
   (`reconcile_passes: false`) as the failure exhibit; it must never be
   averaged into topology tables.

## Limitations

- One GPU, one model (Qwen2.5-7B-Instruct), chat mix only, n=12
  measured requests per cell.
- Both P/D stages share the GPU: this measures *protocol* overhead and
  *contention*, not the steady-state benefit of disagg on dedicated
  prefill/decode hardware (that needs a 2-GPU run; see
  `docs/GPU_EXECUTION_PLAN.md` §4).

## Run log

| Field | Value |
|---|---|
| Date | 2026-07-17 |
| Pod | RunPod secure, 1× H100 SXM 80GB (`ewzxmo3mcttjm8`, AP-IN-1), $2.99/hr |
| vLLM | 0.11.2 |
| Model | Qwen/Qwen2.5-7B-Instruct (`goodputlab-model`) |
| Proxy | `scripts/disagg_proxy.py` @ `284ef6b` |
| Cells | 4 (2 reconciled pairs + 1 kept failure) |
| Pod wall-clock (whole ladder session) | ~53 min |
| Pod spend (whole ladder session) | ~$2.64 |
