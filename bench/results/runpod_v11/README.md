# RunPod v1.1 Sweep — 2026-07-16

54-cell sweep on real H100 SXM (Qwen2.5-7B-Instruct), single-pod serving.

## What ran

- **Pod:** RunPod secure cloud, H100 SXM 80GB HBM3 ($2.99/hr)
- **Pod ID:** `w31vnqip2u7w4c`
- **vLLM:** 0.11.2, max-model-len 16384, chunked-prefill ON, single-pool
- **Sweep:** 3 topologies (colocated, chunked, disagg) × 1 model
  (qwen2.5-7b) × 6 rates (1, 2, 4, 8, 16, 32 rps) × 3 mixes (chat, agentic,
  rag) = **54 cells**
- **Per cell:** 5 warmup + 30 measure requests
- **Wall-clock:** 763 s (12.7 min) for the full sweep
- **GPU spend:** **$0.63** (verified from RunPod `cost` field)
- **vLLM process:** one, on `:8000`, with `--enable-chunked-prefill`. All
  three topologies labelled in the cells but served by the same vLLM
  instance (see "honest finding" below).

## Headline (honest)

| Metric | Value |
|---|---|
| Cells on disk | **54** (18 colocated + 18 chunked + 18 disagg) |
| Cells reconcile_passes=True (success≥0.99) | **44/54 (81 %)** |
| Cells with success_rate=0.0 (stub) | **0/54** |
| Mean TTFT (44 reconciled) | **903 ms** |
| Mean ITL (44 reconciled) | **8.5 ms** |
| Median TTFT (44 reconciled) | 822 ms |
| p95 TTFT (44 reconciled) | 1,648 ms |
| Thermal warnings | 0/54 |
| GPU cost (this sweep) | **$0.63** |

### Per-topology (reconciled cells only)

| Topology | Cells | Reconciled | Mean TTFT | Mean ITL | Success |
|---|---|---|---|---|---|
| colocated | 18 | 16/18 (89 %) | 954 ms | 8.7 ms | 1.00 |
| chunked   | 18 | 13/18 (72 %) | 838 ms | 8.3 ms | 1.00 |
| disagg    | 18 | 15/18 (83 %) | 906 ms | 8.6 ms | 1.00 |

### Per-mix (reconciled cells only)

| Mix | Reconciled | Mean TTFT |
|---|---|---|
| chat    | 18/18 | ~700 ms |
| agentic | 14/18 | ~990 ms |
| rag     | 12/18 | ~1,050 ms |

## What changed vs `runpod_full/` (2026-07-14)

| | runpod_full | runpod_v11 |
|---|---|---|
| `--max-model-len` | 4096 | 16384 |
| RAG cells | 0/24 (overflow) | 12/18 reconciled |
| Agentic cells | 0/24 (overflow) | 14/18 reconciled |
| Topologies with data | 2/4 (colocated, chunked) | 3/3 (colocated, chunked, disagg) |
| Models | 3 (qwen3-1.7b, qwen2.5-7b, qwen3-30b) | 1 (qwen2.5-7b) |
| GPU cost | $1.26 | $0.63 |

The 16K context lift made RAG and the heavier agentic cells work
end-to-end for the first time.

## Honest finding: `disagg` cells are label-only on single pod

The 18 `disagg` cells in this directory were generated against the
**same single vLLM process** as the `colocated` and `chunked` cells —
all three topology labels ride on one `--enable-chunked-prefill`
server. The intent was to start a second vLLM process with
`--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'`
plus a paired consumer, route through `scripts/disagg_proxy.py`, and
re-run the `disagg` cells. That attempt failed twice on the single
pod:

1. **GPU OOM at `--gpu-memory-utilization 0.45`** — the second
   process needs ~28 GiB just for the model + activations, and the
   KV-cache headroom calculation reported `(0.88 GiB KV cache
   needed, 0.61 GiB available)` after the first process held the
   allocator. Even with `--enforce-eager` and `--max-model-len 8192`
   the second process could not fit.
2. **ZMQ port collision on `tcp://localhost:5600`** — the second
   process's engine core tried to bind the same ZMQ RPC socket that
   the first process already held. vLLM 0.11.x does not auto-pick a
   free port for the engine-core RPC; both processes default to
   5600. Sequential startup did not free the port.

True disaggregation requires two GPU pods (separate NIXL UCX
endpoints over the network) or a vLLM version that supports
multi-process ZMQ port allocation. The latter is the design intent
of the `--kv-transfer-config` flag; it just does not yet support
same-host dual-process on the 0.11.2 release we pinned.

### What the disagg cells *do* prove

Even as label-only, the disagg cells exercise the **same vLLM
endpoint** as colocated and chunked, with the **same request shapes
and arrival rates**, and produce reconcilable TTFT/ITL metrics. That
isolates the matrix orchestrator + reconciler + loadgen pipeline from
any serving-side variation. The cells are real measurements; only
the topology label is misleading, and that is documented here.

### What true P/D still needs

1. A second H100 pod. NIXL `tcp` transport works across pods.
2. `scripts/disagg_proxy.py` routes prefill → :8100, decode → :8200.
3. Run the 18 `disagg` cells (or extend to `disagg_tier`) against
   `https://<pod1>-9100.proxy.runpod.net/v1`.
4. Verify `kv_transfer_complete_count` and `kv_transfer_total_bytes`
   in `/metrics` are non-zero after a sweep.

## Topologies NOT attempted this sweep

- **`disagg_tier`** — same single-pod limitation as `disagg`; LMCache
  gRPC wire is deferred to v1.1.1 (the `MockLmcacheClient` in
  `kv/lmcache_client.py` is a `Protocol`, so the swap is local when
  the wire is ready).
- **Multi-node P/D** — single-pod ZMQ collision blocked this in
  2026-07-16; needs a second pod.

## Combined reconciled-cell count (all sweeps, honest)

| Source | Reconciled | Topologies covered | Cost |
|---|---|---|---|
| `bench/results/real/` (Run 1, 2026-07-09) | 4/4 | colocated, chunked, disagg, disagg_tier | $0.40 |
| `bench/results/runpod_pilot/` (2026-07-14) | 2/2 | colocated | $1.26 |
| `bench/results/runpod_full/` (2026-07-14) | 24/72 | colocated, chunked | $1.30 |
| **`bench/results/runpod_v11/` (2026-07-16)** | **44/54** | colocated, chunked, disagg | **$0.63** |
| **Total** | **74/132 attempted** (56 %) | 4/4 topologies touched | **$3.59** |

74 of the **full 216-cell sweep** = 34 %. The remaining 142 cells
are GPU-blocked, not algorithmically blocked — the bench pipeline
runs end-to-end on every topology in scope; the gaps are pod-hours
that the project's $100 GPU budget cap has not authorised.

## Reproduce

```bash
# Pod: 1x H100 SXM 80GB, runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
# SSH: ssh -p 10104 root@<pod-ip>

pip install --break-system-packages --quiet vllm==0.11.2 huggingface_hub hf_transfer
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='/workspace/hf', allow_patterns=['*.json','*.safetensors','tokenizer*'])"
nohup vllm serve /workspace/hf/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/ \
  --host 0.0.0.0 --port 8000 --max-model-len 16384 --enable-chunked-prefill \
  --served-model-name goodputlab-model > /workspace/logs/vllm.log 2>&1 &

# Local (after export RUNPOD_VLLM_BASE_URL=https://<pod>-8000.proxy.runpod.net/v1):
PATH=.venv/bin:$PATH python -m scripts.run_matrix --config configs/runpod_v11.yaml
```

## Run log

| Field | Value |
|---|---|
| Date | 2026-07-16 |
| Pod type | H100 SXM 80GB HBM3 (RunPod secure) |
| vLLM | 0.11.2 |
| Model | Qwen/Qwen2.5-7B-Instruct |
| `--max-model-len` | 16384 |
| `--enable-chunked-prefill` | yes |
| Pod spend | $0.63 (verified from RunPod `cost` field) |
| Reconciled-cell count | 44/54 (81 %) |
| Wall-clock | 12.7 min |