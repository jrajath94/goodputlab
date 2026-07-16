# GoodputLab GPU execution plan

Purpose: isolate every remaining hardware-backed task so another agent can
execute it without touching the already-finished non-GPU control-plane work.

Status as of 2026-07-15:

- Non-GPU implementation is complete enough to test locally.
- Remaining work is measurement, topology validation, and prompt-shape fixes on live GPU.
- The repo already contains the orchestrators, reconciliation gate, dashboard JSON, and result formats.

## What is already done

- Router, admission control, tier policy, spec simulator, autoscaler, metrics registry, and reconciliation logic all ship and test on CPU.
- Real single-node H100 evidence exists in:
  - `bench/results/real/`
  - `bench/results/runpod_pilot/`
  - `bench/results/runpod_full/`
- Grafana placeholder dashboard exists at `deploy/grafana/goodputlab.json`.

## Remaining GPU work

### 1. Finish the matrix honestly

Current failure: many `rag` and `agentic` cells overflow the served model's
context window. The issue is upstream at vLLM admission, not in the
aggregator or reconciler.

Required actions:

1. Increase `--max-model-len` for the real vLLM runs to a value that covers
   the shipped RAG prompts. Start with `16384`.
2. Re-run the pending cells with `BenchMatrix.run_pending`.
3. Keep using the existing reconcile gate; do not count unreconciled cells
   as measured.

Suggested command path:

```bash
export RUNPOD_VLLM_BASE_URL=http://127.0.0.1:8000/v1
python -m scripts.run_matrix --config configs/runpod_matrix.yaml
```

Acceptance:

- pending-cell count decreases to zero for the chosen sweep
- every committed result JSON has `reconcile_passes: true`
- summary artifact updates without manual editing

### 2. Validate true disaggregation instead of label-only emulation

Current limitation: some prior runs were topology-labeled but still used a
single vLLM process. That is acceptable for pipeline shakeout, not for a
world-class claim about P/D disaggregation.

Required actions:

1. Run separate prefill and decode processes.
2. Ensure NIXL / UCX transfer metrics appear in `/metrics`.
3. Re-run `make health` and the relevant matrix cells.

Acceptance:

- health gate passes using real transfer-related metrics
- sentinel still passes
- disagg result JSONs are backed by actual two-process topology

### 3. Run live autoscaler validation

The autoscaler logic is implemented and well-tested, but its core claim is
still missing a real workload-shift run.

Required actions:

1. Construct a prompt-heavy → decode-heavy traffic shift.
2. Scrape queue depth and replica decisions over time.
3. Validate:
   - no in-flight drops during role flips
   - thrash counter stays bounded
   - dwell gate suppresses ping-pong behavior

Acceptance:

- explicit plot or trace over time
- `goodputlab_role_flip_inflight_dropped_total == 0`
- controller thrash either stays zero or is explained by a documented edge case

### 4. Multi-node UCX validation

Single-node `cuda_ipc` is already exercised. The missing work is the real
cross-node path using `tcp` or RDMA-flavored UCX.

Required actions:

1. Provision two GPU pods.
2. Switch the transport away from `cuda_ipc`.
3. Re-run health and at least a minimal disagg sweep.

Acceptance:

- successful KV transfer across nodes
- sentinel passes
- reconcile still holds within the documented tolerance

## What not to redo

- Do not re-implement `core/reconcile.py`, router logic, or autoscaler logic.
- Do not replace the metrics schema unless a live-vLLM incompatibility forces it.
- Do not hand-edit result summaries; regenerate them from the recorded JSON/metrics path.

## Recommended artifact outputs

If another agent performs the GPU work, it should leave behind:

- updated `bench/results/...` JSON artifacts
- updated `docs/REPORT.md` only where new measurements actually exist
- a short run log with:
  - pod type
  - vLLM version
  - model name
  - `--max-model-len`
  - total spend
  - reconciled-cell count

## Blocking assumptions

- GPU spend is authorized outside this session.
- RunPod or equivalent credentials are already configured.
- A real DraftForge head is not required for the non-spec parts of this plan.
