# KV-cache scaling study: runbook

Read `docs/kv_scaling/PRD.md` and `TRD.md` first. This doc is the
execution order. Every GPU step climbs the frugal ladder in
`docs/GPU_COST_OPTIMIZATION.md`: nothing expensive runs before the
cheap rung below it passes.

## Rung 0: local dry-run ($0)

Build a synthetic block manager in pure Python: it allocates
`ceil(tokens/16)` blocks per sequence and frees on completion, with
a known injected bytes/token. Run the full analysis pipeline from
TRD §4 against it (per-cell p50, regression, knee, gates).

Gate: the recovered slope lands within block-quantization bounds of
the injected value and R² >= 0.98. This validates the math before
any spend. If the pipeline cannot recover a known slope from clean
synthetic data, it will not recover one from a GPU.

## Rung 1: pod and engine smoke (<$0.50)

Pod: 1x RTX 4090 24GB, community. Planning assumption $0.34-$0.69/hr
from the repo's cost table (2026-07 figures). The pod page is the
source of truth at launch; recompute if it differs.

On the pod:

```bash
nvidia-smi
python3 --version            # 3.11+
pip install "vllm==<pinned>" # pin it; record the version in the run log
python3 -c "import torch; print(torch.cuda.is_available())"
```

Launch one engine and run a single smoke cell:

```bash
python3 -m bench.kv_scaling \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --revision <pinned-hf-commit-sha> \
  --seq-lens 1024 --batches 1 --n-samples 5 \
  --max-model-len 32768 --enforce-eager \
  --out bench/results/kv_scaling
```

The `--revision` value is the HuggingFace commit SHA of the model
snapshot (recorded in the run log and campaign JSON per TRD
section 2a). Pin it; never float on `main`.

Gate: the engine loads, block accounting is readable, one cell JSON
lands on disk with p50/p95, and the regression code runs. Promote
only on a clean smoke.

## Rung 2: full grid

Command shape (implementers fill in the module path):

```bash
python3 -m bench.kv_scaling \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --revision <pinned-hf-commit-sha> \
  --seq-lens 1024,2048,4096,8192,16384,32768 \
  --batches 1,2,4,8,16 \
  --n-samples 5 \
  --max-model-len 32768 --enforce-eager \
  --out bench/results/kv_scaling \
  --approve-cost
```

Repo conventions the script must follow:

- Print a cost preflight before any GPU work: pending cells,
  estimated wall time, hourly rate, estimated cost, output dir.
  Refuse to start without `--approve-cost` (smoke configs excepted).
- Resume by default: skip cells whose JSON already exists and
  validates. Never overwrite a good cell silently.
- Stop at the first cell that fails a reconciliation gate. Failed
  cells stay on disk as failure exhibits with `reconcile_passes:
  false`; they are never counted in the claim.
- Write `preflight.json` and `summary.json` into the output dir.

## Cost preflight (estimate, not measured)

From `docs/GPU_COST_OPTIMIZATION.md`:

    estimated_cost = hourly_rate x (setup_min + run_min + teardown_min) / 60

Setup 12 min (provision, pip install, model pull of ~1 GB from HF).
Run ~30 min: 150 samples on a 0.5B model; the long-context cells
dominate. Teardown 2 min. Total ~44 min.

- At $0.44/hr: ~$0.32.
- At the top of the repo's 4090 band ($0.69/hr): ~$0.51.

Expected total is under $5 with wide margin. If the pod page shows a
higher rate, recompute before launch. The largest cost risk is an
idle pod, not the cells: tear down immediately when the grid
finishes.

## Reconciliation checks (gates)

- Per cell: n >= 5 valid samples, p50 and p95 recorded.
- Regression R² >= 0.98 across cell p50 points.
- Slope within ±15% of config-derived theory (12,288 bytes/token
  for Qwen2.5-0.5B-Instruct fp16).
- Intercept within block-quantization tolerance of zero.
- Knee `T* = F / b` reported with the measured `F` and its method.
- Any gate failure stops the run. The failing cell is kept as an
  exhibit and excluded from claims, per repo convention.

## Failure modes and what to do

- **OOM at 32k x batch 16.** Check `nvidia-smi` headroom before
  launch. Reduce batch or `max_model_len` and record the change.
  Do not debug on the meter: fix the config, re-run the smoke.
- **Flat slope near zero.** The block-accounting reader is wrong
  (e.g. reading free blocks instead of used). Stop, fix the reader,
  re-run the Rung 0 dry-run. Do not re-run the grid hoping it fixes
  itself.
- **R² below 0.98.** Usually fragmentation or a prefill-spike sample
  leaking into steady-state measurement. Inspect the per-sample
  trace, drop a contaminated sample only with a recorded reason, or
  raise n. If the cause is unclear, stop and write a run log.
- **Spot preemption.** Per-cell JSONs on disk are kept. Resume
  re-runs only the missing cells.
- **vLLM version behaves differently.** Metrics get renamed across
  versions. Adapt the reader, record the version and the change in
  the run log.
- **First unreconciled cell.** Stop the run per the repo's
  stop-on-unreconciled default. Investigate before spending more.

## After measurement

1. Replace the PENDING markers in PRD/TRD with measured values and
   their methods.
2. Add a short measured section to `docs/REPORT.md`. Do not rewrite
   existing sections.
3. The README headline claim ("Measured KV-cache memory scaling in
   vLLM across sequence lengths and batch sizes.") may be used only
   after all gates pass.
4. Write the required run log: date, pod id, GPU type, hourly rate,
   vLLM version, model, `max_model_len`, cells
   attempted/reconciled/failed, wall minutes, estimated and actual
   cost, and the reason for the next run or the stop.

## Conventions implementers must respect

- `kv/` is KV TIERING (LMCache admission policy), a different topic.
  This study measures on-GPU KV memory scaling. Do not mix the two
  in code, docs, or claims.
- Additive only. Do not edit, reformat, or move existing files.
- Every number carries model, vLLM version, and hardware context
  (repo evidence policy). Unmeasured claims stay marked PENDING.
- Per-cell JSONs are immutable once written; resume never
  overwrites.
