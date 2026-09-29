# KV-cache scaling study: PRD

Status: MEASURED (bounded). Campaign `kv-scaling-20260929l-qwen2p5-0p5b-rtx4090`
(2026-09-29) measured 24 of 30 grid cells; see `docs/kv_scaling/RESULTS.md`.
Every measured value below carries its method. Criterion 4 did not fully
hold (24/30, not 30/30), so this is a bounded study and the unqualified
README headline in §Goals-4 is not used.

## Problem

At long context, the KV cache is the largest memory consumer in an
LLM serving stack. Sequence length times batch size decides whether
a request mix fits on a GPU at all. GoodputLab measures latency and
goodput today, but it carries no measured evidence for how KV-cache
memory scales. Capacity questions (max batch at 32k context on a
24 GB card, where the memory knee sits for a given model) are
currently answered by guesswork.

This is also the craft that inference roles hire for. The DeepMind
Model Inference posting names "every GB of GPU RAM" work explicitly.
A measured scaling law, with its error budget, is the artifact that
proves that craft.

## Goals

1. Measure KV-cache bytes per token in vLLM for one model
   (Qwen2.5-0.5B-Instruct, fp16, HF revision pinned per TRD section
   2a) across a sequence-length by batch-size grid.
2. Fit the scaling law `total = F + b * T` and report `b`
   (bytes/token) and `F` (fixed non-KV overhead).
3. Report the knee: the smallest total context tokens `T*` where KV
   bytes exceed 50% of total allocated GPU RAM.
4. Earn the claim the repo will carry, only if the gates pass:
   "Measured KV-cache memory scaling in vLLM across sequence lengths
   and batch sizes."

## Non-goals

- Multi-node. Single GPU only.
- Paged-attention internals tuning. `block_size=16` is a fixed input
  to the error budget, not a variable to tune.
- Multi-model comparison in one campaign. vLLM serves one model per
  process; the campaign is one model.
- Other engines. vLLM only. No SGLang, TensorRT-LLM, or Dynamo
  comparison.
- Latency or throughput claims. This study is memory only.
- KV tiering and LMCache. That is the topic of `kv/tier_policy.py`
  and `kv/lmcache_client.py`, a different study about tier admission.
  Do not conflate the two. This study measures memory scaling of the
  on-GPU cache.
- Quantized weights. fp16 only.

## Success criteria

1. Measured `b` lands within ±15% of the config-derived theory
   (12,288 bytes/token for Qwen2.5-0.5B-Instruct fp16). Method:
   slope of the linear regression in `docs/kv_scaling/TRD.md` §4.
   Status: MEASURED - slope 12,295.8 B/token (+0.06% vs theory),
   campaign `kv-scaling-20260929l`, 2026-09-29.
2. Regression R² ≥ 0.98 across cell medians. Status: MEASURED -
   R² = 1.0000.
3. Knee `T*` reported with its derivation and the measured `F`.
   Status: MEASURED - T* = 131,241 total context tokens (s16384 b8),
   F = 1,021,813,248 bytes (~0.95 GiB).
4. All 30 grid cells reconciled per the gates in
   `docs/kv_scaling/EXECUTION.md`. Failed cells are kept as failure
   exhibits and excluded from claims. Status: BOUNDED - 24/30 cells
   reconciled; s16384 b16 refused by the prefill-timing guard
   (32 prefill chunks, measured root cause in RESULTS.md), the
   s32768 row unattempted. The 6 excluded cells are documented, not
   estimated.
5. Total GPU spend under $5 with a run log in the format of
   `docs/GPU_COST_OPTIMIZATION.md`. Status: MEASURED - about $4
   across 12 campaigns, itemized in `docs/kv_scaling/RESULTS.md`
   (run log; campaigns 8-9 estimated from monitor logs).
6. The headline claim in §Goals-4 may appear in the README only after
   criteria 1-5 hold. Status: NOT USED - criterion 4 holds only in
   bounded form, so the README carries no unqualified headline. The
   bounded claim lives in `docs/kv_scaling/RESULTS.md`.

## Deliverables

- `docs/kv_scaling/PRD.md`, `TRD.md`, `EXECUTION.md` (this set).
- `bench/results/kv_scaling/`: one JSON per cell plus `summary.json`
  and `preflight.json`, following the repo's cell-artifact
  conventions.
- Run log: pod id, GPU type, hourly rate, vLLM version, model,
  `max_model_len`, cells attempted/reconciled/failed, wall minutes,
  estimated and actual cost.
- After measured cells exist: a short measured section in
  `docs/REPORT.md`. Do not touch `docs/REPORT.md` before then.

## Open questions

- vLLM version pin. CLOSED: pinned at `vllm==0.11.2`
  (`~/workspace/kv-scaling/bootstrap_kv.sh` installs exactly this
  into the pod venv; the installed version is printed at bootstrap
  and recorded in the run log). Allocator behavior drifts across
  versions, so a rerun on any other vLLM version is a new campaign
  id, never a silent remeasurement.
- Whether the knee `T*` falls inside the 32k sweep range for this
  model. The TRD derives `T* = F / b`; the measured `F` decides.
  If the knee is out of reach, that is itself the finding.
