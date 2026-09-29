# Provenance: campaign kv-scaling-20260929l-qwen2p5-0p5b-rtx4090

## What this directory holds

`cells.json` — the 24 green cells of attempt 11, transcribed from the
pod's own per-sample readings in the bootstrap.log exhibit
(`~/workspace/kv-scaling/fetched/pod-cl3prvzvvvvrtq/bootstrap.log`).

## Why transcription, not the pod's native artifacts

The grid's 25th cell (s16384 b16, 32 prefill chunks) tripped the
prefill-timing guard, which correctly refused to sweep. Per the run
design, a refused grid keeps partial cells as exhibits and the pod is
terminated after the monitor fetches the log — `results.tgz` was never
packaged, so the pod's native per-cell JSONs are gone with the pod.

The bootstrap.log contains every per-sample reading the native JSONs
would have carried: for each of the 24 green cells, 5 prefill readings
(blocks used, tokens live) and 5 decode readings (blocks used, tokens
live), plus the pod's own computed p50 per cell and the F (baseline)
measurement. Nothing else feeds this file.

## Transcription method

`/tmp/reconstruct_cells.py` (kept with the run materials, not committed):

- For each cell section, takes the 5 decode readings:
  `kv_bytes_sample = used_blocks x 16 x 12288`.
- `total_tokens` = live tokens at the decode reading (identical across
  all 5 samples of every cell — asserted, not assumed).
- Cross-checks: the p50 recomputed from the 5 samples matches the pod's
  own printed p50 for all 24 cells, byte for byte; every decode reading
  is >= its prefill reading (asserted).
- Regression (least squares on the 24 p50 points) and knee computed with
  the same definitions as `bench/kv_scaling/results.py`.

`method` on each cell records the transcription. The refusal at
s16384 b16 is documented in `docs/kv_scaling/RESULTS.md`, not hidden.

## Run identity

- Campaign: kv-scaling-20260929l-qwen2p5-0p5b-rtx4090 (attempt 11)
- Pod: cl3prvzvvvvrtq, NVIDIA GeForce RTX 4090, $0.74/hr
- Model: Qwen/Qwen2.5-0.5B-Instruct @ 7ae557604adf67be50417f59c2c2f167def9a775
- vLLM 0.11.2, enforce_eager, max_model_len=32768,
  max_num_batched_tokens=8192, enable_prefix_caching=False,
  ignore_eos=True, decode window 30 tokens, 5 samples/cell, seed 0
- Code tarball sha256 (recorded on-pod):
  2c610bdf6d7ba0e19297df2a66b846379959dd273635ae2c9d122d0bf4933c3c
  (byte-identical to the committed `bench/kv_scaling/` tree)
- Run date: 2026-09-29
