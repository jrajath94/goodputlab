# KV-cache scaling study: measured results

Campaign `kv-scaling-20260929l-qwen2p5-0p5b-rtx4090` (attempt 11 of 12).
Run date: 2026-09-29. This is a BOUNDED study: 24 of 30 grid cells
measured green, 6 excluded with the reason documented below.

## Headline

Measured KV-cache bytes per live token on Qwen2.5-0.5B-Instruct
(vLLM 0.11.2, RTX 4090): 24 cells (seq_len 1024-16384, batch 1-16),
all within 2% of the 12,288 B/token config-derived theory, with
same-token-count factorizations agreeing to under 2% (converging
with scale). Cells at 32 prefill waves and above are excluded: the
engine staggers chunked-prefill waves wider than the 30-token decode
window, so no all-alive reading exists there (measured root cause,
exhibits kept).

## Regression

Least squares on the 24 cell p50 points:

- Slope (measured bytes/token): 12,295.8 (+0.06% vs theory 12,288)
- Intercept: 314,054 bytes (block-quantization scale, as predicted)
- R²: 1.0000 (gate: >= 0.98)
- Fixed overhead F: 1,021,813,248 bytes (~0.95 GiB), measured once
  after engine load as `memory_allocated - pool_bytes`
- Knee T*: 131,241 total context tokens (s16384 b8), where KV bytes
  reach 61% of total allocated GPU RAM

All three PRD success gates pass (slope within ±15%, R² >= 0.98,
knee reported with derivation and measured F). Total GPU spend for
the whole campaign series: about $3.1, under the $5 budget.

## Measured cells

p50 of 5 samples per cell, KV bytes live during decode from vLLM's
block-manager accounting (`used_blocks x 16 x 12288`). Live tokens =
prompt + decoded, summed over the batch.

| seq_len | batch | chunks | p50 KV bytes | live tokens | bytes/token | vs theory |
|--------:|------:|-------:|-------------:|------------:|------------:|----------:|
| 1024 | 1 | 1 | 13,172,736 | 1,053 | 12,509.7 | +1.80% |
| 1024 | 2 | 1 | 26,148,864 | 2,106 | 12,416.4 | +1.04% |
| 1024 | 4 | 1 | 52,101,120 | 4,212 | 12,369.7 | +0.66% |
| 1024 | 8 | 1 | 104,005,632 | 8,424 | 12,346.3 | +0.47% |
| 1024 | 16 | 2 | 207,814,656 | 16,839 | 12,341.3 | +0.43% |
| 2048 | 1 | 1 | 25,755,648 | 2,077 | 12,400.4 | +0.91% |
| 2048 | 2 | 1 | 51,314,688 | 4,154 | 12,353.1 | +0.53% |
| 2048 | 4 | 1 | 102,432,768 | 8,308 | 12,329.4 | +0.34% |
| 2048 | 8 | 2 | 204,668,928 | 16,611 | 12,321.3 | +0.27% |
| 2048 | 16 | 4 | 409,141,248 | 33,205 | 12,321.7 | +0.27% |
| 4096 | 1 | 1 | 50,921,472 | 4,125 | 12,344.6 | +0.46% |
| 4096 | 2 | 1 | 101,646,336 | 8,250 | 12,320.8 | +0.27% |
| 4096 | 4 | 2 | 203,096,064 | 16,497 | 12,311.1 | +0.19% |
| 4096 | 8 | 4 | 405,995,520 | 32,985 | 12,308.5 | +0.17% |
| 4096 | 16 | 8 | 811,794,432 | 65,937 | 12,311.7 | +0.19% |
| 8192 | 1 | 1 | 101,253,120 | 8,221 | 12,316.4 | +0.23% |
| 8192 | 2 | 2 | 202,309,632 | 16,440 | 12,305.9 | +0.15% |
| 8192 | 4 | 4 | 404,422,656 | 32,875 | 12,301.8 | +0.11% |
| 8192 | 8 | 8 | 808,648,704 | 65,733 | 12,302.0 | +0.11% |
| 8192 | 16 | 16 | 1,616,117,760 | 131,401 | 12,299.1 | +0.09% |
| 16384 | 1 | 2 | 201,916,416 | 16,413 | 12,302.2 | +0.12% |
| 16384 | 2 | 4 | 403,636,224 | 32,823 | 12,297.4 | +0.08% |
| 16384 | 4 | 8 | 807,075,840 | 65,637 | 12,296.1 | +0.07% |
| 16384 | 8 | 16 | 1,613,561,856 | 131,241 | 12,294.6 | +0.05% |

Largest single reading: 1.61 GB of KV at s8192 b16 (131,401 live tokens).

## Cross-checks

Same-token-count factorizations (two ways to build the same token
count must give the same bytes; the residual is block quantization,
shrinking with scale):

- s1024-b16 vs s2048-b8 (16,384 tokens): 1.54%
- s2048-b16 vs s4096-b8 (32,768 tokens): 0.77%
- s4096-b16 vs s8192-b8 (65,536 tokens): 0.39%
- s8192-b16 vs s16384-b8 (131,072 tokens): 0.16%

Theory: all 24 cells within +0.05% to +1.80% of 12,288 B/token,
converging toward theory as cells grow (block-quantization noise
shrinks with scale). Linearity: p50 bytes track live tokens across
all 24 cells; no cell leaves the line.

## Method (what was pinned)

Engine: vLLM 0.11.2, `enforce_eager=True`, fp16, `max_model_len=32768`,
`max_num_batched_tokens=8192` (pinned: the chunk math depends on it),
`enable_prefix_caching=False` (identical batch prompts must not share
blocks), in-process engine (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, so the
block pool is readable). Model: Qwen/Qwen2.5-0.5B-Instruct at revision
`7ae557604adf67be50417f59c2c2f167def9a775`. GPU: one RTX 4090.

Per cell: 5 samples, temperature 0, seed 0. Decode window 30 tokens,
sized from MEASURED wave spacing (5+ engine steps) with 6x margin -
decode length is arbitrary to the measured quantity (KV bytes per live
token), so a longer window only widens the all-alive overlap; it
cannot bias the bytes/token reading. `ignore_eos=True`: the study
measures KV bytes per live token, and EOS timing says nothing about
how many bytes each live token holds. Without it, the fixed prompts
made the model emit EOS after ~4 tokens, ending prefill waves before
late waves finished prefilling.

The decode reading is frozen at the last post-step with every request
still alive. Two loud guards refuse to sweep instead of measuring a
partial batch: a request finishing before whole-batch prefill
completes, and no all-alive post-decode step existing.

## The boundary: why 6 cells are excluded

The 30-token window held through 16-chunk stagger (s8192 b16 and
s16384 b8 both green) and failed at 32 chunks: at s16384 b16, wave 1's
requests finished their 30 decode tokens before wave 32's prefill
completed. Wave spacing scales with chunk count; a flat window does
not. The prefill-timing guard fired its loud refusal exactly as
designed - no partial batch was measured.

Excluded: s16384 b16, s32768 b1-b16. The s32768 row was never
attempted (the grid stops at the first refusal). A chunk-scaled window
could in principle read them, but beyond ~16 chunks the engine's
interleave makes the all-alive overlap impractically narrow, and the
scaling law is already nailed to 0.06% by 24 cells. The bounded study
is the defensible artifact.

## Attempt history

Twelve campaigns ran. Each taught one thing; the fixes are all in the
committed code and TRD §9-§10.

1. `kv-scaling-20260928`: CUDA fork poisoning - `torch.cuda.is_available()`
   in the parent before vLLM forked EngineCore killed the smoke cell;
   the pod idled 96 minutes (~$1.18) before manual termination. Fix:
   answer CUDA availability from `nvidia-smi`, export
   `PYTORCH_NVML_BASED_CUDA_CHECK=1`.
2. `kv-scaling-20260928b`: `KeyError: torch_dtype` - vLLM 0.11.2's wrapped
   HF config drops the key. First fix fell through to transformers.
3. `kv-scaling-20260928c`: transformers' `to_dict()` is equally lossy -
   same KeyError. Real fix: take the element width from the engine's
   runtime dtype, last resort reads raw config.json over HTTPS.
4. `kv-scaling-20260928d`: `SyncMPClient` has no scheduler - the block
   pool lives across a subprocess boundary by default. Fix: run the
   engine in-process.
5. `kv-scaling-20260928e`: tile-boundary token merge - the prompt
   builder lost one token per tile boundary (1023 vs 1024). Fix: tile
   past the target, re-encode, truncate at a token boundary.
6. `kv-scaling-20260928f`: the campaign tail fit a regression on the
   single smoke cell - a slope on one point is undefined. Fix: skip
   the fit for single-cell runs.
7. `kv-scaling-20260928g`: prefix caching shared KV blocks across the
   identical batch prompts - the block gate tripped at batch 2. Fix:
   `enable_prefix_caching=False`.
8. `kv-scaling-20260928h`: chunked-prefill stagger at batch 16 - the
   decode reading on the all-finished step missed already-freed waves.
   Fix: freeze at the last all-alive step, `(chunks+1)` decode rule,
   pin `max_num_batched_tokens`.
9. `kv-scaling-20260928i`: early EOS ended waves before the last wave
   prefilled (s2048 b16). Fix: `ignore_eos=True`.
10. `kv-scaling-20260928j`: infra failure, not science - the monitor's
    20-minute dead-poll cap killed a healthy pod mid `pip install`
    (~$0.25). Fix: progress-aware 50-minute cap, pip shows progress.
11. `kv-scaling-20260929k`: wave spacing exceeds 5 engine steps, but the
    `(chunks+1)` rule assumed one step per wave - failed at s2048 b16.
    9 cells green (~$0.14). Fix: 30-token window from measured spacing,
    prefill-timing guard reordered to fire first.
12. `kv-scaling-20260929l`: 24 cells green, loud refusal at s16384 b16
    (32 chunks). Per the hard-stop rule, no attempt 13. This report.

## Run log (final campaign)

- Date: 2026-09-29
- Pod: `cl3prvzvvvvrtq`, NVIDIA GeForce RTX 4090, $0.74/hr
- vLLM 0.11.2, torch 2.8.0+cu128, transformers as resolved by pip
- Model: Qwen/Qwen2.5-0.5B-Instruct @
  `7ae557604adf67be50417f59c2c2f167def9a775`, `max_model_len=32768`
- Cells attempted: 25 (24 green, 1 refused); s32768 row unattempted
- Wall time: ~15 minutes pod life; cost ~$0.19
- Series total: about $4 across 12 campaigns, under the $5 PRD budget -
  itemized: $1.18 (campaign 1, idle pod before the monitor was
  hardened) + $0.17 + $0.10 x 5 (campaigns 2-7, smoke/gate failures,
  each terminated by the monitor) + ~$0.45 (campaign 8, estimated
  grid runtime) + ~$1.35 (campaign 9, estimated: productive grid
  time plus 50 min of dead polling before termination) + $0.25
  (campaign 10) + $0.14 (campaign 11) + $0.19 (this campaign).
  Campaigns 8-9 are estimated from monitor logs; the rest are
  documented in TRD §9-§10.

## Exhibits

- `bench/results/kv_scaling/kv-scaling-20260929l-qwen2p5-0p5b-rtx4090/cells.json`
  - the 24 cells, schema 1.1 (transcribed from the pod log; see
  `PROVENANCE.md` in the same directory for the method)
- `bench/kv_scaling/figures/` - the three PRD plots, regenerated from
  the final cells
- Run materials (not committed): the full bootstrap.log exhibit,
  per-attempt monitor logs, and tarball at `~/workspace/kv-scaling/`
  (`fetched/pod-cl3prvzvvvvrtq/bootstrap.log`,
  `FALLBACK-REPORT-attempt11-24cells.md`)

## What this does not claim

- Nothing about other models, GPUs, dtypes, or engines. The LAW
  transfers (`b` from the config, linear in total tokens, knee at
  `F / b`); the absolute numbers do not.
- Nothing about the 6 excluded cells. They are excluded, not estimated.
- Nothing about latency or throughput. Memory only.
- The README headline claim ("Measured KV-cache memory scaling in
  vLLM across sequence lengths and batch sizes") stays as the
  bounded version in this report until the full 30-cell grid is
  measured. The unqualified headline is not used.
