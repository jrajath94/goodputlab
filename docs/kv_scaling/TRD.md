# KV-cache scaling study: TRD

Status: theory only. No measurements exist yet. Worked examples below
are arithmetic on published model configs, not measured data.

## 1. The law, from first principles

During autoregressive decoding, each token's key vector and value
vector are kept in GPU memory so later tokens can attend to them
without recomputation. Per token, per layer, per KV head, the cache
holds one key vector and one value vector, each of `d_head`
elements. Bytes per token:

    b = 2 x L x H_kv x d_head x bytes_per_element

- `2`: one K and one V per position.
- `L`: `num_hidden_layers`.
- `H_kv`: `num_key_value_heads`, the GQA head count. This is the
  point the derivation hinges on: with grouped-query attention,
  query heads share KV heads, and the cache stores one K/V per KV
  head, not per query head. Using `num_attention_heads` here
  overstates the cache by the GQA ratio.
- `d_head = hidden_size / num_attention_heads`.
- `bytes_per_element`: 2 for fp16/bf16.

All four inputs come from the HuggingFace config. The measurement
script must read them from the config at runtime. Never hardcode
them.

## 2. Worked examples

Qwen2.5-0.5B-Instruct (fp16):

    L = 24, H_kv = 2, hidden_size = 896, H_q = 14, d_head = 896/14 = 64
    b = 2 x 24 x 2 x 64 x 2 = 12,288 bytes/token (~12 KiB/token)

Llama-3.2-1B (fp16):

    L = 16, H_kv = 8, hidden_size = 2048, H_q = 32, d_head = 2048/32 = 64
    b = 2 x 16 x 8 x 64 x 2 = 32,768 bytes/token (~32 KiB/token)

Correction note: an early draft of this doc used `d_head = 128`
for Qwen2.5-0.5B-Instruct, giving 24,576 bytes/token. That is wrong.
The real config has `hidden_size = 896` and 14 query heads, so
`d_head = 64`. The 128 value belongs to the 1.5B sibling. The
measured-vs-theory gate uses the config-read value, so this doc
records 12,288.

Sanity check at full context: 32,768 tokens x 12,288 bytes =
402,653,184 bytes = 384 MiB of KV for Qwen2.5-0.5B at 32k context.
That fits comfortably on a 24 GB card, which is why the 0.5B model
is the right campaign vehicle.

## 2a. Campaign model selection

The campaign runs on **Qwen/Qwen2.5-0.5B-Instruct** with the HF
revision pinned and recorded in the run log. Raj asked to prefer
the latest Qwen release where it does not slow the sweeps or break
the derivation. The decision for this campaign is the 0.5B model,
for two concrete reasons:

1. Grid integrity on a 24 GB card. Qwen3-0.6B's theory value is
   57,344 bytes/token (2 x 28 layers x 8 KV heads x 64 x 2 bytes).
   The top grid cell (batch 16 x 32,768 tokens) would need about
   30 GiB of KV cache alone, which does not fit an RTX 4090. The
   0.5B model fits the full 30-cell grid with headroom.
2. The scaling law is model-agnostic. What transfers is the method
   (slope of KV bytes on total tokens, knee at F / b), not the
   absolute bytes/token. Re-running the campaign on a newer Qwen
   later is a one-command change (`--model` + `--revision`); the
   gates do not depend on the model choice.

The exact model id and HF commit revision go into every campaign
JSON, CSV header, plot title, and the run log. If a rerun switches
models, it is a new campaign id, never an edit of the old one.

Pinned revision for this campaign:
`7ae557604adf67be50417f59c2c2f167def9a775`, verified against the
live HF config 2026-09-28 (24 layers, 2 KV heads, 14 query heads,
hidden_size 896, head_dim 64, max_position_embeddings 32768,
theory 12,288 bytes/token). Re-verify with
`huggingface.co/api/models/Qwen/Qwen2.5-0.5B-Instruct` before
launch; if `sha` moved, record the new value, do not silently
float.

## 3. What vLLM adds on top of the law

Two vLLM behaviors shape the measurement protocol. Both are handled,
not tuned. Tuning them is a non-goal.

**Block quantization.** PagedAttention allocates KV in blocks of 16
tokens per sequence. A sequence of T tokens holds `ceil(T/16)`
blocks. The waste per sequence is under 16 x b bytes (about 192 KiB
for the 0.5B model at fp16). This is why the grid uses sequence
lengths that are multiples of 16, and why the slope (not any single
point) is the measured quantity: quantization error cancels in the
slope when the seq_len steps are multiples of the block size.

**Pre-allocation.** vLLM sizes its KV pool once at engine init
(`num_gpu_blocks` from the memory profile) and allocates the whole
pool up front. `torch.cuda.memory_allocated()` therefore does not
move with sequence length during a run. A naive before/after delta
protocol on `memory_allocated` would measure a slope near zero and
fail its own gate. The protocol below measures the same law through
vLLM's block accounting instead, and uses `memory_allocated` only
for the fixed overhead `F`, where it is the right tool.

## 4. Measurement protocol

Engine: one vLLM offline `LLM` instance, `enforce_eager=True`, fp16,
`max_model_len=32768`, on one RunPod GPU. Record the vLLM version,
`max_num_batched_tokens`, and `gpu_memory_utilization`.

Per sample:

1. After engine load, record `mem_after_load =
   torch.cuda.memory_allocated()` and `pool_bytes` (sum over the
   `kv_cache` tensors of `numel x element_size`). Fixed non-KV
   overhead: `F = mem_after_load - pool_bytes` (weights, workspace,
   CUDA context).
2. Run generation for the cell's `(seq_len, batch)` with a fixed
   prompt. Decode a fixed 30 tokens per cell (measured wave spacing
   x margin, see §9(k)) so the cache is filled and activations are in
   steady decode, not prefill. Sampling sets `ignore_eos=True`: EOS is disabled, so the
   decode length is exactly the fixed count for every request. EOS
   timing is out of scope for the measured quantity — the study
   measures KV bytes per live token, and when a request emits EOS
   says nothing about how many bytes each live token holds. Without
   this, early EOS on the fixed prompts let finished waves free
   their blocks mid-sample (see §9(k) follow-up).
3. `torch.cuda.synchronize()`. Read used GPU blocks from the
   scheduler's block pool
   (`llm.llm_engine.engine_core.scheduler.kv_cache_manager.block_pool`
   on vLLM 0.11.2's v1 engine; single-GPU offline `LLM` runs the
   EngineCore in-process by default, verified against the 0.11.2
   source): `used = pool.num_gpu_blocks -
   pool.get_num_free_blocks()`.
4. `kv_bytes_used = used_blocks x bytes_per_block`,
   `T_total =` cached tokens (prompt + decoded) summed over the
   batch. Record one point `(T_total, kv_bytes_used)`.
5. Clear the cache between samples so samples are independent
   (`torch.cuda.empty_cache()` plus synchronize).

Per cell (one seq_len x batch): n >= 5 samples. Report p50 and p95
of `kv_bytes_used` and of per-sample bytes/token. The regression
uses the p50 point per cell; the median resists allocator noise.
This is the repo's percentile discipline applied to memory.

Grid: seq_len in {1024, 2048, 4096, 8192, 16384, 32768}, batch in
{1, 2, 4, 8, 16}. 30 cells.

Analysis:

- Linear regression of `kv_bytes_used` on `T_total` across the 30
  cell p50 points. Slope = measured `b`. The slope cancels the fixed
  overhead by construction.
- Intercept must sit within block-quantization tolerance of zero.
  A nonzero intercept means per-request overhead the theory does not
  predict. Report it either way; it is evidence, not a bug to hide.
- R² >= 0.98 required. The law is exactly linear, so a lower R²
  means a protocol bug, not a noisy world.
- Measured `b` must land within ±15% of config-derived `b_theory`
  (±5% if seq_len is restricted to multiples of 256).
- Per-cell incremental bytes/token: `(kv_{i+1} - kv_i) /
  (T_{i+1} - T_i)` between adjacent seq_len cells at fixed batch.
  It must agree with the regression slope within the
  block-quantization band. This is the local check on the global
  fit.
- Knee: `T* = F / b`, the smallest total context tokens where KV
  bytes reach 50% of total allocated GPU RAM. Derivation: with
  `total(T) = F + bT`, setting `bT / (F + bT) = 1/2` gives
  `T* = F / b`. Status: PENDING; the measured `F` decides whether
  `T*` falls inside the sweep range.

## 5. Error budget

- **Block quantization (16 tokens).** Bounded as in §3. Grid steps
  are multiples of 16, so it cancels in the slope.
- **Allocator fragmentation.** Measure `memory_allocated`, not
  `memory_reserved`. Synchronize and empty the cache between
  samples. Median across samples.
- **Transient prefill activations.** Prefill logits are
  `batch x seq_len x vocab x bytes_per_element`. For the 0.5B model
  (vocab 151,936, fp16) that is about 0.29 MiB per token, or roughly
  9 GiB transient for a single 32k prefill, larger if logits are
  upcast to fp32. This spike is real and large at long context.
  Measure after decode steps (steady state), never during prefill.
  Record `max_num_batched_tokens`; it bounds the spike.
- **CUDA graphs.** `enforce_eager=True` keeps graph-capture buffers
  out of `F`. Record the flag.
- **Version drift.** vLLM's allocator and metrics change across
  versions. Pin the version and record it in the run log.

## 6. Limits

- Single model per campaign. vLLM serves one model per process.
- Single GPU.
- fp16 weights only.
- Absolute numbers (`b`, `F`, `T*`) do not transfer to other models
  or GPUs. The scaling LAW transfers: `b` from the config, linear
  in total tokens, knee at `F / b`. Per the repo evidence policy,
  every reported number carries its model, vLLM version, and
  hardware context.

## 7. Exact environment

The campaign pod is defined by the launch scripts in
`~/workspace/kv-scaling/`. These values are the spec. If any of
them changes, it is a new campaign id.

- Pod image:
  `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`.
  That tag fixes Python 3.11, CUDA 12.8.1, cuDNN (devel build), and
  Ubuntu 22.04. The tag also encodes PyTorch 2.8.0. Do not trust
  the tag alone: the bootstrap prints the real torch version and
  CUDA availability when the venv is created
  (`import torch; print('torch', torch.__version__, 'cuda:',
  torch.cuda.is_available())`), and that printed line is the ground
  truth for the run log.
- Python venv: created with `python3 -m venv --system-site-packages
  .venv` inside `/workspace/kvwork`. It inherits the image's torch;
  pip does not install a second torch. The bootstrap prepends
  `.venv/bin` to PATH and calls `hash -r` so every later command
  uses the venv python.
- Install: `timeout 2400 .venv/bin/pip install -q "vllm==0.11.2"
  pydantic matplotlib`, with `TMPDIR=/workspace/pip-tmp`. The
  `-q` flag is in the script; keep it. On failure the bootstrap
  writes `{"status":"failed","reason":"pip_install"}` to
  `/workspace/serve/status.json` and stops.
- `transformers` / `huggingface_hub`: NOT pinned in the script.
  They are whatever pip resolves for `vllm==0.11.2` at build time
  (the model pull also uses `huggingface_hub.snapshot_download`
  directly). The runner must record the exact resolved versions
  (a `pip freeze` in `bootstrap.log`) in the run log. A rerun that
  resolves different versions is a new campaign id.
- Model: `Qwen/Qwen2.5-0.5B-Instruct` at pinned revision
  `7ae557604adf67be50417f59c2c2f167def9a775`, pulled with
  `huggingface_hub.snapshot_download` into
  `/workspace/kvwork/model` under `timeout 1200`. The sweep then
  runs against the local path (`--model /workspace/kvwork/model`)
  with `--model-id Qwen/Qwen2.5-0.5B-Instruct` and the same
  `--revision` for the campaign JSON.
- Engine flags for the sweep (from `bootstrap_kv.sh` and the
  implementer's defaults): `enforce_eager`, `max_model_len=32768`,
  `gpu_memory_utilization=0.9` (the CLI default in
  `__main__.py`; the bootstrap does not override it),
  `block_size=16` (`DEFAULT_BLOCK_SIZE` in `measure.py`; the
  bootstrap does not override it), fixed prompt, decode
  `temperature=0.0`, 5 samples per cell (`--n-samples 5`).
  `__main__.py` has no `--seed` flag; the measure path carries a
  seed default of 0. Record the sampler seed actually used in the
  campaign JSON.

## 8. Commands per phase

The scripts live in `~/workspace/kv-scaling/`. The commands below
are copied from them, not paraphrased. Do not touch any infra
script; read it if the command below looks off.

**Phase 0: local build (cost $0).**

```bash
~/workspace/kv-scaling/build_tarball.sh
```

The script does three things and nothing else:

```bash
REPO=~/workspace/github-audit/clones/goodputlab
OUT=~/workspace/kv-scaling/code.tgz
cd "$REPO"
test -f bench/kv_scaling/__main__.py || { echo "MISSING bench/kv_scaling/__main__.py - implementer not done"; exit 1; }
tar czf "$OUT" bench/kv_scaling
```

It refuses without `bench/kv_scaling/__main__.py` present. The
tarball contains `bench/kv_scaling` only; that module imports
nothing else from the repo.

**Phase 1: pod launch.**

```bash
python3 ~/workspace/kv-scaling/launch_pod.py ~/workspace/kv-scaling/code.tgz
```

What the script does: asserts the tarball, `bootstrap_kv.sh`, and
`serve.py` all exist in the same dir; base64-encodes the tarball,
the bootstrap source, and `serve.py`; mints a random serve token
(`secrets.token_urlsafe(24)`); and POSTs this body to the RunPod
REST API:

- `"name": "goodputlab-kv-scaling"`
- `"imageName":
  "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"`
- `"gpuTypeIds": ["NVIDIA GeForce RTX 4090"]`
- `"cloudType": "SECURE"`
- `"supportPublicIp": true`
- `"ports": ["8080/http"]`
- `"volumeInGb": 50`
- `"containerDiskInGb": 50`
- start command: `echo "$BOOTSTRAP_B64" | base64 -d >
  /workspace/bootstrap.sh && chmod +x /workspace/bootstrap.sh
  && exec bash /workspace/bootstrap.sh`
- env: `BOOTSTRAP_B64`, `CODE_GZ_B64`, `SERVE_B64`, `SERVE_TOKEN`

The REST call goes through `rp()` in `rp.py`, which is
curl-based. Python's HTTP clients get Cloudflare error 1010 on
`api.runpod.io` / `rest.runpod.io` from this VM; curl passes.
The API key travels in a curl config fed through file descriptor
3, never in argv, on disk, or in logs. On success the script
prints the pod id, the pod's `costPerHr` (the cost guard: read it
before proceeding), and the GPU type, then writes `pod_id.txt`
and `serve_token.txt` (mode 600) into `~/workspace/kv-scaling/`.
There is no SSH involved. RunPod GraphQL is broken (500 on every
query during testing); use REST only.

**Phase 2: bootstrap on the pod.**

The bootstrap is the pod's start command. No SSH is needed or
provided. It runs these steps in order, logging everything to
`/workspace/serve/bootstrap.log`:

1. `mkdir -p /workspace/serve /workspace/kvwork`
2. `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`
   (GPU check; prints a warning line and continues if empty)
3. `echo "$SERVE_B64" | base64 -d > /workspace/serve.py`;
   `echo "$SERVE_TOKEN" > /workspace/serve/token.txt`
4. Start the results server immediately:
   `SERVE_TOKEN="$(cat /workspace/serve/token.txt)" python3
   /workspace/serve.py &`. This is the token-authenticated static
   file server on port 8080 (`serve.py`), serving
   `/workspace/serve`. RunPod maps it to
   `https://<pod-id>-8080.proxy.runpod.net`. It starts before any
   heavy step so the VM can tail `bootstrap.log` live through the
   same `?token=` URL pattern.
5. `echo "$CODE_GZ_B64" | base64 -d | tar xzf - -C
   /workspace/kvwork`
6. `cd /workspace/kvwork`; `python3 -m venv --system-site-packages
   .venv`; `export PATH="/workspace/kvwork/.venv/bin:$PATH`;
   `hash -r`; print the torch version and CUDA availability
7. `export TMPDIR=/workspace/pip-tmp; mkdir -p "$TMPDIR"`;
   `timeout 2400 .venv/bin/pip install -q "vllm==0.11.2" pydantic
   matplotlib`. Failure writes
   `{"status":"failed","reason":"pip_install"}` to
   `/workspace/serve/status.json` and stops.
8. Print `vllm.__version__` and export it as `VLLM_VERSION`.
9. Pull the model under `timeout 1200`:
   `huggingface_hub.snapshot_download(repo_id="Qwen/Qwen2.5-0.5B-Instruct",
   revision="7ae557604adf67be50417f59c2c2f167def9a775",
   local_dir="/workspace/kvwork/model")`.
   Failure writes `{"status":"failed","reason":"model_pull"}` and
   stops.
10. Rung 1, smoke cell:

```bash
.venv/bin/python -m bench.kv_scaling \
  --model /workspace/kvwork/model \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --max-model-len 32768 --enforce-eager \
  --out /workspace/serve/results --approve-cost \
  --seq-lens 1024 --batches 1 --n-samples 5
```

Failure writes `{"status":"failed","reason":"smoke"}` and stops
before any grid spend.

11. Rung 2, full grid: same sweep command with
    `--seq-lens 1024,2048,4096,8192,16384,32768 --batches
    1,2,4,8,16 --n-samples 5`. Failure writes
    `{"status":"failed","reason":"grid"}` and stops; partial cell
    JSONs are kept as failure exhibits.
12. Package: `cd /workspace/serve && tar czf results.tgz
    results/`; write `{"status":"done"}` to `status.json`.
    The pod then idles (`wait`) until the VM fetches the results
    and terminates it.

The runner watches progress by fetching
`https://<pod-id>-8080.proxy.runpod.net/bootstrap.log?token=<serve-token>`
(the server serves any file under `/workspace/serve`) and polls
`status.json` the same way.

**Phase 3: fetch and terminate.**

```bash
python3 ~/workspace/kv-scaling/fetch_and_terminate.py --wait-minutes 90
```

The `--wait-minutes` default is 90. The script polls
`https://<pod-id>-8080.proxy.runpod.net/status.json?token=<token>`
every 60 seconds until the status is `done` or `failed`, or the
deadline hits. It then downloads `results.tgz` into
`~/workspace/kv-scaling/fetched/`, verifies the tarball (rejects
members with absolute paths or `..`), extracts it there, and
finally calls `DELETE /v1/pods/<pod-id>` through `rp()`. The pod
must not idle: termination is the cost control. Note the fetch
itself uses Python's `urllib` to the proxy URL; if that starts
hitting Cloudflare 1010 (the `*.proxy.runpod.net` ban set covers
this VM's Python TLS too), redo the fetch with curl and the
`?token=` URL instead of patching urllib.

## 9. Pre-flight checklist

Four gates. Each must report GREEN before Phase 1 runs. No launch
on a red check.

(a) Environment matches this TRD. Check: the image tag exists in
the RunPod catalog; `vllm==0.11.2` is installable from PyPI
(check the simple index, do not install it locally); the
`bench/kv_scaling` tarball builds (Phase 0 passes). GREEN means
the pinned versions resolve today, not last month.

(b) RunPod credential valid and RTX 4090 secure capacity visible.
Check: `rp("GET", "/v1/pods")` returns 200 through the curl
helper (the key lives in `~/.config/llm-secrets.env` as
`RUNPOD_API_KEY`; never print it); the pod creation endpoint
lists `NVIDIA GeForce RTX 4090` under the SECURE cloud type.
Read the actual `costPerHr` at launch and recompute the cost
estimate if it is above the repo's $0.34-$0.69/hr band.

(c) HF model plus pinned revision reachable. Check:
`https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B-Instruct`
returns and its config still matches the TRD §2a values (24
layers, 2 KV heads, 14 query heads, hidden_size 896, head_dim 64,
max_position_embeddings 32768, theory 12,288 bytes/token). If the
`sha` moved, record the new revision in the campaign JSON and do
not float silently.

(d) Small-scale dry run completed end to end. Check: Rung 0
passes locally (`python3 -m bench.kv_scaling --dry-run` runs the
fake block-manager pipeline, CPU only, and passes
`check_dry_run_gate`); then the pod-side smoke cell
(Phase 2 step 10, 1x1024, 5 samples) passes the block
consistency gate before the full grid starts. GREEN means the
math recovers a known slope locally AND the real block reader
reads used blocks correctly on the pod.

(e) No torch CUDA runtime calls in the parent before engine creation.
Check: `cuda_available()` answers from `nvidia-smi -L` in a separate
process and never imports torch; the hostile-fake regression test
(`test_cuda_gate_never_touches_torch_cuda_runtime`) passes, proving
no `torch.cuda.*` attribute is touched pre-engine; the bootstrap
exports `PYTORCH_NVML_BASED_CUDA_CHECK=1` so any incidental
`torch.cuda.is_available()` takes torch's non-poisoning NVML path.
GREEN means the parent cannot fork-poison vLLM v1's EngineCore.
Background: campaign kv-scaling-20260928 died at the smoke cell with
"Cannot re-initialize CUDA in forked subprocess" because the old
`cuda_available()` called `torch.cuda.is_available()` (which runs
`cuInit` in the parent) before `LLM(...)` forked EngineCore; the pod
then idled ~96 minutes (~$1.18) before manual termination.

(f) Theory resolution tolerates lossy engine configs. Check: both
`_resolve_theory` regression tests pass — the runtime-dtype test
(fake engine whose config dict lacks `torch_dtype` but whose
`model_config.dtype` is bf16 yields 12,288 B/token with provenance
`engine:...+runtime-dtype`) and the raw-Hub-config test (no engine
config path -> pinned config.json). GREEN means the smoke cell
cannot die on a missing `torch_dtype` again.
Background: campaign kv-scaling-20260928b died at the smoke cell
with `KeyError: 'torch_dtype'` on vLLM 0.11.2's wrapped config;
campaign kv-scaling-20260928c died the same way on the first fix's
transformers `AutoConfig` fallback (transformers' `to_dict()` is
equally lossy). The hardened monitor saved both exhibits and
terminated both pods itself (~$0.17 and ~$0.1, no idle burn).

(g) Engine runs in-process so the block pool is readable. Check: the
bootstrap exports `VLLM_ENABLE_V1_MULTIPROCESSING=0` (and
`_run_vllm_sweep_impl` setdefaults it before `LLM(...)`); both
`_probe_block_pool` regression tests pass — the InprocClient layout
resolves to the pool, and a SyncMPClient-shaped engine fails loudly
instead of reading a wrong pool. GREEN means the smoke cell can
actually read free GPU blocks.
Background: campaign kv-scaling-20260928d died at the smoke cell
with "could not find a readable GPU block pool" because the default
multiprocess mode puts the scheduler across a subprocess boundary;
the monitor saved the exhibit and terminated the pod (~$0.1).

(h) Prompt builder resolves tile-boundary token merges. Check: the
`test_build_exact_length_prompts_survives_tile_boundary_merge`
regression test passes — a fake tokenizer that merges " " + "The"
into one token at tile boundaries (the Qwen2.5-0.5B behavior) still
yields exact-length prompts, because the builder re-encodes the tiled
text to its true tokenization before truncating. Also verified by hand
against the real Qwen2.5-0.5B-Instruct tokenizer at the pinned revision
for seq_len 128, 1024, and 8192. GREEN means the smoke cell cannot die
on the round-trip gate again.
Background: campaign kv-scaling-20260928e died at the smoke cell with
"prompt round trip gave 1023 tokens, expected 1024". Root cause, found
by reproducing against the real tokenizer: naive id tiling puts a tile
boundary between the trailing " " of one tile and the leading "The" of
the next, and re-encoding merges them into the single token " The" —
one token lost per boundary. Fix: tile past the target, decode, then
re-encode to the TRUE tokenization (merges resolved) and truncate that
at a token boundary; the final re-encode still verifies the length and
fails loudly on a genuinely broken tokenizer. The monitor saved the
exhibit and terminated the pod itself (~$0.1, no idle burn).

(i) Single-cell smoke run skips the campaign regression fit. Check:
both `_finish_campaign_artifacts` regression tests pass — one cell
writes no campaign artifacts (the fit is skipped, the smoke exits 0
on the cell JSON alone) and two cells write cells.json/summary.json
with the fitted slope. GREEN means the smoke cannot die fitting a
slope on one point.
Background: campaign kv-scaling-20260928f died at the smoke cell with
"need at least 2 points for a fit, got 1" — the first campaign to get
past cell measurement, so the first to reach the unconditional
`fit_bytes_per_token` at the end of `run_vllm_sweep`. A slope on one
point is undefined; the smoke's gates are the block-consistency checks
and a positive F. The monitor saved the exhibit and terminated the pod
itself (~$0.1, no idle burn).

(j) Prefix caching is off so identical prompts do not share KV blocks.
Check: the `test_llm_kwargs_disable_prefix_caching` regression test
passes — `_llm_kwargs` (the single place that builds the vLLM
`LLM(...)` kwargs) sets `enable_prefix_caching=False`, verified
against the real 0.11.2 source (`EngineArgs.enable_prefix_caching`
in `vllm/engine/arg_utils.py`, forwarded through `LLM.__init__`'s
`**kwargs`; an explicit False is honored). GREEN means batch cells
cannot undercount blocks through cross-request sharing.
Background: campaign kv-scaling-20260928g's smoke passed, but the
grid's first batch-2 cell died at the block gate: "block accounting
reports 1056 token-slots below the 2050 live tokens". Root cause: the
sweep builds identical prompts per batch and vLLM defaults
`enable_prefix_caching=True`, so the second request's prompt was a
prefix-cache hit on the first request's KV blocks — 66 blocks held
2050 live tokens. The measurement model needs every request to hold
its own blocks, so caching must be off; leaving it on would also
silently undercount bytes in the regression, not just trip the gate.
The monitor saved the exhibit and terminated the pod itself (~$0.1,
no idle burn).

(k) The decode reading survives chunked-prefill stagger. Check: 56
tests pass, including `test_run_cell_survives_staggered_prefill_waves`
(a fake engine whose wave 2 prefills one step late — the new code
reads the full batch, the old code trips the gate; verified by
mutation), `test_sample_decode_window_fails_loud_on_finish_before_prefill`,
and `test_prefill_chunks_and_cell_decode_tokens`; the rung-0 dry run
recovers slope 12,284 vs injected 12,288, R²=1.0000. GREEN means
multi-chunk batches read the whole batch at decode.
Background: campaign kv-scaling-20260928h's smoke and batch-2/4/8
cells passed, but the batch-16 cell died: "decode reading (66 blocks)
is smaller than the prefill reading (1040 blocks)". Root cause: with
chunked prefill, 16x1024 prompt tokens split into 2 prefill waves that
finished (freeing their blocks) on different decode steps, while the
decode reading was taken on the all-finished step — a design that only
works when every request finishes on the same step. Fix, three parts:
(1) the decode reading is frozen at the last post-step with every
request still alive (which is what the docstring always promised);
(2) per-cell decode length is max(default, prefill_chunks + 1) so the
first wave cannot finish before the last wave's prefill completes —
such a last all-alive step is then guaranteed to exist; the
regression uses measured live token counts, so the extra decode
tokens only move the x-axis, never the slope; (3) `max_num_batched_tokens`
is pinned to 8192 in `_llm_kwargs` and read back from the engine, since
the chunk math depends on it. Two loud guards: a finish before
prefill completes for the whole batch, and no all-alive post-decode
step, both refuse to sweep instead of measuring a partial batch. The
monitor saved the exhibit and terminated the pod itself (~$0.1, no
idle burn).

Follow-up (campaign kv-scaling-20260928i, attempt 8): fix (k)'s
(chunks + 1) decode rule assumed a wave lives until max_tokens,
but the fixed prompts make the model emit EOS after ~4 tokens —
the waves ended at EOS, not at max_tokens. At s2048 b16 (4
prefill chunks) wave 0 finished on its 4th decode step, before
wave 3's prefill completed, so no post-decode step ever had every
request alive and the "no all-alive step" guard fired ("cell
s2048 b16: 4 prefill chunks, decode_tokens 4 -> 5" in the run
log). The (chunks + 1) rule cannot fix this: it only bounds
max_tokens, while EOS is what ends the waves. Fix: the sampler
now passes `ignore_eos=True` (vLLM `SamplingParams`), so the
decode length is exactly the fixed token count for every request.
This is a methodology improvement, not a compromise: the study
measures KV bytes per live token, EOS timing is irrelevant to
that quantity, and this TRD already specified a small fixed token
count — ignore_eos makes it exactly fixed. Wave analysis: with
EOS disabled every request survives to max_tokens, so the
last-all-alive freeze always lands on a full-batch step; the
(chunks + 1) rule is kept as defense in depth. Red-green tests: a
fake driver with EOS-after-N simulation plus 2-wave and 4-wave
chunked-prefill stagger trips the loud guard with
ignore_eos=False and reads the full batch with ignore_eos=True
(the guard trips exactly when eos_after <= waves, i.e. an early
wave finishes before a late wave prefills). The monitor saved the
exhibit and terminated the pod itself (no idle burn).

Follow-up (campaign kv-scaling-20260929k, attempt 10): with EOS
disabled the waves survive to max_tokens, but the (chunks + 1)
decode rule still died at s2048 b16 ("cell s2048 b16: 4 prefill
chunks, decode_tokens 4 -> 5" in the run log, then "no
post-decode step had every request still alive"). Root cause,
measured from the exhibit: the 4 chunked-prefill waves stagger so
widely that wave 1's requests finished all 5 decode tokens before
wave 4's requests each generated one token — the wave spacing
exceeds 5 engine steps, while (chunks + 1) assumed one step per
wave. The engine interleaves later prefill chunks with earlier
waves' decodes inside the per-step token budget, so chunks do not
complete one per step. Fix: the per-cell decode window is sized
from the MEASURED spacing (5 steps) with a 6x margin = 30 tokens,
independent of the chunk count (`EXTENDED_DECODE_TOKENS` in
measure.py). This is principled, not a fudge: decode length is
arbitrary to the measured quantity (KV bytes per live token) — a
longer window only widens the all-alive overlap, it cannot bias
the bytes/token reading; the regression uses measured live token
counts, so the extra tokens only move the x-axis, never the slope.
Secondary gap found while diagnosing: the prefill-timing guard
("a request finished before prefill completed") ran AFTER the
prefill assignment, so a finished wave could sneak a
partial-batch prefill past it and the failure surfaced later as
the misleading decode-freeze message. The guard now runs before
the assignment (one-line reorder). Red-green tests:
`_WideWaveFakeDriver` with 6-step spacing (wider than the measured
>5); the old window leaves no all-alive step and the
prefill-timing guard fires its loud error; `run_cell` reads the
full batch with the 30-token window. Suite: 121 passed in the
project venv.

Attempt-11 outcome (campaign `kv-scaling-20260929l`, 2026-09-29):
24 of 30 grid cells green (s1024/s2048/s4096/s8192 b1-b16, s16384
b1-b8), then the prefill-timing guard refused at s16384 b16 (32
chunks) — firing its loud, correct error, exactly as the
guard-reorder intended. Lesson measured from the run: the 30-token
window held through 16-chunk stagger (s8192 b16, s16384 b8 green)
and failed at 32 chunks — wave spacing scales with chunk count,
a flat window cannot. The method as built reads every cell up to
16 prefill chunks. The 24 green cells all sit within 2% of the
12,288 B/token config-derived theory; same-token-count
factorizations agree to <2% (converging with scale). Per the
hard-stop rule no attempt 12 was launched; the 24-cell bounded
report is the fallback artifact.

## 10. Failure modes with fixes

- **HF download stalls or rate-limits.** `snapshot_download`
  resumes partial files. Retry the pull step with backoff; do not
  restart the pod just for a slow pull unless the `timeout 1200`
  already fired (then it is a model-pull failure, pod stops, new
  campaign).
- **Pod preemption.** Secure cloud pods are rarely preempted, but
  if it happens the campaign is over: cells already fetched are
  exhibits, nothing resumes on a new pod. Relaunch is a new
  campaign id. Never reuse a campaign id.
- **OOM at the top grid cell** (batch 16 x 32,768 tokens).
  Expected physics, not a bug: 16 x 32768 x 12,288 bytes is about
  6 GiB of KV plus model, activations, and the 0.9-utilization
  pool. If the cell OOMs, record it as a failure exhibit with
  `reconcile_passes: false`, exclude it from the regression and
  from all claims. Do not shrink the grid silently and do not
  lower `max_model_len` to make it pass; a smaller grid is a
  smaller claim, recorded openly.
- **Smoke gate fails.** Abort the sweep. Do not launch the full
  grid. A failing 1x1024 cell means the block reader is wrong
  (per `_check_block_consistency` in `measure.py`: used blocks
  times block size must sit in
  `[live_tokens, live_tokens + batch_size * block_size + 64]`
  once every request is scheduled). Fix the reader, re-run
  Rung 0, then relaunch. Never re-run the grid hoping the
  failure was transient.
- **Block-pool API mismatch on the pod.** The reader path
  (`llm.llm_engine.engine_core.scheduler.kv_cache_manager.block_pool`
  on vLLM 0.11.2's v1 engine) was verified against the 0.11.2
  source. If the pod's installed vLLM reports a different pool
  layout, the sweep fails loudly: print the engine's actual API
  surface (dir() of the scheduler and kv_cache_manager) into the
  run log, stop, and do not guess attribute names.
- **Python TLS Cloudflare 1010 on api.runpod.io from the control
  VM.** Known on this box (AGENTS.md). Use curl via `rp.py` for
  every RunPod REST call, never python HTTP. The API key travels
  in a curl config through file descriptor 3, never in argv, on
  disk, or in logs.
- **RunPod GraphQL 500s.** Known during testing: every GraphQL
  query returned 500, including `myself`. Use the REST API only.
  If REST also returns 500, stop and write a run log; do not
  retry in a loop (repeated truncation is a stop condition, per
  the cloud post-mortem in AGENTS.md).
- **/tmp full on the control VM.** The VM's /tmp is small.
  `build_tarball.sh` writes `code.tgz` next to the scripts, and
  the pod sets `TMPDIR=/workspace/pip-tmp`. If a local step
  needs temp space, export `TMPDIR=~/.tmp-pip` (or another
  home-dir path) first.
- **CUDA fork-poisoning before engine creation (killed campaign
  kv-scaling-20260928's smoke cell).** Root cause, reproduced from the
  traceback plus the torch/vLLM 0.11.2 sources: the old `cuda_available()`
  called `torch.cuda.is_available()` in the parent, which runs `cuInit`
  (torch's own docstring: `is_available()` poisons fork unless
  `PYTORCH_NVML_BASED_CUDA_CHECK=1`). vLLM v1 then forks EngineCore, and
  the child's first CUDA touch (`get_flash_attn_version()` at
  `flash_attn` import time → `get_device_capability` → `_lazy_init`)
  raises "Cannot re-initialize CUDA in forked subprocess". Fix: the gate
  now answers from `nvidia-smi -L` (separate process, cannot poison),
  the bootstrap exports `PYTORCH_NVML_BASED_CUDA_CHECK=1`, and the
  hostile-fake regression test guards the invariant. Second-order failure
  from the same incident: `fetch_and_terminate.py` had died silently, so
  the dead pod idled ~96 minutes (~$1.18) before manual termination.
  The monitor now runs under `setsid`/`nohup` and its aliveness is
  verified twice after launch; any future silent death is a launch
  blocker, not a retry.
- **Engine config missing torch_dtype (killed campaign
  kv-scaling-20260928b's smoke cell, after the fork fix worked).**
  `_resolve_theory` trusted the first engine config path
  (`model_config.hf_text_config.to_dict()`), but vLLM 0.11.2's wrapped
  HF config drops `torch_dtype` even though the raw config.json carries
  it (`"torch_dtype": "bfloat16"` verified at the pinned revision).
  `config_from_hf_config` raised a bare `KeyError: 'torch_dtype'`.
  Fix: `_resolve_theory` now catches `KeyError` per engine path and
  falls through to the pinned Hub config (`transformers:AutoConfig`),
  with the provenance recorded in `theory_source`. The strict KeyError
  behavior of `config_from_hf_config` itself is unchanged (covered by
  existing tests); the new regression test drives `_resolve_theory`
  with a fake engine lacking `torch_dtype` and asserts the Hub
  fallback yields 12,288 B/token. Lesson: vLLM's wrapped configs are
  lossy views of config.json; never assume a key survives the wrap.
- **transformers' to_dict() is equally lossy (killed campaign
  kv-scaling-20260928c's smoke cell).** The first fix for the
  torch_dtype KeyError fell through to
  `AutoConfig.from_pretrained(...).to_dict()` — but transformers'
  own `to_dict()` drops `torch_dtype` on the versions vLLM 0.11.2
  pulls in, so the fallback raised the same bare KeyError. Real fix:
  `_resolve_theory` now takes the element width from the engine's
  authoritative runtime dtype (`model_config.dtype`, which vLLM
  resolves before the model loads; provenance `+runtime-dtype`), and
  the last-resort fallback reads the RAW config.json at the pinned
  revision over plain HTTPS (no transformers). The transformers
  fallback is gone. Lesson: never derive anything from a
  `to_dict()` you have not seen with your own eyes on the exact
  dependency versions the pod installs.
- **Block pool unreachable: SyncMPClient has no scheduler (killed
  campaign kv-scaling-20260928d's smoke cell).** The probe path
  `engine_core.scheduler.kv_cache_manager.block_pool` was "verified
  against the 0.11.2 source" — but the source reading missed that
  `LLMEngine.engine_core` is an `EngineCoreClient`, not the
  `EngineCore`: with vLLM's default `VLLM_ENABLE_V1_MULTIPROCESSING=1`
  it is a `SyncMPClient` whose scheduler lives in the EngineCore
  subprocess, unreachable from the parent by attribute access. Fix:
  the bootstrap exports `VLLM_ENABLE_V1_MULTIPROCESSING=0` (and
  `_run_vllm_sweep_impl` setdefaults it), so the engine uses
  `InprocClient` — the real `EngineCore` in-process, with the
  documented layout
  `engine_core.engine_core.scheduler.kv_cache_manager.block_pool`,
  which the probe now tries first. Verified against the actual
  0.11.2 sources: `EngineCore.scheduler` (v1/engine/core.py),
  `Scheduler.kv_cache_manager` (v1/core/sched/scheduler.py),
  `KVCacheManager.block_pool` and `BlockPool.num_gpu_blocks` /
  `get_num_free_blocks()` (v1/core/kv_cache_manager.py,
  v1/core/block_pool.py). Side benefit: no EngineCore fork at all,
  and the parent's `torch.cuda.memory_allocated()` now sees vLLM's
  allocations, which is what the F measurement assumes. Lesson:
  "verified against the source" means tracing the exact runtime
  object graph (`LLMEngine.__init__` assigns a CLIENT), not just
  confirming the target class has the attribute.
- **Tile-boundary token merge in prompt construction (killed campaign
  kv-scaling-20260928e's smoke cell).** `build_exact_length_prompts`
  tiled token ids, truncated to `seq_len`, decoded, and re-encoded to
  verify — and the verify failed: 1023 vs 1024. Root cause, reproduced
  against the real Qwen2.5-0.5B-Instruct tokenizer at the pinned
  revision: a tile boundary falls between the trailing " " of one tile
  and the leading "The" of the next, and the re-encode merges them into
  the single token " The". One token lost per tile boundary. Fix: tile
  past the target, decode, re-encode to the tiled text's TRUE
  tokenization (boundary merges resolved), then truncate THAT at a token
  boundary; the final re-encode still verifies the length and raises on
  a genuinely broken tokenizer. Verified by hand against the real
  tokenizer for seq_len 128, 1024, 8192, plus a regression test with a
  fake tokenizer that exhibits the same merge. Lesson: decode/encode
  round trips are only stable within one encoding of one text; the
  moment ids are concatenated from two encodings, re-encode before you
  trust the count.
- **Regression fit on a single smoke cell (killed campaign
  kv-scaling-20260928f's smoke cell).** `run_vllm_sweep` ended with an
  unconditional `fit_bytes_per_token(cells...)`, and the smoke grid is
  one cell — "need at least 2 points for a fit, got 1". This was the
  first campaign to get past cell measurement, so the first to reach
  that line; every earlier campaign died before it. A slope on one
  point is mathematically undefined, and the smoke's real gates are the
  block-consistency checks plus a positive F. Fix: the tail is now
  `_finish_campaign_artifacts`, which skips the fit, the theory
  validation, the knee, and the campaign/summary JSON for single-cell
  runs (the cell JSON is already on disk; the grid resumes it). Two
  regression tests pin both branches. Lesson: the smoke path must be
  able to pass with exactly one cell — "the regression code runs on the
  single cell without crashing" (TRD §11) means the code handles one
  cell, not that one cell yields a slope.
- **Prefix caching shares KV blocks across identical batch prompts
  (killed campaign kv-scaling-20260928g's grid).** The sweep builds
  the same prompt string for every request in a batch
  (`prompts=[prompt] * batch_size`), and vLLM 0.11.2 defaults
  `enable_prefix_caching=True`. The grid's first batch-2 cell died at
  the block gate: "block accounting reports 1056 token-slots below
  the 2050 live tokens" — the second request's prompt was a
  prefix-cache hit, so 66 blocks held 2050 live tokens. The smoke
  (batch 1) could never see this. Beyond the gate, sharing would
  silently undercount KV bytes in the regression, so this is a
  measurement-validity fix, not just a gate fix. Fix: `_llm_kwargs`
  sets `enable_prefix_caching=False`, verified against the real
  0.11.2 source (`EngineArgs.enable_prefix_caching` in
  `vllm/engine/arg_utils.py`, forwarded through `LLM.__init__`'s
  `**kwargs`; explicit False is honored, only None takes the
  default). Lesson: a block-accounting gate assumes one token-slot
  per live token — any engine feature that shares KV (prefix
  caching, and in future KV-sharing across requests) breaks the
  assumption, so the sweep must pin the engine config that keeps it
  true.
- **Chunked prefill staggers batch waves across finish steps (killed
  campaign kv-scaling-20260928h's grid at batch 16).** 16x1024 prompt
  tokens exceed `max_num_batched_tokens=8192`, so prefill ran in 2
  chunks; wave 1 finished (freeing its blocks) a decode step before
  wave 2. The decode reading was taken on the all-finished step, which
  only represents the full batch when every request finishes on the
  same step — it read 66 blocks against a 1040-block prefill reading
  and the gate fired. The smoke and batch-2/4/8 cells could never see
  this (single-chunk prefills). Fix: (1) freeze the decode reading at
  the last post-step with every request still alive — the docstring's
  original promise; (2) per-cell decode length
  `max(default_decode_tokens, prefill_chunks + 1)` so the first wave
  cannot finish before the last wave's prefill completes, guaranteeing
  such a step exists (the regression uses measured live counts, so the
  extra tokens move only the x-axis); (3) pin `max_num_batched_tokens`
  in `_llm_kwargs` and verify it from the engine, since the chunk math
  depends on it. Two loud guards — a finish before whole-batch
  prefill completes, and no all-alive post-decode step — refuse to
  sweep rather than measure a partial batch. Lesson: any reading that
  compares two moments in a sample must name the live set each moment
  requires; "the decode state" is meaningless once requests have
  different lifetimes.
- **Early EOS ends prefill waves before the last wave prefills
  (killed campaign kv-scaling-20260928i's grid at s2048 b16).**
  Fix (k)'s (chunks + 1) decode rule assumed waves live until
  max_tokens, but the fixed prompts make Qwen2.5-0.5B-Instruct emit
  EOS after ~4 tokens: the waves ended at EOS, not at max_tokens.
  At s2048 b16 (4 prefill chunks) wave 0 finished on its 4th decode
  step, before wave 3's prefill completed — no post-decode step ever
  had every request alive, and the "no all-alive step" guard fired
  instead of measuring a partial batch (the prefill reading in the
  exhibit shows 24,609 live tokens vs 32,768 submitted: wave 0's
  blocks already freed). The (chunks + 1) rule cannot fix this; it
  only bounds max_tokens. Fix: the sampler passes `ignore_eos=True`
  (`SamplingParams`), so decode length is exactly the fixed token
  count for every request. Methodology improvement, not a
  compromise: the study measures KV bytes per live token, EOS timing
  is irrelevant to that quantity, and the TRD already specified a
  small fixed token count. Lesson: "the decode length" is not one
  thing — max_tokens bounds the plan, EOS ends the reality; pin both
  or the staggered-wave analysis is about the wrong lifetime.
- **Cost overrun.** The only cost risk that matters is an idle
  pod, not the cells. Expected spend is under $1 (about 44
  minutes at $0.44-$0.69/hr, per EXECUTION.md). If anything goes
  off-script, terminate the pod immediately
  (`rp("DELETE", "/v1/pods/<pod_id>")`) and sort it out later.
  Hard threshold: ask Raj before any single campaign crosses
  $40 total, including relaunches.

## 11. Verification criteria per phase

Phase 0: the tarball exists at `~/workspace/kv-scaling/code.tgz`,
`tar tzf` shows `bench/kv_scaling/` with `__main__.py`, and the
full repo test suite for `bench/kv_scaling` is green before the
build (the build refuses without `__main__.py`, but a present
file is not a passing suite).

Phase 2 smoke (1x1024 cell): the cell JSON lands in
`/workspace/serve/results/` with p50/p95; the block consistency
gate passes (used blocks times block size within block-rounding
of live tokens, per `measure.py`'s band); F (fixed overhead from
`memory_allocated - pool_bytes`) is positive; the regression
code runs on the single cell without crashing. Gate fails mean
no grid.

Phase 2 grid: 30 cell JSONs attempted; each carries n >= 5 valid
samples with p50/p95; regression across cell p50 points has
R² >= 0.98; slope within ±15% of 12,288 bytes/token (the
config-derived theory); intercept within block-quantization
tolerance of zero; per-cell incremental bytes/token between
adjacent seq_len cells at fixed batch agrees with the slope
within the block-quantization band. Cells that fail any gate are
kept as failure exhibits with `reconcile_passes: false` and are
excluded from the claim.

Phase 3: `~/workspace/kv-scaling/fetched/results.tgz` downloads
and extracts cleanly; it contains the campaign JSON, the 30 cell
JSONs (or the failure exhibits for failed cells), `summary.json`,
and `preflight.json`; the pod is DELETEd (verify with a GET on
the pod id); the run log records pod id, GPU type, hourly rate,
vLLM version, model plus revision, `max_model_len`, cells
attempted/reconciled/failed, wall minutes, and estimated and
actual cost.

## 12. Rollback plan

There is no rollback to a previous state because no shared state
is ever mutated. Every run is a new campaign id, and all data is
additive under `bench/results/kv_scaling/` (one directory per
campaign id). A failed run is terminated and its partial data is
kept as failure exhibits, never edited, never merged into a later
campaign. Re-running means a fresh pod, a fresh tarball build,
and a fresh campaign id. The old pod is deleted, not reused.
If a run must be abandoned mid-grid (preemption, bad smoke,
cost), the runner writes the run log up to the stop point,
terminates the pod, and starts over. Nothing is patched in place.
