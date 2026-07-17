# GoodputLab GPU Cost-Optimization Plan

Scope: GoodputLab's expensive work is live vLLM measurement: matrix sweeps,
true P/D disaggregation, multi-node UCX validation, and autoscaler workload
shifts. This plan reduces GPU spend while preserving the quality of the
project's evidence.

## Cost Thesis

GoodputLab should not run a large matrix until a smaller matrix proves that
the topology, prompt lengths, metrics, and reconciler are all healthy. The
right strategy is staged measurement with hard promotion gates.

Quality comes from clean experimental design, paired comparisons, and
reconciled metrics, not from spending on a huge unreconciled sweep.

## Evidence Ladder

### Rung 0: CPU-Only Contract

Run before any GPU spend:

```bash
pytest -q tests/test_runpod_matrix.py tests/test_matrix_aggregator.py \
  tests/test_reconcile.py tests/test_grafana_dashboard.py
```

Acceptance:

- matrix enumeration is correct
- pending-cell resume works
- reconciler rejects drift
- dashboard still references all metrics

### Rung 1: Single-Cell Health Probe

Run one cheap colocated cell before any topology sweep:

- topology: `colocated`
- model: smallest representative model
- rate: one moderate rate
- mix: `chat`
- `n_warmup`: 2 to 3
- `n_measure`: 8 to 10

Acceptance:

- vLLM endpoint healthy
- telemetry captures prompt tokens and output tokens
- reconciliation passes
- result JSON includes pod id, model, topology, and timing metadata

### Rung 2: Paired Minimal Comparison

Run paired cells only:

- colocated vs chunked for chat
- colocated vs disagg for chat
- colocated vs disagg_tier for one prefix-reuse workload

Use the same model, same rate, same prompt set, same seed. This preserves
statistical power with fewer cells.

Acceptance:

- all paired cells reconcile
- no topology is label-only
- disagg cells are backed by actual separate P/D processes where claimed

### Rung 3: Focused Sweep

Expand only dimensions that changed the decision:

- if RAG prompt overflow occurs, fix context length or reduce prompt shape
  before running more cells
- if chat shows no topology separation, do not spend on more chat rates
- if prefix reuse shows a tiering win, spend there first
- if disagg handshake dominates at low load, skip low-load disagg repeats

### Rung 4: Final Matrix

Run the full matrix only after Rungs 1-3 pass. The final matrix is for
publication polish, not debugging.

## Concrete Changes (IMPLEMENTED 2026-07-16)

### 1. Frugal Matrix Configs — DONE

Configs on disk, each header states cell count, expected wall time,
promotion gate, output directory, and whether true disagg is required:

- `configs/runpod_smoke.yaml` — 1 cell, `smoke: true` (gate exempt)
- `configs/runpod_paired_chat.yaml` — 4 cells, colocated vs chunked
- `configs/runpod_paired_disagg.yaml` — 2 cells, only with true P/D
- `configs/runpod_context_repair.yaml` — 2 cells, RAG/agentic repair
- full sweep stays separate: `configs/runpod_matrix_full.yaml` (final only)

### 2. Cost Preflight — DONE

`scripts/run_matrix.py` now prints, before any request: pending cell
count (and how many existing cells are skipped), topology/model/rate/mix
dimensions, warmup+measured requests, estimated wall time, hourly rate,
estimated cost, and output directory (implementation: `bench/preflight.py`).

Non-smoke runs refuse to start without `--approve-cost` or
`APPROVE_GPU_SPEND=yes` (exit code 5). The preflight is also written to
`<output_dir>/preflight.json` as part of the run's evidence trail.

### 3. Resume, Do Not Restart — DONE

`run_pending` is the default; `--run-all` is the explicit rerun path and
prints an overwrite warning. Paid runs stop after the first unreconciled
measured cell (`stop_on_unreconciled`, on by default; `--keep-going`
disables it for dry-run/mock benches only).

Invariants held:

- existing valid cell JSONs are never overwritten by default
- failed/unreconciled cells are marked (`reconcile_passes: false`), not
  silently retried; rerunning them requires an explicit `--run-all`

### 4. Prompt Waste — DONE (root cause measured)

The prompt preflight in `scripts/run_matrix.py` generates the exact
traces the cells would fire (same generators, same seeds) and checks
prompt + output budgets against the config's `max_model_len` before any
spend. Overflow aborts with exit code 6 unless `--allow-overflow`.

Measured locally on 2026-07-16: RAG prompts run ~18.2-18.4K tokens with a
worst-case prompt+output of **18,539 tokens** — this is the exact reason
the reduced sweep's RAG cells returned HTTP 400 under a 16384 context
window. `configs/runpod_context_repair.yaml` therefore sets
`max_model_len: 20480`; launch vLLM with `--max-model-len 20480` after
checking memory headroom. Agentic worst case is 11,710 tokens (fits).

The prompt-length distribution is recorded in `<output_dir>/preflight.json`.

Do not pay for repeated HTTP 400 failures.

### 5. Pair Every Topology Against the Same Baseline

Use paired comparisons:

- same prompt ids
- same arrival process
- same seed
- same model
- same rate

This lets the report claim topology deltas with far fewer samples.

### 6. Separate Pipeline Validation From Scientific Measurement

Pipeline validation can use:

- smaller model
- fewer requests
- fewer rates
- chat-only workload

Scientific claims require:

- production-relevant model
- true topology implementation
- reconciled metrics
- enough repeats to estimate p95/p99 stability

Do not confuse the two in docs.

## Revised GPU Spend Policy

No GPU run should start unless the agent can answer:

- What exact question does this run answer?
- Which prior cheaper rung passed?
- How many pending cells will run?
- What artifact will be created?
- What condition stops the run early?
- What result would justify the next rung?

## Stop Gates

Stop and tear down immediately if:

- first measured cell fails reconciliation
- vLLM emits context-length errors
- true disagg metrics are absent for a disagg-labeled run
- GPU utilization is near zero for measured cells
- output directory already contains enough data to answer the question

## Quality Guardrails

- Never count unreconciled cells in headline metrics.
- Never mix label-only disagg with true disagg in the same table.
- Never use a placeholder dashboard as evidence of live behavior.
- Keep cost numbers in docs tied to actual run summaries, not estimates.

## Relationship to Existing Handoff

This document modifies the execution strategy in `docs/GPU_EXECUTION_PLAN.md`:
the remaining GPU work is still valid, but it must be executed through the
staged ladder above rather than a direct full sweep.

## Current Project State Fit

Current disk state matters:

- `configs/runpod_matrix.yaml` is already a 2-cell pilot.
- `configs/runpod_matrix_full.yaml` and `bench/runpod_matrix.py` support the
  216-cell shape, but current measured artifacts show the full path should not
  be run blindly.
- `bench/results/runpod_full/README.md` says the 72-cell reduced sweep had
  only 24/72 reconciled cells, with RAG/agentic context failures and no
  disagg/disagg_tier cells on disk.
- `bench/results/real/` has useful 1xH100 single-process evidence, but true
  multi-process P/D remains a separate claim.
- `bench/ollama_smoke.py` and `bench/results/ollama/` exist, so the M1 Max can
  do local endpoint and parser smoke work before RunPod.

Therefore the frugal path is:

1. Use M1 Max/Ollama to validate request generation and result plumbing.
2. Use one cheap RunPod cell to validate vLLM + reconciliation.
3. Use paired topology cells to answer specific topology questions.
4. Run any broad sweep only after the paired cells are clean.

## Current RunPod Price Assumptions

As of 2026-07-16, public RunPod pages and GPU price aggregators list roughly:

| GPU | Practical role | Community estimate | Secure estimate |
|---|---|---:|---:|
| RTX 4090 24GB | local-ish cheap vLLM smoke for small models | ~$0.34-$0.69/hr | varies |
| A100 PCIe 40/80GB | frugal vLLM smoke and medium models | ~$1.19/hr | ~$1.39/hr |
| A100 SXM 80GB | safer medium-model runs | ~$1.39/hr | ~$1.49/hr+ |
| H100 PCIe 80GB | preferred final frugal H100 path | ~$1.99/hr | ~$2.39/hr |
| H100 NVL/SXM 80-94GB | final evidence if PCIe unavailable | ~$2.59-$2.69/hr | ~$2.99/hr |

Use these only as planning assumptions. RunPod is marketplace-priced and billed
by small time increments, so the actual pod page is the source of truth before
launch.

## Cost Model

Use this formula before every run:

```text
estimated_cost = hourly_rate * (setup_minutes + run_minutes + teardown_minutes) / 60
```

For GoodputLab, use conservative setup overheads:

| Scenario | Setup | Run | Teardown | H100 PCIe community @ $1.99/hr | H100 secure @ $2.39/hr |
|---|---:|---:|---:|---:|---:|
| M1/Ollama local smoke | 0 | local | 0 | $0 | $0 |
| 1-cell vLLM health probe | 12 min | 3 min | 2 min | ~$0.56 | ~$0.68 |
| 2-cell pilot | 12 min | 6 min | 2 min | ~$0.66 | ~$0.80 |
| 6-cell paired topology probe | 15 min | 18 min | 2 min | ~$1.16 | ~$1.39 |
| 18-cell focused RAG/tier probe | 15 min | 50 min | 2 min | ~$2.22 | ~$2.67 |
| 72-cell reduced sweep | 18 min | 75 min | 3 min | ~$3.18 | ~$3.82 |
| Full 216-cell sweep | 25 min | 210 min | 5 min | ~$7.96 | ~$9.56 |

These are compute-only estimates. They exclude paid persistent storage and any
idle time. The single largest cost risk is not per-cell runtime; it is leaving
pods idle or debugging context failures on rented hardware.

## Use the M1 Max and Ollama First

Local role: validate workload shape, request parsing, local baseline behavior,
and result plumbing.

Run:

```bash
ollama serve
ollama pull qwen3:8b
make install-dev
GOODPUTLAB_RUN_OLLAMA=1 pytest -q tests/test_ollama_smoke.py
python3 -m bench.ollama_smoke --model qwen3:8b --n 8
pytest -q tests/test_runpod_matrix.py tests/test_matrix_aggregator.py \
  tests/test_reconcile.py tests/test_real_bench.py tests/test_client.py
```

Use the local run to answer:

- are prompts generated correctly?
- does streaming parsing return non-empty telemetry?
- do result summaries render?
- does the matrix runner skip existing cells?

Do not use Ollama numbers for vLLM/P-D claims. Use it only to avoid paying
RunPod for obvious request-shape failures.

## RunPod Execution Ladder

### Rung 1: Pod Preflight, No Matrix

Goal: prove the pod is worth keeping alive.

Commands on pod:

```bash
nvidia-smi
python3 --version
docker --version
make compose-config
```

Stop if any fail.

Expected cost: less than 5 minutes, usually `<$0.25` on H100.

### Rung 2: One vLLM Health Cell

Goal: prove vLLM, model, client, and reconciler are connected.

Use `configs/runpod_smoke.yaml` (on disk; `smoke: true`, so it is the one
config exempt from the spend gate). Set `pod_id` to the actual pod before
launching.

Run:

```bash
export RUNPOD_VLLM_BASE_URL=http://127.0.0.1:8000/v1
python -m scripts.run_matrix --config configs/runpod_smoke.yaml
python -m scripts.sweep_report --config configs/runpod_smoke.yaml \
  --cells-dir bench/results/runpod_smoke
```

Promote only if the cell reconciles.

Expected cost: `$0.50-$0.80`.

### Rung 3: Paired Topology Probe

Goal: answer topology questions with minimal cells.

Use `configs/runpod_paired_chat.yaml` (4 cells: colocated vs chunked at
4 and 16 rps, chat only):

```bash
python -m scripts.run_matrix --config configs/runpod_paired_chat.yaml --approve-cost
```

If — and only if — disagg is truly configured with separate P/D
processes, run the 2-cell comparison in
`configs/runpod_paired_disagg.yaml`. A colocated server labeled "disagg"
is label-only data and must never enter a topology table.

Expected cost: `$1-$2` if model is already loaded.

Promote only if:

- every cell reconciles
- disagg-labeled rows have true transfer metrics
- paired deltas are interpretable

### Rung 4: RAG/Agentic Context Repair Probe

Goal: stop paying for prompt overflow.

Before GPU, the prompt preflight runs automatically inside
`scripts/run_matrix.py` — it generates the exact traces the cells would
fire and aborts on predicted overflow. Already measured on the M1
(2026-07-16): RAG worst prompt+output = 18,539 tokens, agentic = 11,710.
So `--max-model-len 16384` can never work for RAG; use 20480.

On GPU, before raising `--max-model-len`, check memory headroom:

```bash
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
```

Then run only 2 cells via `configs/runpod_context_repair.yaml`
(`max_model_len: 20480`, matching the vLLM launch flag):

```bash
python -m scripts.run_matrix --config configs/runpod_context_repair.yaml --approve-cost
```

Expected cost: `<$1` after model load.

Promote only if both cells avoid HTTP 400/context errors and reconcile.

### Rung 5: Focused Matrix

Only now run a focused matrix. Recommended first publication-grade run:

- topologies: colocated, chunked, disagg_tier only if each is real
- model: one model
- rates: 4, 16, 32
- mixes: chat, rag
- warmup: 5
- measure: 30

This is at most 18 cells and should cost low single-digit dollars if executed
without idle time.

### Rung 6: Full Matrix

The 216-cell matrix is a final polish run. It should not be used for
debugging. Run only when:

- smoke passed
- paired topology passed
- RAG/agentic context issue fixed
- disagg implementation is true, not label-only
- output directory is empty or intentionally resumed

Expected optimized cost: `$8-$12` on current H100 community/secure rates if
setup is clean and no idle debugging occurs. Budget `$20` as a safety cap.

## RunPod Pod Selection

Use this decision tree:

1. M1 Max/Ollama for local shape tests: `$0`.
2. RTX 4090 only for very small vLLM sanity if model fits in 24GB.
3. A100 80GB for qwen2.5-7b, qwen3-1.7b, and most smoke/focused runs.
4. H100 PCIe for final H100-class evidence if available near `$1.99/hr`.
5. H100 SXM/NVL only when PCIe is unavailable or memory/perf is required.
6. Avoid 4xH100 unless explicitly proving multi-node/multi-GPU behavior.

Do not rent H200/B200 for this project. They increase cost without answering
a different research question.

## Hard Budget Caps

Recommended caps:

| Phase | Max spend |
|---|---:|
| local/M1 validation | $0 |
| one-cell vLLM smoke | $1 |
| paired topology probe | $3 |
| RAG/agentic repair probe | $2 |
| focused publication run | $6 |
| full matrix final run | $20 |
| multi-node P/D validation | $10-$15 |

If a phase exceeds its cap, stop and write a run log. Do not continue on the
same pod out of momentum.

## Required Run Log

Every paid run must write:

```text
date_utc:
pod_id:
gpu_type:
hourly_rate_usd:
secure_or_community:
model:
max_model_len:
topologies:
rates:
mixes:
n_warmup:
n_measure:
cells_attempted:
cells_reconciled:
cells_failed:
wall_minutes:
estimated_cost_usd:
actual_cost_usd:
reason_for_next_run_or_stop:
```

