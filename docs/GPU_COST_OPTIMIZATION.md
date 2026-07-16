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

## Concrete Changes for the Next Agent

### 1. Add Frugal Matrix Configs

Create explicit configs:

- `configs/runpod_smoke.yaml`
- `configs/runpod_paired_minimal.yaml`
- `configs/runpod_focused_rag.yaml`
- keep full sweep separate as `configs/runpod_full.yaml`

Each config must state:

- cell count
- expected wall time
- promotion gate
- output directory
- whether true disagg is required

### 2. Add a Cost Preflight

Before a GPU run, the runner should print:

- number of pending cells
- warmup requests
- measured requests
- estimated wall time
- configured hourly rate
- estimated total cost
- output directory

It should require an explicit `--approve-cost` or env var for non-smoke runs.

### 3. Resume, Do Not Restart

Use `BenchMatrix.run_pending` for all paid sweeps. Never use `run_all` on a
paid pod unless intentionally regenerating a corrupt output directory.

Required invariant:

- existing reconciled cell JSONs are never overwritten by default
- failed/unreconciled cells are marked, not silently retried

### 4. Reduce Prompt Waste Without Weakening Evidence

The prior RAG/agentic failures were caused by prompt-length mismatch. Fix by:

- measuring prompt token lengths before GPU run
- filtering or truncating only to the experimental context budget
- recording prompt-length distribution in the run summary
- increasing `--max-model-len` only after memory feasibility is checked

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

