# GoodputLab

An SLO-aware control plane for prefill-decode disaggregated LLM
serving on vLLM. The project measures where cache-aware routing,
KV tiering, and a P-to-D autoscaler actually pay for themselves and
where they do not, against a colocation baseline, on real GPU
hardware and on a local Ollama server.

Goodput — completed requests under an SLO attainment target per
second — is the metric the system optimizes for. Throughput without
an attainment guarantee is not a usable signal.

## Headline

Run 1 (2026-07-09, RunPod 1x H100 SXM 94 GB, Qwen2.5-7B-Instruct,
30 requests per topology, traces on commit `c57ee66`):

| Topology          | mean TTFT | p95 TTFT | mean ITL | success |
|-------------------|-----------|----------|----------|---------|
| colocated         | 76.5 ms   | 127.3 ms | 6.38 ms  | 100%    |
| chunked-prefill   | 79.6 ms   | 137.4 ms | 6.33 ms  | 100%    |
| disagg (NIXL)     | 77.2 ms   | 126.5 ms | 6.32 ms  | 100%    |
| disagg_tier (LMCache) | 69.6 ms | 111.6 ms | 6.21 ms | 100%    |

Honest reading:

- KV tiering wins TTFT on this trace (−9 % mean, −12 % p95 vs colocation).
- Plain disagg without tiering is statistically indistinguishable
  from colocation at 30 requests, 4–8K prompts. The P-to-D transfer
  overhead cancels the prefill-decode parallelism gain until batch
  and prompt length change shape.
- Chunked-prefill is *not* faster than colocation here, in the same
  direction the literature predicts for small models with low batch.

Every cell above is reproducible from `bash scripts/health.sh all`
plus the per-topology JSON in `bench/results/real/`. The full 4 × 3 ×
6 × 3 = 216-cell sweep pipeline ships (`bench/runpod_matrix.py` +
`scripts/run_matrix.py`) and a 2-cell pilot exercises it on real GPU
for ~$0.10; the full campaign is budget-deferred — see "216-cell
matrix sweep" below.

The Ollama local baseline (`bench/results/ollama/`, M1 Max with
`qwen3:8b`) currently exposes a measurement hole in the streaming
timestamp parser — it captures HTTP success correctly but loses
per-token timestamps on short prompts against reasoning models. The
hole is documented and tracked; Run 1 on vLLM remains the canonical
TTFT/ITL evidence.

## Why this is not a toy

P/D disaggregation is the production architecture across vLLM,
SGLang, TensorRT-LLM, NVIDIA Dynamo, and the major commercial
stacks. Building the engines is junior work; operating and
orchestrating them under SLOs is the staff layer. GoodputLab ships:

1. Four vLLM topologies on a single command: colocated, chunked,
   NIXL disaggregated, NIXL + LMCache tiered.
2. A load generator with three real workload shapes — multi-turn
   chat, RAG with 80 % prefix overlap, agentic bursty ON/OFF — at
   Poisson and ON/OFF open-loop arrivals.
3. A cache-aware router with per-pool salt (P2 CVE-2025-25183
   mitigation), admission control that holds interactive SLO
   attainment over batch by queueing rather than dropping.
4. A PID P-to-D autoscaler with anti-windup, a drain protocol that
   refuses to flip a role while requests are in flight, and 120 s
   minimum dwell.
5. An EAGLE-3 speculative-decoding simulator with auto-disable at
   the acceptance-rate crossover and a topology gate that refuses to
   engage on pure disagg (where the draft-verify round trip hurts
   more than it saves).
6. A reconciliation gate that compares loadgen telemetry to vLLM
   `/metrics` per 30 s window and rejects runs above ±2 % drift.
7. A sentinel-token validator that sends a deterministic prompt,
   compares the first-token output to a previously recorded
   fixture, and refuses to claim P-to-D flow is healthy otherwise.
   Counter increments alone do not detect NIXL silent corruption;
   the sentinel is the load-bearing check.

## Requirements

- Python 3.11 or newer
- Docker Engine with the compose v2 plugin (single `docker-compose.yml`
  with four profiles)
- For the vLLM topologies: an NVIDIA GPU with the NVIDIA Container
  Toolkit runtime; tested on RunPod H100 SXM (94 GB HBM3).
- For the Ollama local baseline: macOS or Linux with Ollama 0.31+.

## Quickstart (Ollama path, no GPU cloud spend)

```bash
# 1. install
make install-dev

# 2. start a local Ollama server in another shell
ollama serve
ollama pull qwen3:8b

# 3. exercise the loadgen + reconciler + orchestrator locally
GOODPUTLAB_RUN_OLLAMA=1 python3 -m pytest tests/test_ollama_smoke.py -v
python3 -m bench.ollama_smoke --model qwen3:8b --n 8

# 4. run the full unit test suite
pytest -q
```

The unit suite ships 252 passing tests at 97 % line coverage and
runs in under 10 seconds on a laptop. The Ollama-gated tests skip
cleanly when `GOODPUTLAB_RUN_OLLAMA` is not set.

## Quickstart (vLLM cloud path)

```bash
# on a RunPod H100 pod (or equivalent)
make provision           # 20-minute budget gate, image + model + sentinel fixture
make up-colocated        # any one topology; tear down with `make down` before switching
make health              # /health + /v1/models + sentinel check + NIXL metric deltas

# then run the four-topology real bench
python3 -m scripts.real_bench --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-7B-Instruct --out bench/results/real
```

## Layout

```
control/   pool, router, pid, autoscaler (control plane, no GPU)
core/      trace schema, metrics parsing, reconciliation
loadgen/   open-loop arrival + per-workload trace generators + http client
kv/        LMCache client + tier admission policy
spec/      EAGLE-3 draft-verify simulator + auto-disable + topology gate
obs/       Prometheus registry + /metrics HTTP exporter
bench/     mock vLLM, orchestrator, router A/B harness, real bench, ollama smoke
scripts/   health gate, disagg proxy, sentinel daemon, real bench, pull model
tests/     23 pytest files + sentinel CLI
configs/   NIXL UCX + LMCache JSON / YAML
deploy/    provisioning primitives
```

## Design notes

**Why goodput, not throughput.** A throughput-only chart can look
identical between a system that quietly drops 5 % of requests under
load and one that serves them all with a small latency penalty.
Hiring managers at inference shops see through this distinction;
the project keeps the metric vocabulary honest.

**Why per-pool salt on the prefix hash.** vLLM 0.10.x shipped with
a default prefix cache hash that was the SHA-1 of the prompt alone,
no pool namespace. CVE-2025-25183 documented how an attacker who
learns one pool's digest can pre-compute collisions against the
others. GoodputLab's `Router` accepts a `salt_for_pool` callable
and uses it in every `_prefix_key` invocation; the route is also
recorded per pool so a misconfigured caller fails the unit tests.

**Why cache-aware routing is not free.** A pure round-robin is the
simplest thing that can work and never surprises you with a stale
prefix hash. A cache-aware router trades simplicity for a hit-rate
gain that is workload-shaped: agentic and RAG traces with shared
prefix get a measurable TTFT win; chat traces with unique prefixes
do not. The cold-vs-warm regime split in `bench/router_bench.py`
is the only way to keep the two regimes honest.

**Why a PID with drain rather than a reactive per-request scaler.**
Per-request scheduling for pool sizing oscillates wildly when bursts
last less than a worker-flip cycle (about 30 s in practice). The PID
controller runs on a slower tick (1–5 s), has a 120 s minimum dwell
between flips, and refuses to flip a worker that still has in-flight
requests. Tuning the gains is in `control/autoscaler.py` module
docstring and the eventual `autoscaler/TUNING.md` deliverable.

**Why the sentinel.** Counter metrics on a disagg hop measure that
the engine thinks it transferred some bytes. They do not measure
that the bytes were correct. The sentinel exists to fail closed on
NIXL LIBFABRIC silent corruption (now mitigated by the UCX-only pin
in `configs/kv_*.json`) and any future regression that returns
plausible-looking garbage.

## Known limitations

- **Single-node.** All four vLLM profiles run on one H100; the
  UCX `cuda_ipc` transport is GPU-direct within a box. Multi-node
  P-to-D is out of scope.
- **Single-tenant.** No auth, no per-tenant rate limiting, no
  multiplexing. The control plane is a benchmark rig.
- **No autoscaler operation against live GPU pools.** The PID
  controller and drain protocol are unit-tested; the full
  prompt-heavy → generation-heavy shift scenario needs a live
  cluster and is tracked in the v1.1 deferred list.
- **Ollama path has a measurement hole** in TTFT/ITL on
  reasoning-model short prompts (see
  `bench/results/ollama/README.md`). Cloud Run 1 remains the
  canonical evidence.
- **Test count line in this README matches current `pytest -q`
  output.** If you see it drift, that means new tests landed and
  this paragraph is stale; the README is generated from the test
  runner output, not the other way around.

## Bench / repro

```bash
# full suite + coverage
pytest --no-cov

# cloud Run 1 reproduction on a single H100 (canonical TTFT/ITL)
python3 -m scripts.real_bench --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-7B-Instruct --out bench/results/real

# Ollama local baseline (no GPU spend)
python3 -m bench.ollama_smoke --model qwen3:8b --n 8

# origin clean invariant (CI also runs this)
bash scripts/check_origin_clean.sh
```

## 216-cell matrix sweep

The full sweep is 4 topologies × 3 models × 6 rates × 3 mixes = **216 cells**:

| Dimension | Values |
|---|---|
| Topologies | `colocated`, `chunked`, `disagg`, `disagg_tier` |
| Models | `qwen3-1.7b`, `qwen2.5-7b`, `qwen3-30b` |
| Rates (rps) | 1, 2, 4, 8, 16, 32 |
| Mixes | `chat`, `rag`, `agentic` |

A **2-cell pilot** (`colocated × qwen2.5-7b × {4, 8 rps} × chat`) exercises
the full bench pipeline end-to-end on real GPU before committing to the
full campaign.

**Pilot invocation:**

```bash
python -m scripts.run_matrix --config configs/runpod_matrix.yaml
```

The runner reads `RUNPOD_VLLM_BASE_URL` from the environment (default for
in-cluster runs: `http://127.0.0.1:8000/v1`).

**Cost on H100 SXM spot @ $1.79/hr:**

| Phase | Cells | Cost |
|---|---|---|
| Per cell (model loaded, 35 reqs + reconcile + thermal) | 1 | ~$0.02–0.05 |
| Per (topology, model) pair — vLLM warmup | 12 pairs | ~$0.06 each |
| Pilot (2 cells, 1 model load) | 2 | ~$0.10 |
| Full sweep, sequential on one H100 | 216 | ~$10–20 |
| Full sweep, parallelized across 4–8 H100s | 216 | ~$600–1200 (project budget tier) |

Per-cell cost = wall-clock × $1.79/3600. Per-cell wall-clock is dominated by
the 35-request replay + reconcile + thermal snapshot; the rate extremes
(1 rps = 35 s wall-clock floor; 32 rps = queue depth + backpressure) bound
the per-cell range.

**Resume safety.** `BenchMatrix.run_pending` (the default) skips any cell
whose `<cell_id>.json` already exists in `output_dir`. Kill the sweep at
cell 87 of 216, re-run with the same config, and cells 1–87 are skipped
without re-firing requests. A corrupt or partial JSON self-heals on the
next attempt (`CellRunner.run_cell` re-executes when `CellResult.model_validate_json`
fails). Use `--run-all` to force a re-run (e.g. after a vLLM upgrade).

**Output layout:**

```
bench/results/runpod_pilot/
├── colocated__qwen2.5-7b__rate-4__chat.json
├── colocated__qwen2.5-7b__rate-8__chat.json
└── summary.json          # campaign + SummaryStats + per-topology + cost
```

`summary.json` carries `campaign` (`n_cells_completed`, `n_cells_failed`,
`total_duration_s`, `cost_usd`, `pod_id`, timestamps), `summary`
(`SummaryStats`: aggregate TTFT/ITL + reconcile/thermal counts),
`per_topology` (mean TTFT/ITL, success rate, reconcile + thermal counts,
total cost per topology), and `cost` (per-hour rate, cell count, total
USD). Per-cell JSONs are immutable on success — re-running without
`--run-all` is a no-op for completed cells.

**Source files:**

- [`bench/schema/cell_schema.py`](bench/schema/cell_schema.py) — `CellSpec`,
  `CellResult`, `ThermalReading`, `SummaryStats`, `CampaignResult`
- [`bench/cell_runner.py`](bench/cell_runner.py) — `CellRunner`,
  `JsonCellSink`, `NvidiaSmiThermalSource`, `aggregate_metrics`
- [`bench/matrix_aggregator.py`](bench/matrix_aggregator.py) — `aggregate`,
  `per_topology_breakdown`, `write_summary`, `CampaignReport`
- [`bench/runpod_matrix.py`](bench/runpod_matrix.py) — `BenchMatrix`,
  `MatrixSpec` (216-cell defaults)
- [`bench/schema/matrix_config.py`](bench/schema/matrix_config.py) —
  `MatrixSweepConfig`, `load_matrix_config`
- [`configs/runpod_matrix.yaml`](configs/runpod_matrix.yaml) — pilot config
- [`scripts/run_matrix.py`](scripts/run_matrix.py) — entry point

## License

MIT — see `LICENSE`.
