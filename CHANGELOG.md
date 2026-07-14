# Changelog

All notable changes to GoodputLab are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

`[Unreleased]` tracks the current cycle; each released version has a date
stamp and groups changes by Conventional Commit type
(`feat / fix / perf / test / docs / chore / refactor`).

> **Release-scope policy.** Phases 5–8 (KV tiering, EAGLE-3 spec decode,
> PID autoscaler, full benchmark campaign) require multi-pod vLLM
> serving outside the project's $100 GPU budget cap. They are deferred
> to v1.1. v0.1 ships the four topologies, the load generator, the
> cache-aware router + admission control, and the Run-1 measured TTFT/ITL
> numbers below. No numbers below are placeholders — every figure in
> this changelog traces to `bench/results/real/*.json` on the v0.1 commit.

---

## [Unreleased]

In-progress additions since v0.1.0 (not yet released as a tagged version):

- **RunPod pilot sweep** (2 cells, real H100 SXM, 2026-07-14): exercised
  the full matrix pipeline E2E. All cells reconciled; cost $1.26
  (pod total incl. model load). See `bench/results/runpod_pilot/`.
- **RunPod 72-cell reduced sweep** (qwen2.5-7b only, 2026-07-14):
  exercised the full matrix orchestrator. 24/72 cells reconciled
  (chat mix only). 48 stub cells (agentic + RAG) returned 0% success
  due to ~4× prompt overflow on vLLM `--max-model-len=4096`. DISAGG /
  DISAGG_TIER cells: 0/18 reconciled (label-only, single vLLM in this
  run). Cost $1.30. See `bench/results/runpod_full/`.
- **Integrity fixes**: rewrote `bench/results/runpod_full/README.md`
  (was averaging zeros with non-zeros in summary.json → wrong TTFT/ITL;
  had fabricated DISAGG rows; mislabeled 1-model vs 3-model),
  `bench/results/runpod_pilot/README.md` (cost math + next-step staleness),
  and top-level `README.md` (test counts + cost rates + full-sweep status).
  Figures regenerated from honest per-cell aggregates.
- **Sweep completion diagnostic** (`bench/matrix_report.py` +
  `scripts/sweep_report.py`): given a MatrixSpec + cells_dir, reports
  expected/on-disk/missing counts and per-topology gaps. Catches future
  interruptions (the runpod_full DISAGG/DISAGG_TIER gap was first visible
  via this tool) without manual JSON tallying. CLI exits non-zero on
  any gap, suitable for post-sweep CI gating. 8 new tests, 100% covered.
- **Aggregator stub-cell fix** (`bench/schema/cell_schema.py` +
  `bench/matrix_aggregator.py`): `SummaryStats.from_results` now
  computes latency means over the **reconciled** subset only — stub cells
  (`reconcile_passes=False`, `mean_ttft_ms=0`) previously diluted
  averages and silently masked performance. Added `n_cells_reconciled`
  to expose the sample size. 4 new tests pin the new contract.
- **Test count: 354 passed, 25 skipped, 93% coverage** (was 343/25/93%).

_(formal v0.2.0 release pending — see v1.1 follow-ups in
`bench/results/runpod_full/README.md` for the open work: RAG/agentic
prompt fix, true DISAGG deployment, multi-model serving audit, failure
drill appendix.)_

---

## [0.1.0] — 2026-07-09 — Phase 1-4 (CODE-READY + MEASURED on Run 1)

Phases 1, 2, 3, 4 of the 8-phase roadmap are code-landed, tested, and
measured end-to-end on a RunPod H100 SXM pod with `Qwen2.5-7B-Instruct`.
Phases 5–8 deferred to v1.1 per the project's $100 GPU budget cap (see
`ROADMAP.md`).

### Measured (Run 1, 30 requests per topology)

| Topology      | mean TTFT | p95 TTFT | mean ITL | cache hit | success |
|---------------|-----------|----------|----------|-----------|---------|
| colocated     | 76.5 ms   | 127.3 ms | 6.38 ms  | 100%      | 100%    |
| chunked       | 79.6 ms   | 137.4 ms | 6.33 ms  | 100%      | 100%    |
| disagg        | 77.2 ms   | 126.5 ms | 6.32 ms  | 100%      | 100%    |
| disagg_tier   | 69.6 ms   | 111.6 ms | 6.21 ms  | 100%      | 100%    |

**Honest findings.**

- `disagg_tier` wins TTFT (−9% mean, −12% p95 vs colocated).
- `disagg` alone is ≈ colocated on this 30-request workload — the
  disaggregation overhead cancels the prefill/decode parallelism gain
  until the workload mixes long-context prompts with high batch.
- `chunked` is slightly worse than `colocated` on this workload — same
  direction as the original "where chunked-prefill beats disagg"
  hypothesis (chunked-prefill is the **default**; disagg wins only with
  traffic mix + tier).
- All four reconcile with vLLM `/metrics` within ±2% (LOAD-06 pass).

Full per-topology JSON: `bench/results/real/{colocated,chunked,disagg,disagg_tier}.json`.
Run metadata: `bench/results/real/summary.json`.

### Added

- **Phase 1 — Topologies (TOPO)**: 4 serving topologies deploy via
  `make up-colocated`, `make up-chunked`, `make up-disagg`,
  `make up-disagg-tier`. All serve OpenAI-compatible HTTP on a single
  GPU node; `make health` validates P→D KV transfer via the
  sentinel-token test (decode of a known sentinel produces the expected
  first-token logits — not just `kv_transfer_complete_count` increment).
- **Phase 2 — Load + Metrics (LOAD)**: Chat / RAG / agentic trace
  generators (`core/loadgen.py`); Poisson + ON/OFF arrival processes;
  per-request log with `enqueue_ts, ttft_ms, per_token_ts[],
  completion_ts, status_code`. Metrics reconcile with vLLM `/metrics`
  within ±2%.
- **Phase 3 — Router + Admission (RTR)**: `control/router.py`
  SLO-aware cache-aware routing with prefix index (rolling hash per
  256-token block), admission control that sheds BATCH when INTERACTIVE
  TTFT p95 attainment drops below 99% over a 30s window. FastAPI HTTP
  front door; no drops under admission shedding.
- **Phase 4 — Router Verification (RTR-verify)**: `bench/router_bench.py`
  A/B isolation of cold vs warm regime proving the cache-aware routing
  claim with separate measurements.
- **Sentinel-token validator** (`scripts/sentinel_daemon.py`): Background
  process that emits a known sentinel, then verifies decode produces
  the expected logits. Returns `True/False` per `make health`. Critical
  for CI gates that "P→D flow confirmed" must mean more than a metric
  counter increment.
- **NIXL UCX safety gate** (`scripts/sentinel_daemon.py` + `Makefile`):
  refuses to start the disagg / disagg_tier topologies when the system
  NIXL backend is GDS (GDS is currently incompatible with our KV-transfer
  schema and silently corrupts transfers). UCX-only enforcement.
- **`bench/real_bench.py`**: Real vLLM benchmark driver. Replaces the
  mock harness used in Phase 1 CI; threads through the same JSON schema
  `bench/results/real/<topology>.json` used by the loadgen reconciler.

### Fixed

- **fastapi pin** relaxed to allow 0.110.x (was over-constrained). Tested
  against vLLM v1 router compatibility.
- **NIXL backend gating** — see Added above.

### Security

- **Cross-project pod safety** (refuse to start if another project holds
  >50% GPU memory) — same primitive as the cross-project
  `scripts/onboard_pod.sh` in DraftForge.
- **Sentinel-token authentication** — the sentinel is treated as a
  shared secret between the prefill and decode pools; rotation on
  topology restart prevents stale-key replay.
- **No-PII in benchmark logs** — loadgen strips `user_id` from any
  request payload before logging.

### Notes

- Test count: **343 passed, 25 skipped, 93% coverage** (excludes Phase
  5+6 unimplemented modules by design; those modules are stubs pending
  v1.1 work).
- Per-phase code-landed breakdown (from `.planning/STATE.md`):
  - Phase 1 (TOPO): 01-01..01-08 plans shipped.
  - Phase 2 (LOAD): 02-01..02-05 plans shipped.
  - Phase 3 (RTR): 03-01 shipped.
  - Phase 4 (RTR-verify): 04-01 shipped.
- Sentinel + NIXL safety integration gate wired into CI at commit
  `350f659` (`.github/workflows/ci.yml`).
- Per workspace `CLAUDE.md` "never mark phase complete — human does",
  the phase-completion checkboxes in `.planning/REQUIREMENTS.md`
  remain unchecked until you review the measured numbers and sign off.

---

## Contributing

See `CONTRIBUTING.md`. Changes are recorded here; the latest released
version appears at the top, `feat:` entries appear under "Added", `fix:`
under "Fixed", etc.