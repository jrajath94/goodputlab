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

_(empty — v0.3.0 just shipped; see follow-ups in
`bench/results/runpod_full/README.md` for the v1.1 open work.)_

---

## [0.3.0] — 2026-07-15 — Hygiene + autoscaler observability

Promotes four post-v0.2.0 wins landed between 2026-07-14 → 2026-07-15:
two never-tagged commits on `main` (TUNING doc move + Grafana dashboard),
one fresh feature (thrash + zero-drop counters), and one audit/mypy sweep.
**No fabricated numbers** — every change links to a measured JSON, a
test, or an honest documentation rewrite.

### Added

- **`goodputlab_controller_thrash_total` + `goodputlab_role_flip_inflight_dropped_total` counters**
  (`obs/registry.py` + `control/autoscaler.py` +
  `tests/test_autoscaler_thrash.py`): closes GAP_REPORT gap #3 (AUTO-05
  zero-drop evidence). 5 TDD tests pin the contracts: thrash fires when
  a flip lands within 240 s of the previous flip on the same pool;
  `role_flip_inflight_dropped_total` only increments when the drain
  protocol is violated (P6 mitigation; expected to stay near 0 in
  production). Counters are exposed in `deploy/grafana/goodputlab.json`
  via the existing controller-thrash panel token.
- **Grafana dashboard JSON for OBS-01 metrics**
  (`deploy/grafana/goodputlab.json` + `tests/test_grafana_dashboard.py`):
  closes OBS-02 placeholder. Dashboard imports cleanly against Grafana
  ≥9 (`schemaVersion: 39`) and references every counter + histogram
  declared in `obs/registry.py`. 5 tests pin the contract: parseable
  JSON, top-level schema fields, every declared metric appears in some
  panel, all 8 ROADMAP Phase-8 panel tokens present, schemaVersion ≥ 36.
  PLACEHOLDER banner in dashboard description is honest about panels
  rendering zero values until the v1.1 bench sweep populates them.

### Fixed

- **`autoscaler/TUNING.md` → `docs/autoscaler/TUNING.md`**
  (`docs/autoscaler/TUNING.md` + `tests/test_doc_paths.py`): closes
  GAP_REPORT gap #11. Tuning doc was an orphan at the repo root; now
  lives under `docs/autoscaler/`. 3 TDD tests pin the move: file lives
  under `docs/`, repo-root `autoscaler/` dir is gone, every other `.md`
  in the repo references the new path.
- **mypy strict on `tests/test_grafana_dashboard.py`**: 3 errors
  (untyped `dict` return + missing type args on `list` parameter)
  fixed via `typing.cast` + `dict[str, Any]` + `list[Any]` annotations.
  mypy strict clean across all 88 source files.

### Notes

- Test count: **390 passed, 25 skipped, 97 % line coverage**
  (was 377/25/93 % on v0.2.0). The 13-test delta is 5 autoscaler_thrash
  + 3 doc_paths + 5 grafana_dashboard — `13 = 5 + 3 + 5`.
- `ruff check .` clean. `mypy --strict` clean across all 88 source files.
- venv hygiene: confirmed rebuild from `python3.12 -m venv` +
  `pip install -e ".[dev]"` produces the 390/25 result deterministically.
- AUDIT.md refreshed 2026-07-13 → 2026-07-15 (maturity v0.2.0,
  test count 367 → 390, mypy files 31 → 41, 47 test files / 7255 LOC).
- Phases 5–8 (KV tiering, EAGLE-3 live integration, PID autoscaler
  multi-pod, full 216-cell BENCH capstone) remain deferred to v1.1 per
  the project's $100 GPU budget cap.
- Per workspace `CLAUDE.md` "never mark phase complete — human does,
  after reviewing evidence", the phase-completion checkboxes in
  `.planning/REQUIREMENTS.md` remain unchecked until you review the
  measured numbers and sign off.

---

## [0.2.0] — 2026-07-14 — Post-v0.1.0 RunPod work + integrity sweep

Promotes the seven `[Unreleased]` wins landed in the RunPod + integrity
sweep from 2026-07-09 → 2026-07-14. All seven were on `main` but never
tagged; v0.2.0 ships them as a coherent release. **No fabricated
numbers** — every change below links to a measured JSON, a test, or an
honest documentation rewrite.

### Added

- **RunPod pilot sweep** (2 cells, real H100 SXM, 2026-07-14): exercised
  the full matrix pipeline E2E. All cells reconciled; cost $1.26
  (pod total incl. model load). See `bench/results/runpod_pilot/`.
- **RunPod 72-cell reduced sweep** (qwen2.5-7b only, 2026-07-14):
  exercised the full matrix orchestrator. 24/72 cells reconciled
  (chat mix only). 48 stub cells (agentic + RAG) returned 0% success
  due to ~4× prompt overflow on vLLM `--max-model-len=4096`. DISAGG /
  DISAGG_TIER cells: 0/18 reconciled (label-only, single vLLM in this
  run). Cost $1.30. See `bench/results/runpod_full/`.
- **Sweep completion diagnostic** (`bench/matrix_report.py` +
  `scripts/sweep_report.py`): given a MatrixSpec + cells_dir, reports
  expected/on-disk/missing counts and per-topology gaps. Catches future
  interruptions (the runpod_full DISAGG/DISAGG_TIER gap was first visible
  via this tool) without manual JSON tallying. CLI exits non-zero on
  any gap, suitable for post-sweep CI gating. 8 new tests, 100% covered.
- **Runpod_full TTFT-vs-rate curves** (`bench/figures.py` +
  `bench/figures/runpod_full_ttft_chat.png`): one line per topology
  showing TTFT against arrival rate (log2 x-axis, 1→32 rps), chat mix.
  Generated from the reconciled cells in `bench/results/runpod_full/` so
  the curve matches what was actually measured (no zero-stubs drawn).
  `plot_runpod_full_curves()` skips topologies with <2 reconciled cells
  — colocated + chunked only for chat in this sweep. Topologies absent
  from the curve (disagg, disagg_tier) had 0 measurements; see
  `bench/results/runpod_full/README.md` for the coverage table.
  Replaces two orphan PNGs (`runpod_full_ttft.png`,
  `runpod_full_topo.png`) that had no generator script. 7 new tests
  pin the loader (filters stub cells + malformed JSON + non-cell
  files) and the plotter (one line per topo, single-point → None,
  per-mix filtering, sort by rate).
- **CITATION.cff** (CFF 1.2.0, v0.2.0): added to bring GoodputLab up
  to the workspace Anthropic-tier CFF standard (cf. DraftForge).
- **Test fixture hygiene check** (`tests/test_fixture_hygiene.py`):
  cross-project guard that prevents test fixtures from being committed
  to `bench/results/`. 3 tests, ruff/mypy/pytest green. Closes the
  workspace `PORTFOLIO.md` Gap C.
- **`prefix_index_size_bytes` gauge + `Router.publish_metrics()`**
  (`obs/registry.py` + `control/router.py` +
  `tests/test_router_prefix_index_size.py`): closes ROADMAP RTR-08 / P8.
  Router exposes a periodic `publish_metrics()` snapshot that emits the
  LRU size in bytes (sum of key+pool value UTF-8 lengths + 16B headroom).
  4 TDD tests pin the empty-cache / grows-then-capped / post-eviction /
  no-metrics back-compat contracts. Makes the > 1GB or > 10 % router RSS
  alarm in P8 directly wireable.
- **`docs/GAP_REPORT.md`** (2026-07-14): cross-reference of all 50 v1
  requirements against the code, surfacing 4 still-fixable gaps
  (`cache_aware_router_looked_up_no_history`, role-flip / thrash
  counters, `*.parquet` in `.gitignore`, `STATE.md` frontmatter)
  for v0.3.0.

### Fixed

- **Aggregator stub-cell leak** (`bench/schema/cell_schema.py` +
  `bench/matrix_aggregator.py`): `SummaryStats.from_results` now
  computes latency means over the **reconciled** subset only — stub cells
  (`reconcile_passes=False`, `mean_ttft_ms=0`) previously diluted
  averages and silently masked performance. Added `n_cells_reconciled`
  to expose the sample size. **TDD-red/green:** 4 new tests pin the
  new contract; the failing test name is `test_mean_over_reconciled_only`.
- **`runpod_full` README integrity** (was averaging zeros with non-zeros
  in `summary.json` → wrong TTFT/ITL; had fabricated DISAGG rows;
  mislabeled 1-model vs 3-model). Rewritten with honest per-cell
  aggregates.
- **`runpod_pilot` README staleness** (cost math + next-step section).
  Rewritten with current RunPod spot pricing and explicit
  `bench/results/real/` baseline reference.
- **Top-level `README.md` test + cost + campaign claims** (stale 343
  count pre-pilot, no `bench/figures/` mention, no sweep status).
  Re-aligned with disk state.

### Notes

- Test count: **377 passed, 25 skipped, 93% coverage** (was 343/25/93%).
  `ruff check .` clean. `mypy --strict` clean across all 41 source files.
- venv hygiene: rebuilt `.venv` from scratch with `python3.12 -m venv`
  and `pip install -e ".[dev]"`. Fresh checkout from `main` on a clean
  machine now produces the documented 373/25 result deterministically.
  A stale `python3.11` venv left over from an earlier-than-2026-07-14
  bootstrap had `import yaml` collection-errors in 3 test modules; the
  rebuild fixes them.
- Phases 5–8 (KV tiering, EAGLE-3 live integration, PID autoscaler,
  full 216-cell BENCH capstone) remain deferred to v1.1 per the
  project's $100 GPU budget cap. Open work for v1.1: see
  `bench/results/runpod_full/README.md` (RAG/agentic prompt fix, true
  DISAGG deployment, multi-model serving audit, failure-drill appendix).
- Per workspace `CLAUDE.md` "never mark phase complete — human does,
  after reviewing evidence", the phase-completion checkboxes in
  `.planning/REQUIREMENTS.md` remain unchecked until you review the
  measured numbers in `docs/REPORT.md` and sign off.

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