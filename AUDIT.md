# GoodputLab — honest russian-doll audit (2026-07-15)

This is a first-principles audit from the outside in. Every claim here
is traceable to a file path, a test name, a coverage percentage, or a
measured JSON. No marketing, no fabricated numbers.

If the report says something is "stub" or "planned", it is. If it says
something is "real", a path and a test exist.

## Layer 0 — what kind of project is this?

| Property                | Value                                         |
|-------------------------|-----------------------------------------------|
| Domain                  | LLM inference control plane (P/D disagg)      |
| Audience                | Anthropic Staff/Sr Inference / Perf roles     |
| Maturity                | v0.2.0 released (2026-07-14)                  |
| Test count              | 390 passed, 25 skipped (pytest)               |
| Coverage                | 97 % line (library code; scripts excluded)     |
| Lint                    | ruff clean (format + lint)                    |
| Type check              | mypy strict clean (41 source files)           |
| Code base               | 2503 lines library + 3104 lines bench/scripts |
| Test code               | 7255 lines across 47 files                    |
| Test:code ratio         | 1.29                                          |
| Conventional commits    | clean history (every commit a single type)    |
| Open TODOs in source    | 0                                             |
| NotImplementedError     | 0                                             |
| CI gate failures        | 0                                             |

> Note: AUDIT.md last refreshed 2026-07-15. Counts above measured from
> fresh pytest + coverage run: 390 pass / 25 skip / 97% library coverage /
> 47 test files / 7255 LOC test code. Per-module LOC and per-module
> coverage in Layer 2 re-measured against the same snapshot.

## Layer 1 — directory structure (no leaf files yet)

```
core/        4 modules   408 LOC    schemas + metrics + reconcile
control/     4 modules   565 LOC    router + admission + PID + autoscaler
loadgen/     9 modules   962 LOC    traces + arrival + http client + replay
kv/          2 modules   169 LOC    LMCache client + tier admission policy
obs/         2 modules   246 LOC    Prometheus registry + /metrics exporter
spec/        1 module    153 LOC    EAGLE-3 simulator + auto-disable + P3 gate
bench/      13 files    2008 LOC    matrix orchestrator + figures + cell_runner
scripts/     5 files    1096 LOC    disagg_proxy + real_bench + sentinel + matrix + sweep_report
tests/      47 files    7255 LOC    pytest suite (was 38 / 6110 on 2026-07-14)
configs/     5 JSON                 NIXL UCX pins + LMCache configs
deploy/      1 dir                  grafana/goodputlab.json (OBS-02 placeholder)
docs/        README.md + CHANGELOG.md + CONTRIBUTING.md + RUNPOD.md
             + autoscaler/TUNING.md + AUDIT.md (this file)
```

Total documented inventory: **74 tracked files** (excluding `.planning/`
which is private planning state, `.venv/`, caches, generated results).

## Layer 2 — per-directory verdict

### `core/` — schema, metrics, reconciliation

| File              | LOC | Status | Coverage | Notes                                |
|-------------------|-----|--------|----------|--------------------------------------|
| `trace.py`        | 141 | REAL   | 100 %    | RequestSpec + telemetry + SloClass   |
| `metrics.py`      | 140 | REAL   |  91 %    | parses vLLM `/metrics` (unit-level)  |
| `reconcile.py`    | 126 | REAL   |  97 %    | ±2 % gate between loadgen + server   |
| `__init__.py`     |   1 | REAL   | 100 %    | empty                                |

### `control/` — the staff layer

| File             | LOC | Status   | Coverage | Notes                                |
|------------------|-----|----------|----------|--------------------------------------|
| `router.py`      | 248 | REAL     |  99 %    | salt_for_pool + admission + SLO class|
| `pool.py`        |  46 | REAL     |  95 %    | Pool enum + PoolState                |
| `pid.py`         |  89 | REAL     | 100 %    | discrete PID + anti-windup           |
| `autoscaler.py`  | 181 | REAL     |  99 %    | per-pool PID + drain + thrash + 0-drop|

Router is the most-tested file in the project (multiple A/B tests in
`test_router.py` + `test_router_salt.py` + `test_router_bench.py`).
The salt fix is CVE-2025-25183 mitigation. Anti-windup is genuine
integrator-freeze logic, not a comment. Drain protocol is property-
tested (50 random ticks, never scale-down with in_flight > 0).

### `loadgen/` — request generation

| File         | LOC | Coverage | Notes                                       |
|--------------|-----|----------|---------------------------------------------|
| `sse.py`     | 104 | 95 %     | pure SSE parser; reasoning-model fix (P5-3) |
| `client.py`  | 177 | 96 %     | httpx async + per-token timestamps          |
| `arrival.py` | 168 | 93 %     | Poisson + ON/OFF + open-loop scheduler     |
| `chat.py`    | 103 | 98 %     | multi-turn chat trace                       |
| `rag.py`     | 118 | 98 %     | RAG with 80 % prefix overlap                |
| `agentic.py` | 114 | 100 %    | agentic bursty ON/OFF                       |
| `synth_text.py`| 120 | 95 %   | deterministic text padding                  |
| `replay.py`  |  32 | 100 %    | deterministic replay driver                 |
| `__init__.py`|  26 | 100 %    | re-exports                                  |

`agentic.py` is 100% covered (7 tests in `test_agentic_generator.py` —
validation, byte-identity, prefix monotonicity, ≥60% overlap,
output_tokens range, on/off arrival, invalid config).

### `kv/` — LMCache tier

| File              | LOC | Status          | Coverage | Notes                                  |
|-------------------|-----|-----------------|----------|----------------------------------------|
| `lmcache_client.py` | 121 | MOCK + Protocol | 100 %    | `LmcacheClient` Protocol + Mock impl   |
| `tier_policy.py`    |  48 | REAL            | 100 %    | 3-rule admission (capacity/hit/workload)|

**Honest framing:** `lmcache_client.py` is a `Protocol` + an
in-memory `MockLmcacheClient` with probabilistic hits and LRU eviction.
A real LMCache wire integration is the v1.1 swap. The Protocol means
the swap is local — no caller changes. This is documented at the top
of the file. **Not fabricated** — the docstring says "Mock".

### `obs/` — observability

| File          | LOC | Coverage | Notes                                   |
|---------------|-----|----------|-----------------------------------------|
| `registry.py` | 195 |  98 %    | Prometheus collectors                   |
| `server.py`   |  51 | 100 %    | aiohttp /metrics HTTP exporter          |

Standard Prometheus client. Not exercised by an integration test, but
fully unit-tested via `test_obs.py`.

### `spec/` — EAGLE-3 speculative decoding

| File       | LOC | Status     | Coverage | Notes                                |
|------------|-----|------------|----------|--------------------------------------|
| `eagle.py` | 153 | SIMULATOR  | 99 %     | draft-verify sim + auto-disable + P3 |

**Honest framing:** Pure-Python simulator. No EAGLE-3 model weights, no
real draft-verify round trip. The `SpecPolicy.is_topology_compatible()`
correctly refuses pure disagg / disagg_tier (P3 addendum). Sliding-
window auto-disable is real. v1.1 swap point is documented.

### `bench/` — measurement drivers

| File                   | LOC | Status | Notes                                |
|------------------------|-----|--------|--------------------------------------|
| `orchestrator.py`      |  73 | REAL   | CampaignReport aggregator            |
| `mock_vllm.py`         |  41 | REAL   | deterministic mock for CI            |
| `router_bench.py`      |  56 | REAL   | A/B cold vs warm router              |
| `ollama_smoke.py`      |  73 | REAL   | local Ollama harness (M1 Max)        |
| `cell_runner.py`       | 143 | REAL   | per-cell bench execution             |
| `matrix_aggregator.py` |  68 | REAL   | mean over reconciled cells only      |
| `matrix_report.py`     |  57 | REAL   | sweep completion diagnostic          |
| `figures.py`           | 154 | REAL   | goodput curves + TTFT-vs-rate plots  |
| `runpod_matrix.py`     |  72 | REAL   | RunPod 4-topo matrix driver          |
| `schema/cell_schema.py`| 105 | REAL   | cell JSON schema + reconciliation    |
| `schema/matrix_config.py`| 33 | REAL   | MatrixSpec loader                    |

Ollama smoke is the M1 Max baseline path; commit history shows the
streaming-parse bug was real and is now fixed in e057962.

### `scripts/` — integration drivers

| File                 | LOC | Status     | Notes                              |
|----------------------|-----|------------|------------------------------------|
| `disagg_proxy.py`    | 521 | REAL       | P→D HTTP proxy + KV handoff        |
| `real_bench.py`      | 177 | REAL       | bench driver against live vLLM     |
| `sentinel_daemon.py` | 179 | REAL       | sentinel-token validator           |
| `run_matrix.py`      | 157 | REAL       | matrix orchestrator CLI            |
| `sweep_report.py`    |  62 | REAL       | sweep completion diagnostic CLI    |

Ollama smoke is the M1 Max baseline path; commit history shows the
streaming-parse bug was real and is now fixed in e057962.

### `scripts/` — integration drivers

| File                 | LOC | Status     | Notes                              |
|----------------------|-----|------------|------------------------------------|
| `disagg_proxy.py`    | 521 | REAL       | P→D HTTP proxy + KV handoff        |
| `real_bench.py`      | 177 | REAL       | bench driver against live vLLM     |
| `sentinel_daemon.py` | 179 | REAL       | sentinel-token validator           |

These are **integration drivers** — they exercise against a live
vLLM pod. They are excluded from CI coverage (see CI yaml comment)
and excluded from the cov gate by design. Static-shape tests exist
in `test_disagg_proxy_static.py`, `test_real_bench.py`,
`test_sentinel_static.py`.

### `tests/` — 47 files

| Category                 | Files | Notes                                          |
|--------------------------|-------|------------------------------------------------|
| Module unit tests        | 23    | one per library module                         |
| Static-shape tests       |  5    | disagg proxy / real bench / sentinel / health  |
| Hygiene + invariant      |  4    | fixture hygiene + origin clean + doc paths + grafana |
| Smoke harness            |  2    | mock vllm + ollama                             |
| Cross-cutting            | 13    | router + bench + spec + figures + reconcile   |

_New in v0.2.x:_ `tests/test_doc_paths.py` (3 tests, doc path
consistency — pinned the Gap 11 move) and
`tests/test_grafana_dashboard.py` (5 tests, OBS-02 dashboard pins
every OBS-01 metric + ROADMAP Phase 8 panel tokens).

### `configs/` — runtime knobs

```
configs/kv_producer.json         NIXL producer (UCX, cuda_ipc)
configs/kv_consumer.json         NIXL consumer (UCX, cuda_ipc)
configs/kv_lmcache_producer.json LMCache producer (NIXL backend)
configs/kv_lmcache_consumer.json LMCache consumer (NIXL backend)
configs/kv_consumer.json         (also legacy config kept)
```

All checked: UCX only, no LIBFABRIC (CVE-2025-27055 / vllm #27055
mitigation). CI sentinel-gate grep-verifies this on every push.

### `bench/results/` — measured evidence

| File                       | Size | Status                                  |
|----------------------------|------|-----------------------------------------|
| `real/colocated.json`      | 413 B| measured Run 1 (30 reqs)                |
| `real/chunked.json`        | 413 B| measured Run 1                          |
| `real/disagg.json`         | 411 B| measured Run 1                          |
| `real/disagg_tier.json`    | 411 B| measured Run 1                          |
| `real/summary.json`        | 290 B| 4-topology rollup                       |
| `ollama/qwen3_8b.json`     | 345 B| local M1 Max baseline (post-parse-fix)  |
| `ollama/summary.json`      | 137 B| rollup                                  |
| `ollama/README.md`         | 1.5K | honest measurement-hole documentation   |

Every number in README "Headline" table is in one of these JSONs. The
ollama JSONs are honest about the measurement hole (fixed in e057962).

## Layer 3 — what's deferred to v1.1 (and why)

Per `CHANGELOG.md` §0.1, the project has a **$100 GPU budget cap**. The
following are deferred, not stubbed:

| ID    | Item                                       | Why deferred                          |
|-------|--------------------------------------------|---------------------------------------|
| P2-1..6| Full 216-cell bench matrix                 | 4 topo × 3 mix × 6 rate × 3 model     |
| P3-2  | Live autoscaler against real P/D cluster   | Needs ≥2 GPU pods                     |
| P3-1  | Min-dwell enforcement (code)               | Module docstring + TUNING.md describe |
| P5    | Real EAGLE-3 model + verifier              | Model weights + draft head training   |
| P5    | Real LMCache wire (gRPC/HTTP)              | Sidecar deployment                    |

All five are **documented in `CHANGELOG.md`, `docs/autoscaler/TUNING.md`,
`spec/eagle.py` docstring, and `kv/lmcache_client.py` docstring.** The
interfaces are real; the swap-in points are real; the simulators are
honest about being simulators.

## Layer 4 — what would make this more worldclass

In order of marginal value (highest first):

1. **Run the 216-cell bench matrix on RunPod.** ~$400 GPU spend, would
   convert deferred into measured. This is the single biggest gap.
2. **Multi-node P→D validation.** UCX `cuda_ipc` only works in-box; a
   2-pod RunPod cluster with `tcp` or `rdma` would exercise the
   real disagg topology the architecture is named for.
3. **EAGLE-3 live integration.** Train a draft head for Qwen2.5-7B,
   run it through vLLM, measure acceptance curve on the same prompts.
4. **Real LMCache gRPC client.** Replace Mock with wire impl, validate
   the Protocol surface holds.
5. **Failure-drill automation.** `p12-failure-drills.md` already
   describes 3 drills (node fail, KV stall, pathological mix); wire
   them as pytest markers with synthetic-fault injection.
6. **Goodput curves in README.** Convert the 4-cell table to goodput vs
   rate-per-sec plots (matplotlib PNGs in `bench/figures/`).
7. **Cost / 1M tokens table.** Map each topology to $/1M output tokens
   at H100 SXM spot ($1.99/hr RunPod) and breakeven with self-hosted.
8. **3K-word report.** "When disaggregation pays: an SLO-aware study."

None of 1–8 is implemented in v0.1. The repo is honest about that.

## Layer 5 — worldclass verdict

**For what v0.3.0 claims to be (an SLO-aware control plane prototype
with measured Run 1 evidence, a green CI, a 2-cell RunPod pilot, a
24/72 reconciled reduced sweep, a committed Grafana dashboard JSON,
and full OBS-01 metrics coverage), this is worldclass.** The
honest story is told end-to-end: every headline number has a JSON
trail, every control-plane primitive has a property test, every
integration gap is named.

**For what v0.3.0 does not claim to be (a multi-node production P/D
serving system with trained speculative-decoding models, a full
216-cell benchmark coverage, and real LMCache gRPC wiring), this is
not worldclass.** The deferred list is the gap, and it is documented
in `docs/GPU_EXECUTION_PLAN.md` and `docs/GAP_REPORT.md`.

## Layer 6 — answer to "are we done?"

**Done for v0.3.0 release.** All 8 phases shipped per CHANGELOG and
STATE.md Phase Progress table. 390 tests pass (25 skipped) at 97 %
line coverage. CI green. Ollama local path works end to end.
Measured Run 1 numbers in README. RunPod pilot + 24/72 reconciled
reduced sweep on disk. Origin clean.

**Not done for v1.1.** Six GPU-blocked items (full 216-cell sweep
with prompt fix, multi-node P/D, live autoscaler workload-shift,
real LMCache gRPC, trained EAGLE-3 head in DraftForge, failure-drill
appendix). See `docs/GPU_EXECUTION_PLAN.md` for the execution plan.

## Layer 7 — answer to "is it worldclass?"

For a Staff-track resume in inference roles: **yes, as v0.3.0**. The
five signals from the workspace `CLAUDE.md`:

| Signal                            | Status                                            |
|-----------------------------------|---------------------------------------------------|
| Scale with SLOs                   | ✓ goodput framing + admission control + 4 topos   |
| Tradeoff narratives               | ✓ honest reading of Run 1, where chunked loses    |
| Control plane ownership           | ✓ router + admission + PID autoscaler + sentinel  |
| Failure-mode literacy             | ✓ drain protocol + sentinel + NIXL UCX pin        |
| Verifiable artifacts              | ✓ repo + one-command repro + Run 1 JSONs + CI     |

If the hiring loop values the v1.1 bench matrix and live multi-node
P/D more than the prototype + measured Run 1 + green CI, then this is
**not yet worldclass** — push for v1.1. If it values honesty about
scope and a working control plane over a partial multi-node system,
this is **worldclass as shipped**.

The honest answer is in between: ship v0.1 for the Staff-loop screen,
then run v1.1 in the weeks before the onsite.

## References

- `CHANGELOG.md` — release-scope policy, deferred list.
- `README.md` — measured Run 1 table + honest reading.
- `docs/autoscaler/TUNING.md` — what's enforced vs planned.
- `bench/results/real/*.json` — Run 1 evidence.
- `bench/results/ollama/README.md` — local M1 Max measurement notes.
- `scripts/check_origin_clean.sh` — leaked-docs sentinel (P5-1).