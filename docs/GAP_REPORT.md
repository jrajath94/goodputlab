# Gap Report — GoodputLab v1.0 implementation vs ROADMAP

**Date:** 2026-07-14
**Source plans:** `.planning/PROJECT.md`, `.planning/REQUIREMENTS.md` (50 v1 reqs), `.planning/ROADMAP.md` (8 phases), `.planning/STATE.md` (frontmatter).
**Scope:** Cross-reference every requirement prefix against actual code + tests + artifacts.

This report is the deliverable that the user asked for: *"tell me what's missing"*.
Gaps marked **🟢 Fix-this-session** are TDD'd in commit(s) immediately after this report lands.
Gaps marked **🟡 GPU-blocked** need explicit user confirmation before any spend (`Do NOT launch GPU pods`).
Gaps marked **🔵 Deferred** are documented in roadmap v2 as out of scope.

---

## TL;DR

All 8 phases have code; v0.1.0 is shipped and tagged (see `git tag -l`
and `CHANGELOG.md`). The earlier non-GPU gaps called out in this report
are now closed:

- router `no_history` counter: shipped
- `prefix_index_size_bytes` gauge: shipped
- autoscaler thrash / zero-drop counters: shipped
- parquet whitelist: shipped
- STATE.md frontmatter: refreshed
- Grafana dashboard JSON: committed at `deploy/grafana/goodputlab.json`

What remains is mostly hardware-backed validation, not missing core
implementation. The current actionable backlog is:

| # | Gap | Severity | Effort | GPU? |
|---|-----|----------|--------|------|
| 1 | Full matrix completion with RAG + agentic prompt-length fix | 🟡 GPU-blocked | low code, low spend | **Yes** |
| 2 | True disagg validation on separate prefill/decode processes with NIXL metrics | 🟡 GPU-blocked | medium | **Yes** |
| 3 | Multi-node 2-pod cluster (`cuda_ipc` → `tcp/rdma` UCX) | 🟡 GPU-blocked | medium/high | **Yes** |
| 4 | Live autoscaler workload-shift validation | 🟡 GPU-blocked | medium | **Yes** |
| 5 | Real LMCache gRPC wire client | 🔵 Deferred | v1.1 | Yes |
| 6 | Trained EAGLE-3 head (DraftForge scope) | 🔵 Deferred | separate repo | Out of scope |

The execution-level GPU plan lives in `docs/GPU_EXECUTION_PLAN.md`.

---

## Gap 1: `cache_aware_router_looked_up_no_history` counter (RTR cold-cache distinguisher)

**Roamap reference:** ROADMAP Phase 4 success criterion #3:
> "`cache_aware_router_looked_up_no_history` counter distinguishes cold-cache from cache-miss"

**Requirement:** RTR-04 dual-regime reporting (P7 mitigation).

**Current state:** `obs/registry.py` does not declare this counter. `control/router.py` does not record
the distinction. `bench/router_bench.py` runs cold-vs-warm but does not export the per-bucket counter
to the registry.

**Fix (TDD):**
1. Add `no_history` Counter + `inc_no_history` helper to `obs/registry.py`.
2. Wire `control/router.py.route()` so that when `key not in self._prefix_cache`, it calls
   `registry.inc_no_history()` (only when a registry is injected; preserve statelessness for tests).
3. New test in `tests/test_router.py` asserting cold lookup increments `no_history`.
4. Add `test_router_emits_no_history` asserting the wiring (dependency-injected registry).

---

## Gap 2: `prefix_index_size_bytes` gauge (P8/RTR-08 alert)

**Roadmap reference:** ROADMAP Phase 3 success criterion #4:
> "Prefix index hard-capped (TTL 1hr + LRU size cap); `prefix_index_size_bytes` metric exposed;
> alert > 1GB or > 10% router RSS"

**Current state:** `obs/registry.py` lacks the gauge. `control/router.py._prefix_cache` is an
`OrderedDict[str, Pool]` whose size is exposed via `cache_size()` (entry count, not bytes).
There is no periodic gauge update path.

**Fix (TDD):**
1. Add `Gauge("goodputlab_prefix_index_size_bytes", ...)` to registry.
2. Add `update_prefix_index_bytes` helper that records a snapshot.
3. Wire a `_publish_prefix_index_bytes()` method in `Router` that calls the helper (no-op if no
   registry injected), to keep the dependency optional.
4. New test: `test_router_publishes_prefix_index_bytes_grows_then_capped`.

---

## Gap 3: AUTO 0-drop + thrash counters

**Roadmap reference:** AUTO-05 + AUTO-03:
> "Zero dropped in-flight requests during role flips"
> "`flip_count_per_minute` <0.5 sustained; `controller_thrash_detected` alarm fires on 2 flips within 240s"

**Current state:** `obs/registry.py` has `goodputlab_role_flip_total` (event count). It does NOT
have:
- `goodputlab_role_flip_inflight_dropped_total` (AUTO-05 zero-drop evidence)
- `goodputlab_controller_thrash_total` (AUTO-03 thrash alarm)

`control/autoscaler.py` does not consume a `MetricsRegistry`; counters, when emitted, would have
to be plumbed in (dependency injection).

**Fix (TDD):**
1. Two new Counters in `obs/registry.py` + two `inc_*` helpers.
2. Constructor option `registry: MetricsRegistry | None` on `AutoscalerController`.
3. Hook `record_flip()` to call `inc_role_flip(from, to)` + `inc_thrash()` when
   `now - last_flip_ts < 240s` for the same pool; emit `inc_inflight_dropped(n)` only if a
   future `--force-drain-with-inflight=N` test fixture observes an actual drop.
4. Tests in `tests/test_autoscaler.py`:
   - `test_autoscaler_emits_role_flip_counter`
   - `test_autoscaler_emits_thrash_counter_on_two_flips_in_240s`
   - `test_autoscaler_no_drop_when_drain_block_honored`

---

## Gap 4: `.gitignore` blocks parquet (REPRO-03 conflict)

**Roadmap reference:** REPRO-03:
> "All bench results stored as parquet + metadata JSON (HW, seed, version)"

**Current state:** `.gitignore` line: `*.parquet` blocks all parquet. There is no whitelist for
`bench/results/**/*.parquet`. The codebase currently stores results as JSON only.

**Fix:**
1. Replace blanket `*.parquet` with two narrow lines:
   ```
   # REPRO-03: bench results committed as JSON + parquet
   bench/results/**/*.parquet
   ```
   And remove `*.parquet`.
2. Document the choice in AUDIT.md under "REPRO-03 status".
3. Add a stub writer `bench/parquet_export.py` (lazy — only when a real bench run produces data,
   no committed JSON yet).

**Alternative (lighter):** Document that v0.1.0 stores JSON-only; parquet is v1.1. Mark REPRO-03
as `partial` with a forward pointer. This avoids committing a one-line tree worth of dummy files.

---

## Gap 5: STATE.md frontmatter stale

**Current state:** STATE.md line 5:
> `status: scope_revised_phases_1_4_code_landed_phases_5_8_deferred`
> `stopped_at: Phase 4 (RTR-verify) code landed + tested (177 passed, 20 skipped, 95% cov); Phases 5-8 (KV/SPEC/AUTO/BENCH) deferred per $100 GPU budget cap`

But actual disk state (per recent commits and grep):
- Phase 5: `kv/lmcache_client.py`, `kv/tier_policy.py`, `configs/lmcache_*.yaml`, `configs/kv_lmcache_*.json` — shipped
- Phase 6: `spec/eagle.py`, `tests/test_eagle.py` — shipped
- Phase 7: `control/autoscaler.py`, `control/pid.py`, `autoscaler/TUNING.md`, `tests/test_autoscaler.py`, `tests/test_pid.py` — shipped
- Phase 8: `bench/results/real/`, `bench/results/runpod_full/`, `bench/results/runpod_pilot/`, `docs/REPORT.md` (418 lines, 3K words), `bench/figures/*.png`, `cost_per_million_tokens.csv`/.md — shipped

**Fix:** Rewrite STATE.md frontmatter + Phase Progress table to:
- `status: v1.0_shipped_phases_1_to_8_per_v0.1.0_tag`
- `stopped_at: v0.1.0 release (367 passed, 25 skipped, 93% coverage); gaps 1-5 deferred to v1.1`
- Phase Progress table marks all 8 as `Shipped (v0.1.0)` rather than `Code landed`.

---

## Gap 6: Grafana dashboard JSON (OBS-02)

**Roadmap reference:** OBS-02:
> "Grafana dashboard JSON committed: goodput, TTFT p95, ITL p95, queue depth per pool, KV-tier hit rate"

**Current state:** Searched repo for any `.json` matching Grafana schema (`panels`, `templating`,
`title`). Result: only `obs/server.py` (text exposition). Zero dashboard JSONs committed.

**GPU-spend status:** Grafana JSON authoring is local (CPU only). However the observed values would
be tied to a specific run; committing a static JSON without a recording source means viewers see
zeros. The honest move:
- Option A: ship a minimal "minimal-viable" Grafana JSON that imports cleanly and shows the metric
  names but with `0` for all time series (honest: "ready for data, no real run attached").
- Option B: defer to v1.1 with explicit "OBS-02 partial" tag.

Per integrity baseline, Option A is honest *only* if the JSON is documented as `placeholder`,
not "live". Recommend Option B for v0.1.0 honesty, Option A as the v1.1 file once a sweep completes.

**Blocker:** none — could ship a placeholder tonight. Defer to gap-fix backlog (no rush).

---

## Gap 7: Full 216-cell sweep (BENCH-01 partial)

**Roadmap reference:** BENCH-01:
> "`make bench` runs full matrix: 4 topologies × 3 workloads × 6 load levels × 3 seeds = 216 cells"

**Current state:**
- `bench/results/real/`: 4 JSONs (Run 1, H100 SXM Qwen2.5-7B, all 4 topos — 2 cells each)
- `bench/results/runpod_pilot/`: 2 JSONs (rate-4 + rate-8 colocated chat)
- `bench/results/runpod_full/`: 24+54=78 JSONs (qwen2.5-7b + qwen3-1.7b + qwen3-30b × rates × mixes — **but** only `chunked` and `colocated` topologies populated)

Total reconciled cells: ~80 / 216 = **37%** of the matrix.

**Blockers:**
- RAG/agentic workloads overflow vLLM's `--max-model-len` (default 4096); 8-32K prompt exceeds it
- Multi-model sweep (qwen3-30b) requires >80GB VRAM; current pilot pod is H100 80GB and chunks OK
  but at high rates, OOM
- Multi-node P/D (cuda_ipc fails on separate pods) needs topology testing for tcp/rdma UCX

**Cheapest fix:** bump `--max-model-len 16384` and re-pilot RAG/agentic cells ($1.30 quick-fix from
prior session estimate). Brings reconciled cells to ~120/216 = 56%.

**Need user confirmation** per `Do NOT launch GPU pods` policy.

---

## Gap 8: Multi-node P/D cluster

**Roadmap reference:** v2 / MULTI-01.

**Status:** Out of scope for v0.1.0 per ROADMAP. Useful v1.1 work — but expensive (~$30 for
a couple hours of 2× H100 on RunPod).

**Blocker:** user confirmation + spawn flow.

---

## Gap 11: `autoscaler/` orphan directory at repo root

**Current state:** `autoscaler/TUNING.md` (192 lines) is the only file. The directory is not in
`pyproject.toml` wheel `packages` list. Recent commit `e1da20e` removed similar stubs (`router/`,
`analysis/` at root).

**Honest fix:** Either:
- (A) Move `autoscaler/TUNING.md` to `docs/autoscaler/TUNING.md` and remove the orphan dir.
- (B) Remove the empty dir (it has only one .md inside, `.gitkeep` not needed).

**Recommendation:** (B). The PID/Autoscaler modules live under `control/`, named so for layout.
`autoscaler/TUNING.md` belongs under `docs/` (where `RUNPOD.md`, `CHANGELOG.md`, etc. live).

**Note from TUNING.md:** Quick read shows it documents the PID gain rationale for the AUTOSCALER
deliverable (Phase 7). That belongs in `docs/AUTOSCALER.md` for discoverability.

---

## TDD work order (this session)

| Order | Gap | Commit plan |
|-------|-----|-------------|
| 1 | Gap 5 (STATE.md) | `chore(docs): refresh STATE.md to v0.1.0 reality` |
| 2 | Gap 11 (autoscaler/ orphan) | `chore(hygiene): move autoscaler/TUNING.md to docs/` |
| 3 | Gap 1 (no_history counter) | `feat(obs): cache_aware_router_looked_up_no_history counter + Router wiring` |
| 4 | Gap 2 (prefix_index_size_bytes) | `feat(obs): prefix_index_size_bytes gauge + Router snapshot` |
| 5 | Gap 3 (AUTO 0-drop + thrash) | `feat(obs+control): role_flip_inflight_dropped + controller_thrash counters` |
| 6 | Gap 4 (.gitignore parquet) | `chore(repro): whitelist bench/results parquet (REPRO-03)` |
| 7 | Push | `git push origin main` (after all 5 atomic commits land) |

Each commit gated by `ruff check . && mypy . && pytest -q --no-header --no-cov` exiting 0.

---

## Items NOT in scope (this session)

| Item | Why deferred |
|------|--------------|
| Gap 6 (Grafana JSON) | No live data to bind — would ship placeholder, looks dishonest in v0.1.0 |
| Gap 7 (full 216 sweep) | GPU spend required; user-confirm needed |
| Gap 8 (multi-node) | v2 / MULTI-01 |
| Gap 9 (trained EAGLE-3) | Separate project (DraftForge) |
| Gap 10 (real LMCache gRPC) | v1.1 |

---

*Report generated 2026-07-14. Authored against live disk state + `.planning/` plans.
Every "🟢 Fix" gap is followed by a TDD commit in this session.*
