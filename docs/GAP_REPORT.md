# Gap Report — GoodputLab v1.0 implementation vs ROADMAP

**Date:** 2026-07-15 (refresh of 2026-07-14 original)
**Source plans:** `.planning/PROJECT.md`, `.planning/REQUIREMENTS.md` (50 v1 reqs), `.planning/ROADMAP.md` (8 phases), `.planning/STATE.md` (frontmatter).
**Scope:** Cross-reference every requirement prefix against actual code + tests + artifacts.

This report is the deliverable that the user asked for: *"tell me what's missing"*.
Gaps marked **🟢 Fix-this-session** are TDD'd in commit(s) immediately after this report lands.
Gaps marked **🟡 GPU-blocked** need explicit user confirmation before any spend (`Do NOT launch GPU pods`).
Gaps marked **🔵 Deferred** are documented in roadmap v2 as out of scope.
Gaps marked **✅ Closed (v0.3.0)** were fixed in commits between 2026-07-14 and 2026-07-15 and are listed here for traceability.

---

## TL;DR

All 8 phases have code; **v0.3.0** is shipped (390 passed / 25 skipped
/ 97 % line coverage). Every non-GPU gap called out in the 2026-07-14
version of this report is now closed:

- router `no_history` counter: shipped
- `prefix_index_size_bytes` gauge: shipped
- autoscaler thrash / zero-drop counters: shipped
- parquet whitelist: shipped
- STATE.md frontmatter: refreshed
- Grafana dashboard JSON: shipped (placeholder values, honest framing)
- `autoscaler/` orphan directory: moved under `docs/`

What remains is hardware-backed validation, not missing core
implementation. The current actionable GPU-blocked backlog is:

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

**Current state (closed in commit `c5f7cc6` 2026-07-14):** STATE.md was rewritten to reflect v1.0 reality. As of v0.3.0 (2026-07-15), the frontmatter reads:

> `milestone: v0.3.0`
> `stopped_at: v0.3.0 (2026-07-15) shipped: 8 phases code-landed + measured Run 1 evidence + 2-cell pilot + 24/72 reconciled cells in runpod_full + Grafana dashboard JSON placeholder (390 passed / 25 skipped / 97 % coverage)`

The Phase Progress table marks all 8 phases as `Shipped (v0.1.0)`.
Status: **✅ Closed (v0.3.0)**.

> Historical (now-closed) gap description, retained for traceability:
> The 2026-07-14 STATE.md line 5 said:
> > `status: scope_revised_phases_1_4_code_landed_phases_5_8_deferred`
> > `stopped_at: Phase 4 (RTR-verify) code landed + tested (177 passed, 20 skipped, 95% cov); Phases 5-8 (KV/SPEC/AUTO/BENCH) deferred per $100 GPU budget cap`
>
> But actual disk state was:
> - Phase 5: `kv/lmcache_client.py`, `kv/tier_policy.py`, `configs/lmcache_*.yaml`, `configs/kv_lmcache_*.json` — shipped
> - Phase 6: `spec/eagle.py`, `tests/test_eagle.py` — shipped
> - Phase 7: `control/autoscaler.py`, `control/pid.py`, `autoscaler/TUNING.md`, `tests/test_autoscaler.py`, `tests/test_pid.py` — shipped
> - Phase 8: `bench/results/real/`, `bench/results/runpod_full/`, `bench/results/runpod_pilot/`, `docs/REPORT.md` (418 lines, 3K words), `bench/figures/*.png`, `cost_per_million_tokens.csv`/.md — shipped

---

## Gap 6: Grafana dashboard JSON (OBS-02)

**Roadmap reference:** OBS-02:
> "Grafana dashboard JSON committed: goodput, TTFT p95, ITL p95, queue depth per pool, KV-tier hit rate"

**Current state (closed in commit `ade9526` 2026-07-15, refined v0.3.0):** The
dashboard JSON ships at `deploy/grafana/goodputlab.json` as an explicit
**placeholder** — Option A above, with the dashboard's top-level
`description` field naming the gap, per integrity baseline.

Status: **✅ Closed-as-placeholder (v0.3.0)**. The 5 tests in
`tests/test_grafana_dashboard.py` pin: parseable JSON, modern schema
(`schemaVersion=39`), every OBS-01 metric referenced by some panel,
all ROADMAP Phase 8 panel tokens present, and the dashboard
self-describes as a placeholder.

Honest framing: panels render zero values against a fresh Prometheus.
They will populate when the v1.1 bench sweep produces recorded
counters and histograms.

> Historical (now-closed) gap description, retained for traceability:
> Searched repo for any `.json` matching Grafana schema. Result: only
> `obs/server.py` (text exposition). Zero dashboard JSONs committed.
>
> **GPU-spend status:** Grafana JSON authoring is local (CPU only).
> However the observed values would be tied to a specific run;
> committing a static JSON without a recording source means viewers
> see zeros. The honest move:
> - Option A: ship a minimal "minimal-viable" Grafana JSON that imports
>   cleanly and shows the metric names but with `0` for all time
>   series (honest: "ready for data, no real run attached").
> - Option B: defer to v1.1 with explicit "OBS-02 partial" tag.
>
> Per integrity baseline, Option A is honest *only* if the JSON is
> documented as `placeholder`, not "live". Recommend Option B for
> v0.1.0 honesty, Option A as the v1.1 file once a sweep completes.
> **Outcome:** Option A shipped at v0.3.0; the placeholder banner is
> in the dashboard's `description` field; 5 tests pin the contract.

---

## Gap 7: Full 216-cell sweep (BENCH-01 partial)

**Roadmap reference:** BENCH-01:
> "`make bench` runs full matrix: 4 topologies × 3 workloads × 6 load levels × 3 seeds = 216 cells"

**Current state (2026-07-16):**
- `bench/results/real/`: 4 JSONs (Run 1, H100 SXM 80 GB Qwen2.5-7B, all 4 topos — 30 reqs each, all reconciled; the canonical TTFT/ITL evidence cited in `docs/REPORT.md` and `README.md` headline).
- `bench/results/runpod_pilot/`: 2 JSONs (rate-4 + rate-8 colocated chat, all reconciled; pilot cost $0.008 per cell, $1.26 pod total).
- `bench/results/runpod_full/`: 72 JSONs (24 reconciled + 48 unreconciled stub cells; only `chunked` (6/18 reconciled) and `colocated` (18/54 reconciled) topologies populated; `disagg` and `disagg_tier` cells were never generated — the sweep stopped before reaching them, see `bench/results/runpod_full/README.md`).
- `bench/results/runpod_v11/`: **54 JSONs** (44 reconciled, 10 unreconciled; 2026-07-16, H100 SXM 80 GB Qwen2.5-7B, `--max-model-len=16384`, $0.63 spend). 16K context lifts the RAG and agentic overflow from `runpod_full/`. 18 cells labelled `disagg` are served by the same single-vLLM process as `colocated` and `chunked` — see `bench/results/runpod_v11/README.md` §"Honest finding" for why. 0/54 stub cells.

Honest reconciled-cell count: **74 of 216 = ~34 %** (4 Run 1 + 2 pilot
+ 24 reduced-sweep + 44 v1.1 sweep). 50 attempted cells failed to
reconcile (mostly RAG overflow at 16K and rate-saturation at 32 rps).
The unreconciled cells in `runpod_full/` were stub cells caused by
vLLM `--max-model-len=4096` rejecting the 16K-token RAG and 5K+
agentic prompts — a prompt-shape mismatch, not an aggregator bug. The
unreconciled cells in `runpod_v11/` are HTTP 400 responses at the
16K prompt ceiling, again a model-cap issue.

**Blockers:**
- RAG/agentic workloads overflow vLLM's `--max-model-len` (default 4096); 8-32K prompt exceeds it.
- Multi-model sweep (qwen3-30b) requires >80 GB VRAM; current pilot pod is H100 80 GB and chunks OK
  but at high rates, OOM.
- Multi-node P/D (`cuda_ipc` fails on separate pods) needs topology testing for `tcp` / `rdma` UCX.
- `disagg` and `disagg_tier` cells in the reduced sweep were never generated (sweep stopped before reaching them); re-running with `run_pending` would resume from chunked × qwen2.5-7b and pick up the remaining cells.

**Cheapest fix (updated 2026-07-16):** 16384 is NOT enough — the v1.1
sweep already ran at `--max-model-len=16384` and RAG cells still
returned HTTP 400. The local prompt preflight (`bench/preflight.py`,
run at $0 on the M1) measures the RAG worst-case prompt+output at
**18,539 tokens**; the correct budget is `--max-model-len 20480`.
Verify with the 2-cell `configs/runpod_context_repair.yaml` probe
(<$1) before re-running any pending cells. This is the first
GPU-blocked item in `docs/GPU_EXECUTION_PLAN.md`, executed through the
staged ladder in `docs/GPU_COST_OPTIMIZATION.md`.

**Need user confirmation** per `Do NOT launch GPU pods` policy.

---

## Gap 8: Multi-node P/D cluster

**Roadmap reference:** v2 / MULTI-01.

**Status:** Out of scope for v0.1.0 per ROADMAP. Useful v1.1 work — but expensive (~$30 for
a couple hours of 2× H100 on RunPod).

**Blocker:** user confirmation + spawn flow.

---

## Gap 11: `autoscaler/` orphan directory at repo root

**Current state (closed in commit `885891b` 2026-07-14):** `autoscaler/TUNING.md` moved to `docs/autoscaler/TUNING.md`; the orphan `autoscaler/` repo-root directory was removed. 3 TDD tests in `tests/test_doc_paths.py` pin the move and the removal.

Status: **✅ Closed (v0.2.0)**.

> Historical (now-closed) gap description, retained for traceability:
> `autoscaler/TUNING.md` (192 lines) was the only file in a repo-root
> `autoscaler/` directory not in `pyproject.toml` wheel `packages`.
> Recommended fix: move to `docs/autoscaler/TUNING.md` and remove
> the orphan dir. **Outcome:** done.

---

## TDD work order (this session — completed 2026-07-14)

| Order | Gap | Commit | Status |
|-------|-----|--------|--------|
| 1 | Gap 5 (STATE.md) | `c5f7cc6 chore(planning): STATE.md frontmatter refresh — all 8 phases shipped` | ✅ |
| 2 | Gap 11 (autoscaler/ orphan) | `885891b chore(hygiene): move autoscaler/TUNING.md to docs/ (close GAP_REPORT §11)` | ✅ |
| 3 | Gap 1 (no_history counter) | wired in router + registry | ✅ |
| 4 | Gap 2 (prefix_index_size_bytes) | wired in router + registry | ✅ |
| 5 | Gap 3 (AUTO 0-drop + thrash) | `3b2e964 feat(autoscaler): thrash + inflight-dropped counters (AUTO zero-drop gate)` | ✅ |
| 6 | Gap 4 (.gitignore parquet) | REPRO-03 parquet whitelist | ✅ |
| 7 | Gap 6 (Grafana JSON placeholder) | `ade9526 feat(obs): Grafana dashboard JSON for OBS-01 metrics (OBS-02 placeholder)` | ✅ |

Each commit gated by `ruff check . && mypy . && pytest -q --no-header --no-cov` exiting 0.

---

## Items NOT in scope (this session)

| Item | Why deferred | Where it lives now |
|------|--------------|--------------------|
| Gap 7 (full 216 sweep) | GPU spend required; user-confirm needed | `docs/GPU_EXECUTION_PLAN.md` §1 |
| Gap 8 (multi-node) | v1.1 / MULTI-01 | `docs/GPU_EXECUTION_PLAN.md` §4 |
| Gap 9 (trained EAGLE-3) | Separate project (DraftForge) | DraftForge repo |
| Gap 10 (real LMCache gRPC) | v1.1 | `docs/GPU_EXECUTION_PLAN.md` §2 (proxy until real wire) |

---

*Report originally generated 2026-07-14; refreshed 2026-07-15 to
mark Gaps 5, 6, and 11 as closed and align gap counts with the
v0.3.0 disk state (390/25/97 %). Authored against live disk state
+ `.planning/` plans.*
