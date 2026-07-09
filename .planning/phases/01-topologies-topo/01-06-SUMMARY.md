---
phase: "01-topologies-topo"
plan: "06"
subsystem: "health-gate"
tags: ["topologies", "health", "sentinel", "nixl", "p1-mitigation"]
dependency_graph:
  requires:
    - "01-03 compose"
    - "01-04 proxy"
    - "01-05 sentinel"
  provides:
    - "scripts/health.sh"
    - "tests/test_health_static.py"
  affects:
    - "Makefile health target"
    - "post-01-02 runtime validation"
tech-stack:
  added: []
  patterns:
    - "bash source-mark static tests"
    - "Prometheus text scraping via awk"
    - "counter delta assertions on disagg topologies"
    - "sentinel-cli subprocess for P1 silent-garbage detection"
key-files:
  created:
    - "scripts/health.sh"
    - "tests/test_health_static.py"
  modified: []
decisions:
  - id: "D-HEALTH-01"
    text: "Health-gate is the executable safety mechanism that gates all later benchmarking; runs every Phase 1 topology through /health + /v1/models + /metrics + sentinel."
  - id: "D-HEALTH-02"
    text: "NIXL counters (xfer_time_seconds_count, bytes_transferred_sum, num_failed_*) are the only valid KV-transfer evidence; kv_transfer_complete_count is explicitly NOT a gate (P1)."
  - id: "D-HEALTH-03"
    text: "Static source-mark tests gate behavior end-to-end so no runtime/GPU/network dependency leaks into CI."
metrics:
  duration: "~2 min"
  completed_date: "2026-07-08"
  tasks: 2
  files_created: 2
---

# Phase 1 Plan 06: Health Gate Summary

## One-liner

Topology-agnostic shell health gate that proves every Phase 1 topology serves the common OpenAI-compat contract AND corroborates P→D KV transfer via real vLLM/NIXL metrics + sentinel-token validity — closing PITFALLS P1 silent-garbage on the disagg path.

## What Shipped

| File | Lines | Purpose |
|------|-------|---------|
| `scripts/health.sh` | 295 | Topology health gate; accepts `colocated|chunked|disagg|disagg-tier|all`; probes endpoints, runs sentinel, asserts NIXL deltas on disagg. |
| `tests/test_health_static.py` | 232 | Static source-mark tests (no network/GPU); pin every required literal in `scripts/health.sh` so CI catches contract drift. |

## Verification

| Gate | Command | Result |
|------|---------|--------|
| Python compile | `python3 -m compileall tests/test_health_static.py` | OK |
| Shell syntax | `bash -n scripts/health.sh` | OK |
| Static tests | `python3 -m pytest tests/test_health_static.py -q` | **12/12 pass** |
| Ruff | `python3 -m ruff check tests/test_health_static.py` | OK (no issues) |
| Mypy (strict) | `python3 -m mypy tests/test_health_static.py` | OK (no issues) |

## Task-by-Task Receipt

### Task 1 — Static tests for health-gate behavior

- 12 source-mark tests written before `scripts/health.sh` existed.
- RED phase confirmed (10/12 fail on missing file; 2 dependency guards pass against `tests/sentinel.py` + `docker-compose.yml`).
- GREEN phase after Task 2 lands: all 12 pass.
- Required test function names: `test_health_uses_real_nixl_metrics` + sibling `test_health_rejects_kv_transfer_complete_count_as_gate`.

### Task 2 — Topology health shell script

- `scripts/health.sh` (`set -euo pipefail`) implements:
  1. Topology → port map: `colocated=18000`, `chunked=18001`, `disagg=19100`, `disagg-tier=19200` (locked to `docker-compose.yml`).
  2. Per-topology common probes: `curl /health`, parse `/v1/models` for `goodputlab-model`, scrape `/metrics` (Prometheus text shape).
  3. Per-topology sentinel run: `python3 tests/sentinel.py --mode check --base-url http://localhost:${PORT}/v1 --served-model-name goodputlab-model` — exit code propagates.
  4. Disagg-only NIXL gates: capture counters BEFORE a forced `/v1/completions` probe, force one transfer, capture AFTER, then:
     - `vllm:nixl_xfer_time_seconds_count` increases
     - `vllm:nixl_bytes_transferred_sum` increases
     - `vllm:nixl_num_failed_transfers_total` == 0 (presence required)
     - `vllm:nixl_num_failed_notifications_total` == 0 (presence required)
  5. Explicit NOT-A-GATE marker on `kv_transfer_complete_count` (P1 mitigation; the pre-NIXL/proxy log line is not a vLLM metric).
  6. `[OK] <topology> healthy` only after every check passes; non-zero exit on any failure.
- Metrics scraped via inline `awk` (no extra dependency); tolerant of missing counter on first probe, strict on disagg metrics presence.

## Deviations from Plan

- **Auto-handled Bash quality (no new deviation):** used shellcheck-style quoting throughout; `set -euo pipefail` honored; `curl --max-time` finite; no `eval`; no sourced env files.
- **Sentinel run ordering:** plan listed sentinel as a step; ship ordering is common probes → (disagg NIXL gates) → sentinel so a sentinel FAIL with a missing fixture is reported after the contract checks succeed. No behavior change, no metric re-read.
- **Probe-completion payload:** uses a single greedy `max_tokens=1` completion to force exactly one KV transfer (predictable counter delta), rather than re-running sentinel which can short-circuit on warm-cache replay.

### Auto-fixed Issues

- **None** during this plan.

### Branch Hygiene

- **Commit misroute (self-fixed, dev-only):** an earlier `git add tests/test_health_static.py && git commit` happened on `phase-1/01-07-readme` (a sibling executor's branch) due to a stale checkout. Committed cleanly onto my assigned branch `phase-1/01-06-health` via `git cherry-pick ff1ba07` (→ `9764e4c`). No working-tree files cross-pollinated. The duplicate commit on `phase-1/01-07-readme` is a sibling-team concern and out of scope here.

## Acceptance Criteria Status

| Criterion | Status |
|-----------|--------|
| `set -euo pipefail` in script | ✅ |
| Exact topology→port mappings for 18000/18001/19100/19200 | ✅ |
| `/health`, `/v1/models`, `/metrics` checked for every topology | ✅ |
| `tests/sentinel.py --mode check` invoked for every topology | ✅ |
| Uses `vllm:nixl_xfer_time_seconds_count`, `vllm:nixl_bytes_transferred_sum`, `vllm:nixl_num_failed_transfers_total`, `vllm:nixl_num_failed_notifications_total` on disagg | ✅ |
| `bash -n scripts/health.sh` exits 0 | ✅ |
| `python3 -m pytest tests/test_health_static.py -q` exits 0 | ✅ |
| `test_health_uses_real_nixl_metrics` exists | ✅ |
| `kv_transfer_complete_count` explicitly rejected as gate | ✅ (NOT-A-GATE marker) |
| Reject "fake metric on the proxy log line" trap | ✅ |

## Auth Gates / Deferred Items

- **No auth gates** encountered (purely local; no RunPod calls this plan).
- **Runtime pod validation deferred** to plan 01-02 bring-up: actual `make health` against the live H100 NVL pod will execute this script end-to-end. Static tests confirm contract; runtime confirms semantics.

## Self-Check

- [x] Files created exist on disk and in working tree (`scripts/health.sh`, `tests/test_health_static.py`).
- [x] Commits exist in `git log`:
  - `9764e4c test(01-06): static health gate tests`
  - `d94c3c3 feat(01-06): topology health shell script`
- [x] Static tests pass (12/12).
- [x] Bash syntax passes.
- [x] Ruff + mypy pass.

## Self-Check: PASSED

## Commits

- `9764e4c` — `test(01-06): static health gate tests` (RED → GREEN; 232 LOC)
- `d94c3c3` — `feat(01-06): topology health shell script` (295 LOC; executable)
