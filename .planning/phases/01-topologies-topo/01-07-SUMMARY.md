---
phase: "01-topologies-topo"
plan: "07"
subsystem: topologies-quickstart
tags: [docs, test, topo-01, topo-02, topo-03, topo-04, topo-07]
dependency_graph:
  requires:
    - "01-03 compose profiles"
    - "01-04 disagg proxy contract"
  provides:
    - "tests/test_topos.py"
    - "tests/test_schema_uniformity.py"
    - "README.md Phase 1 quickstart"
  affects:
    - "Phase 2-8 README sections (future updates)"
tech-stack:
  added: []
  patterns:
    - "Live-mode pytest gate via GOODPUTLAB_RUN_LIVE=1"
    - "Skip-by-default for offline / pod-stopped development"
    - "Common endpoint contract assertion across all profiles"
key-files:
  created:
    - "tests/test_topos.py"
    - "tests/test_schema_uniformity.py"
  modified:
    - "README.md"
decisions:
  - "Marker-only vLLM metric assertion (`vllm:`, `vllm_disagg_prefill_`, `vllm_disagg_decode_`) instead of exact metric names that drift across vLLM minors."
  - "Skip-mode default (no pod call) so local pytest never fails when t3son251d5gcvg is stopped."
  - "All Phase 1 measured-result cells kept as `[NOT YET MEASURED]`; no benchmark numbers fabricated ahead of the pod run."
  - "Single-sentence project description kept literal (control plane framing) — no marketing adjectives."
metrics:
  duration: "~3 min"
  completed_date: "2026-07-08"
  tasks: 3
  files: 3
---

# Phase 1 Plan 7: README + integration tests Summary

One-liner: Phase 1 quickstart surface + skip-by-default runtime smoke tests for the four topology endpoint contracts.

## Tasks Executed

| # | Name | Commit | Files |
|---|------|--------|-------|
| 1 | Runtime topology smoke tests | `75fb259` | `tests/test_topos.py` |
| 2 | Schema uniformity tests | `fd9948d` | `tests/test_schema_uniformity.py` |
| 3 | Phase 1 quickstart + placeholders | `c5323b7` | `README.md` |

## Verification Results

| Check | Command | Result |
|-------|---------|--------|
| Compile | `python3 -m compileall tests/test_topos.py tests/test_schema_uniformity.py` | pass |
| Skip-mode tests | `GOODPUTLAB_RUN_LIVE=0 python3 -m pytest tests/test_topos.py tests/test_schema_uniformity.py -q` | pass (10 skips, 0 fail) |
| Ruff | `python3 -m ruff check tests/test_topos.py tests/test_schema_uniformity.py` | pass |
| mypy | `python3 -m mypy tests/test_topos.py tests/test_schema_uniformity.py` | pass |
| README strings | `grep -q 'make provision' README.md` (and the 12 other required strings) | pass |
| README no emoji | scan for U+1F300-U+1FAFF + adjacent ranges | pass (0 emoji) |
| Acceptance: 4 topologies + 4 ports + gate + model id in test_topos | grep checks | pass |
| Acceptance: `/health`, `/v1/models`, `/metrics`, model id, `[NOT YET MEASURED]` (comment only) in test_schema_uniformity | grep checks | pass |

## Deviations from Plan

None — plan executed exactly as written.

## Auth Gates

None. No external auth required for this plan; tests are gated by
`GOODPUTLAB_RUN_LIVE=1` and default to skip.

## Deferred Items

| Category | Item | Deferred To |
|----------|------|-------------|
| Live evidence | Cold-start time, sentinel pass count, TTFT, ITL, cost numbers | Phase 8 BENCH campaign against `t3son251d5gcvg` |
| Helm / multi-node chart | Topology deploy across multiple pods | Phase 8 stretch (REPRO section) |
| Full benchmark of the four profiles | Goodput curves, crossover analysis | Phase 8 BENCH plan |

## Threat Surface Scan

New security surface created by this plan:

| Flag | File | Description |
|------|------|-------------|
| `threat_flag: spoofing` | `tests/test_topos.py`, `tests/test_schema_uniformity.py` | Tests assert served-model id from `/v1/models` so a wrong `--served-model-name` upstream is caught (T-01-07-S, mitigated). |
| `threat_flag: denial-of-service` | `tests/test_topos.py`, `tests/test_schema_uniformity.py` | Default pytest run skips live checks; `GOODPUTLAB_RUN_LIVE=1` is opt-in (T-01-07-D, mitigated). |
| `threat_flag: repudiation` | `README.md` | Every measurement cell marked `[NOT YET MEASURED]` until logs exist (T-01-07-R, mitigated). |
| `threat_flag: information-disclosure` | `tests/*`, `README.md` | No secrets, tokens, or private env values documented; tests require no API key (T-01-07-I, mitigated). |
| `threat_flag: tampering` | `pyproject.toml` deps | `requests` and `pytest` already declared; no new package installs in this plan (T-01-SC, mitigated). |

## Notes for the Next Plan (Wave 4 / 01-02)

- `make provision` is still a stub. 01-02 must land `provision.sh` so
  the README's `make provision` quickstart command becomes runnable.
- `make health` is also a stub pending `scripts/health.sh` from 01-06;
  when 01-06 is merged, README commands can be exercised end-to-end.
- After 01-06 lands, the live path of these tests can be exercised on
  the pod with `GOODPUTLAB_RUN_LIVE=1 pytest tests/test_topos.py tests/test_schema_uniformity.py -q`.

## Self-Check

- [x] `tests/test_topos.py` exists, contains `goodputlab-model`,
      `18000`/`18001`/`19100`/`19200`, `GOODPUTLAB_RUN_LIVE`, four
      topology names, and a parametrized `test_topology_chat_completion`.
- [x] `tests/test_schema_uniformity.py` exists, contains `/health`,
      `/v1/models`, `/metrics`, `goodputlab-model`, and the same live
      gate.
- [x] `README.md` rewritten with status, requirements, quickstart,
      topology table, measured-results placeholders, safety section,
      and limitations.
- [x] Three commits on `phase-1/01-07-readme` branch:
      `75fb259`, `fd9948d`, `c5323b7`.

Self-Check: PASSED
