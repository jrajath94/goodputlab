---
phase: "01-topologies-topo"
plan: "01"
subsystem: skeleton
tags: [foundation, tooling, scaffolding]
requires: []
provides:
  - "Python 3.11+ project metadata + ruff/mypy/pytest config"
  - "Importable core/ + control/ package stubs"
  - "Makefile command surface for all Phase 1 plans"
  - "Hardened .gitignore covering protected secrets + generated artifacts"
affects: ["01-02", "01-03", "01-04", "01-05", "01-06", "01-07", "01-08"]
tech-stack:
  added:
    - "Python 3.11+ (requires-python)"
    - "ruff 0.15.x (lint)"
    - "mypy 1.19.x (strict mode, static checks)"
    - "pytest 9.x + pytest-cov (smoke + coverage)"
    - "FastAPI 0.115, Pydantic v2, uvicorn, httpx, requests, prometheus-client (declared deps)"
  patterns:
    - "Makefile is the public command surface (no flag memorization)"
    - "mypy strict with missing-imports tolerance for Phase-1 deferred packages"
    - "Single docker-compose.yml w/ profiles (declared in Makefile, ships in 01-03)"
key-files:
  created:
    - "pyproject.toml"
    - "Makefile"
    - "core/__init__.py"
    - "control/__init__.py"
    - "tests/conftest.py"
    - "tests/test_project_skeleton.py"
  modified:
    - ".gitignore"
key-decisions:
  - "Package build via Hatchling (lightweight; no setuptools)"
  - "mypy uses ignore_missing_imports so later topology-plan deps can be declared without breaking today's strict scan"
  - "Makefile uses `python -m <tool>` so it inherits whatever interpreter the operator invokes; avoids shebang drift"
  - "health/sentinel targets point to scripts/ and tests/sentinel.py that ship in 01-05/01-06 — these dry-run today and fail loudly at runtime until the later plans land"
requirements-completed: ["TOPO-07"]
duration: ~10 min
completed: "2026-07-08"
---

# Phase 01 Plan 01: Project Skeleton Summary

Established the Python 3.11+ project skeleton with ruff + mypy + pytest, an importable `core`/`control` package stub layout, the Makefile command surface every later Phase 1 plan hangs off of, and a hardened `.gitignore` covering protected secrets, runtime env files, and generated sentinel fixtures. The plan is pure scaffolding — no GPU containers, no `scripts/health.sh`, no `tests/sentinel.py` (those ship in 01-05 / 01-06 per the locked plan slice).

## Acceptance gate (run on phase-1/01-01-skeleton HEAD)

```
$ python3 -m ruff check .
All checks passed!

$ python3 -m mypy .
Success: no issues found in 4 source files

$ python3 -m pytest -q
.                                                                        [100%]
================================ tests coverage ================================
_______________ coverage: platform darwin, python 3.14.3-final-0 _______________

Name                  Stmts   Miss  Cover   Missing
---------------------------------------------------
control/__init__.py       0      0   100%
core/__init__.py          0      0   100%
---------------------------------------------------
TOTAL                     0      0   100%
1 passed in 0.05s
```

All Makefile targets dry-run cleanly:

```
make -n lint            → python -m ruff check . && python -m mypy .
make -n test            → python -m pytest
make -n compose-config  → docker compose --profile {colocated,chunked,disagg,disagg-tier} config
make -n up-colocated    → docker compose --profile colocated up -d
make -n up-chunked      → docker compose --profile chunked up -d
make -n up-disagg       → docker compose --profile disagg up -d
make -n up-disagg-tier  → docker compose --profile disagg-tier up -d
make -n down            → docker compose down (all profiles)
make -n health          → bash scripts/health.sh all
make -n sentinel        → python3 tests/sentinel.py --mode check --base-url http://localhost:19100/v1
make -n provision       → echo "see 01-02 (provision.sh ships with the next plan)"
```

## Tasks Completed

| Task | Name | Commit | Files |
| ---- | ---- | ------ | ----- |
| 1 | Python project + importable package skeleton | `74cd48c` | pyproject.toml, core/__init__.py, control/__init__.py, tests/conftest.py, tests/test_project_skeleton.py |
| 2 | Makefile command contract + hardened .gitignore | `ab89891` | Makefile, .gitignore |

## Deviations from Plan

None — plan executed exactly as written.

### Notes for downstream plans
- `scripts/health.sh` is referenced by `make health` and `scripts/health.sh all` in `make -n` output; runtime execution will fail until plan 01-06 ships the script. This matches the locked 01-01 plan slice.
- `tests/sentinel.py` is referenced by `make sentinel`; runtime execution will fail until plan 01-05 ships the sentinel CLI. This also matches the locked 01-01 plan slice.
- `provision` target is a stubbed echo per orchestrator constraint; `provision.sh` ships in plan 01-02.
- `pyproject.toml` declares FastAPI/Pydantic/etc. as runtime deps but does not install them; `make install-dev` (a pip install) is the gate before those become available. Today's acceptance runs pre-install and the test only imports `core`/`control`.

## Files Created/Modified

```
.gitignore                     | 15 +++++++++++
Makefile                       | 52 ++++++++++++++++++++++++++++++++++++++
control/__init__.py            |  1 +
core/__init__.py               |  1 +
pyproject.toml                 | 57 ++++++++++++++++++++++++++++++++++++++++++
tests/conftest.py              |  1 +
tests/test_project_skeleton.py | 10 ++++++++
7 files changed, 137 insertions(+)
```

All seven paths match the `files_modified` list in PLAN.md frontmatter.

## Branch Isolation

- Branch: `phase-1/01-01-skeleton`
- Two commits pushed on the branch (`74cd48c`, `ab89891`); no commits on `main` since branch creation.
- Branch NOT merged; NOT deleted (per orchestrator constraint).

## Self-Check: PASSED

- All 7 listed files present on disk (`OK` for every `for` probe).
- Branch HEAD points at `phase-1/01-01-skeleton`.
- ruff / mypy / pytest all green.
- `make -n` for every required target emits the documented command.
