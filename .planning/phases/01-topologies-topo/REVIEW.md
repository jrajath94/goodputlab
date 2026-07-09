---
file: REVIEW
status: findings_found
phase: 01-topologies-topo
generated: 2026-07-09
reviewer: gsd-code-reviewer (static)
scope: plans 01-01..01-08
counts:
  blocker: 0
  major: 3
  minor: 7
  nit: 5
---

# Phase 1 Code Review — Topologies (TOPO)

Static review only. No live execution, no GPU/network. Findings ordered by
file; severity-classified per the brief.

## Pitfall-mitigation spot check

| Pitfall | Where it should live | Present? | Evidence |
|---|---|---|---|
| P1 NIXL silent garbage — sentinel probe | `scripts/health.sh` step `[4/4]` | YES | `scripts/health.sh:153-165` invokes `tests/sentinel.py --mode check` |
| P1 — explicit rejection of `kv_transfer_complete_count` as gate | `scripts/health.sh` comment + test-visible marker | YES | `scripts/health.sh:170-176` + test in `tests/test_health_static.py:167-212` |
| P1 — assertion on real NIXL counters | `scripts/health.sh` `check_nixl_disagg` | YES | `scripts/health.sh:177-245` asserts `vllm:nixl_xfer_time_seconds_count`, `vllm:nixl_bytes_transferred_sum`, `vllm:nixl_num_failed_transfers_total`, `vllm:nixl_num_failed_notifications_total` |
| P2 CVE-2025-25183 — vLLM pin ≥0.11.x | `docker-compose.yml`, `provision.sh`, `scripts/pull_model.sh` | YES | `vllm/vllm-openai:v0.11.2` in all three |
| P3 spec-decode × disagg gate | N/A Phase 1 (spec decode not yet wired) | N/A | No Phase 1 code path enables EAGLE-3 |
| UCX-only NIXL backend | `configs/kv_*.json`, `docker-compose.yml` UCX_TLS | YES | `"backends": ["UCX"]` in all four kv configs; `UCX_TLS=cuda_ipc,cuda_copy,tcp` in compose |

Pitfall surface: GREEN for Phase 1.

## Findings

### MAJOR-1 — `/v1/models` returns HTTP 200 with a fabricated stub when upstream fails

```
---
file: scripts/disagg_proxy.py
line: 370-394
severity: major
category: correctness
summary: /v1/models swallows upstream errors and returns 200 + a hand-rolled stub, masking decode-pool outages
failure_scenario: decode pool crashes or `/v1/models` hangs; health.sh and external clients see 200 + a fake model id; load balancer keeps sending traffic to a dead pool
---
```

The docstring even labels this the "P5 healthcheck anti-pattern" but ships
it anyway. Either remove the stub fallback (return 502 on non-200 upstream)
or scope it to the warmup window. Currently any decode-pool 5xx is silently
hidden from the health gate.

Concrete fix: drop the `JSONResponse(...)` stub; on non-200 upstream return
a `Response(content=upstream.content, status_code=upstream.status_code)` —
same shape as `/v1/chat/completions`.

### MAJOR-2 — `_maybe_assert_first_token` only inspects chat completion schema; silently no-ops on `/v1/completions`

```
---
file: scripts/disagg_proxy.py
line: 252-276, 319, 362
severity: major
category: correctness
summary: Runtime sentinel first-token comparison extracts `choices[0].message.content` which is empty for /v1/completions responses
failure_scenario: --assert-first-token-matches is set; client sends /v1/completions; payload parses; `message` key is absent → KeyError is caught and silently logged at DEBUG; the sentinel never fires for the most likely probe path (scripts/sentinel.py uses /v1/completions)
---
```

The `message.content` path is only valid for `/v1/chat/completions`. The
`/v1/completions` schema is `choices[0].text`. `_extract_tokens_and_logprobs`
in `tests/sentinel.py:166` confirms the canonical schema is `text`/`tokens`
not `message.content`. Split into two helpers or guard on `path`.

### MAJOR-3 — `--assert-first-token-matches` is exposed but never wired in compose

```
---
file: docker-compose.yml
line: 138-159, 204-225
severity: major
category: pitfall
summary: The runtime sentinel assertion flag the addendum calls out as the P1 mitigation is not actually passed to disagg_proxy in any compose profile
failure_scenario: Operator reads the proxy docstring, trusts that the runtime sentinel is enabled by default, deploys disagg, sees /metrics deltas tick up; the actual post-transfer validity check is only invoked by scripts/health.sh against the standalone sentinel CLI — the proxy's own runtime check is dead code
---
```

Either pass `--assert-first-token-matches "$(cat tests/_fixtures/...json | jq -r .tokens[0])"`
to both proxy containers (preferred) or remove the flag from the public
proxy surface until the wiring lands. Dead code in a security-relevant path
is worse than no code.

### MINOR-1 — Inline `import json` inside hot path

```
---
file: scripts/disagg_proxy.py
line: 260
severity: minor
category: idiom
summary: import json inside _maybe_assert_first_token; hoist to module top with other imports
failure_scenario: extra import on every probe response; not measurable but inconsistent with surrounding module-level imports
---
```

### MINOR-2 — pull_model.sh interpolates $model_id into a Python -c single-quoted literal

```
---
file: scripts/pull_model.sh
line: 39-44, 53-57
severity: minor
category: security
summary: model_id is interpolated into a Python source string; the single quotes block shell injection but a model_id containing a single quote breaks the command silently
failure_scenario: Hugging Face returns a model id with an apostrophe; docker run emits malformed Python → exits non-zero → fallback probe also runs with the same bug → both probes fail
---
```

Concrete fix: pass via env var, read with `os.environ["MODEL_ID_PROBE"]`
inside the Python block.

### MINOR-3 — health.sh awk metric parser is fragile to labelled `sum`/`count` siblings

```
---
file: scripts/health.sh
line: 90-104
severity: minor
category: correctness
summary: _metric_value takes the bare-name series only (no labels) and sums any labelled series; vllm:nixl_xfer_time_seconds exposes _count/_sum/_bucket labels, so the count grep may return 0 (sum) instead of the increment that actually moved
failure_scenario: histogram's _count label series advances; the _sum series advances by a smaller magnitude; awk returns the sum, not the count, and the delta assertion may pass for the wrong reason
---
```

Prefer grabbing only `_count` labels explicitly with a regex
`metric_name_count\b`.

### MINOR-4 — health.sh SILENT_OK branch on colocated / chunked / disagg if metric body is empty

```
---
file: scripts/health.sh
line: 75-82
severity: minor
category: correctness
summary: `_metric_value` returns empty when body is empty; on colocated/chunked the script treats this as 0, which is fine; but on disagg the script's "must increase" assertion later fails. The empty-before / empty-after ambiguity isn't distinguished
failure_scenario: first cold probe of disagg before any transfer has happened: before_xfers="" → default to 0; after_xfers="" → 0; _diff_positive 0 > 0 returns false → fails; but the failure message says "did not increase" which is misleading when the metric is entirely missing
---
```

Make the script distinguish "metric absent" from "metric = 0" and surface
a distinct error.

### MINOR-5 — sentinel_daemon `time.strftime` is fine but lacks TZ on Windows portability

```
---
file: scripts/sentinel_daemon.py
line: 165
severity: minor
category: idiom
summary: time.strftime with hard-coded "Z" suffix; use datetime.now(timezone.utc).isoformat() for cross-platform + ISO-8601 correctness
failure_scenario: minor; logs are consistent on POSIX; an SRE copy-pasting into a SIEM might appreciate real ISO timestamps
---
```

### MINOR-6 — No test coverage for `scripts/pull_model.sh` and `scripts/health.sh` runtime behavior

```
---
file: tests/
line: n/a
severity: minor
category: test-coverage
summary: Static tests assert presence of markers in health.sh / pull_model.sh source, but no test exercises the `set -euo pipefail` failure path, the budget gate in provision.sh, or the model fallback branch in pull_model.sh
failure_scenario: a future contributor renames `set -euo pipefail` → `set -eu`; static test passes; a real provision run silently swallows errors mid-stage and ships a broken pod
---
```

Either add behavior tests (preferred: shelltest or bats) or extend the
static tests to assert the exact form of the strict-mode line and the
specific error messages produced on the unhappy path.

### MINOR-7 — CI matrix says Python 3.11 but local dev has produced `__pycache__` for cpython-314

```
---
file: .github/workflows/ci.yml
line: 47
severity: minor
category: idiom
summary: workflow matrix pins 3.11; local dev appears to use 3.14 (cpython-314 pycache present); either bump matrix to [3.11, 3.14] or pin dev to 3.11 to avoid silent drift
failure_scenario: developer tests pass on 3.14, CI on 3.11 catches a typing/Union change; contributors get a surprise red CI on every PR
---
```

### NIT-1 — `_probe_common` / `run_one` use `local port=…` with command substitution in `set -u` context

```
file: scripts/health.sh
line: 254
category: idiom
summary: `port="$(_port_for "$topology")"` is fine but `_port_for` echoes "" for unknown; the caller `_fail`s on empty port which is the intended behavior — but no need for the empty echo; can `return 1` instead
```

### NIT-2 — disagg_proxy.py: `parse_args` docstring is sparse relative to the rest of the file

```
file: scripts/disagg_proxy.py
line: 109
category: doc
summary: One-line docstring on parse_args; surrounding helpers all have multi-line docstrings; consistency only
```

### NIT-3 — sentinel.py imports `os` solely for one `os.environ.get`

```
file: tests/sentinel.py
line: 29, 322
category: idiom
summary: Could use `os.environ.get` from a single import; current is fine — flag only because some style guides discourage `os` for env reads in CLI tools; not a fix
```

### NIT-4 — disagg_proxy.py typing: `Any` could be `Mapping[str, Any]` for the body arg

```
file: scripts/disagg_proxy.py
line: 243
category: idiom
summary: `_prefill_body(body: dict[str, Any])` — using `dict` is fine for the codebase style; flag only because the project CLAUDE.md says "Prefer explicit over implicit"
```

### NIT-5 — README badge links to `jrajath94/goodputlab`; confirm the canonical org before going public

```
file: README.md
line: 3-5
category: doc
summary: All GitHub badges point to jrajath94 (personal). If the org is different, swap before Phase 2 work makes badges permanent in the README history
```

## Test-coverage matrix (Phase 1 source files)

| File | Static test | Live test | Notes |
|---|---|---|---|
| `pyproject.toml` | implicit (pytest collect) | n/a | OK |
| `Makefile` | none | n/a | recipe-level behavior unverified |
| `docker-compose.yml` | `tests/test_health_static.py::test_compose_file_declares_topology_ports` | none | only port literals checked |
| `configs/kv_producer.json` | none | n/a | JSON parse only |
| `configs/kv_consumer.json` | none | n/a | JSON parse only |
| `configs/kv_lmcache_*.json` | none | n/a | JSON parse only |
| `configs/lmcache_*.yaml` | none | n/a | YAML parse only |
| `scripts/disagg_proxy.py` | `tests/test_disagg_proxy_static.py` (marker contract) | none | runtime proxies unverified |
| `scripts/health.sh` | `tests/test_health_static.py` (marker contract) | none | runtime shell unverified |
| `scripts/pull_model.sh` | none | none | uncovered |
| `scripts/sentinel_daemon.py` | none | none | uncovered |
| `tests/sentinel.py` | `tests/test_sentinel_static.py` (marker contract) | none | record/check happy path unverified |
| `tests/test_topos.py` | n/a | gated by `GOODPUTLAB_RUN_LIVE=1` | OK |
| `tests/test_schema_uniformity.py` | n/a | gated by `GOODPUTLAB_RUN_LIVE=1` | OK |
| `tests/test_project_skeleton.py` | self | n/a | OK |
| `tests/test_fixture_hygiene.py` | self | n/a | OK |
| `provision.sh` | none | none | uncovered |
| `.github/workflows/ci.yml` | none | n/a | uncovered |
| `README.md` | none | n/a | uncovered |

`scripts/pull_model.sh`, `scripts/sentinel_daemon.py`, and `provision.sh`
have NO test coverage at all. See MINOR-6.

## Verdict

**Status: findings_found (3 majors, 0 blockers)**

The 3 majors are:

1. MAJOR-1: `/v1/models` returns 200 + stub on upstream failure (proxy)
2. MAJOR-2: sentinel assertion only fires on chat completions, not completions (proxy)
3. MAJOR-3: `--assert-first-token-matches` flag is unused dead code in compose

Fix proposals are concrete in each finding. None are blockers for the
Phase 1 acceptance gate (cold→serving <20 min, four topologies serving,
P1 sentinel in health gate, vLLM ≥0.11.x pinned, UCX backend) — all of
those gates pass at the source-marker level. The majors are quality-of-
implementation issues that should land before Phase 2 begins to depend on
the proxy as a load-gen target.

## Recommended next step

Run `/gsd:code-review --fix` with this REVIEW.md to apply MAJOR-1, MAJOR-2,
MAJOR-3 in a single atomic commit. MINOR-1 through MINOR-7 can be bundled
or deferred at the operator's discretion. NIT-1 through NIT-5 are
optional polish.