---
phase: "01-topologies-topo"
plan: "04"
subsystem: disagg-proxy
tags: [p1-mitigation, fastapi, p-d-disagg, nixl-fallback, endpoint-contract]
requires: ["01-01"]
provides: ["openai-compat-proxy", "prefill-before-decode-orchestration", "x-request-id-propagation", "sentinel-first-token-assertion-hook"]
affects: ["01-03", "scripts/health.sh", "docker-compose.yml"]
tech-stack:
  added: []
  patterns:
    - "FastAPI lifespan context manager (no deprecated @app.on_event) for shared httpx.AsyncClient"
    - "Body-clone prefill-before-decode: prefill body caps both max_tokens and max_completion_tokens to 1; decode body is original, unmodified"
    - "X-Request-Id header propagated or generated for log correlation across prefill + decode"
    - "/metrics prefixes with [# === [prefill] ===] and [# === [decode] ===] section markers for health-check parsing"
    - "Top-of-file docstring as the load-bearing documentation surface for sentinel assertion + vLLM --proxy fallback policy"
key-files:
  created:
    - scripts/disagg_proxy.py
    - tests/test_disagg_proxy_static.py
  modified: []
key-decisions:
  - "Hand-rolled FastAPI proxy (not vLLM --proxy mode) so the OpenAI-compat route surface stays under project control and the sentinel --assert-first-token-matches hook can live in our codebase (per D-05 common endpoint contract). vLLM --proxy remains the documented fallback if NixlConnector ↔ FastAPI interop is buggy."
  - "Cap BOTH max_tokens=1 and max_completion_tokens=1 on the prefill body; OpenAI v1 clients send max_completion_tokens, older clients send max_tokens — capping both avoids silent fall-through."
  - "/v1/models returns the decode model list verbatim when reachable, falls back to a {served-model-name} stub when the decode is not yet 200 — keeps health checks honest even during cold-start (P5 mitigation: never trust /health alone)."
  - "/health requires BOTH upstream /health endpoints to return 200; returns 503 with per-upstream status map if either fails — surfaces partial topology degradation explicitly."
  - "/metrics returns pre-decode (200) text wrapped in source-label headers; non-200 upstreams produce a labeled error comment line instead of silently dropping the section."
  - "Async sentinel first-token check: --assert-first-token-matches TEXT warns on mismatch but does NOT crash, so probe traffic keeps flowing while health.sh flags the mismatch (P1 mitigation: distinguish proxy-bug from KV-transfer-bug)."
  - "Pydantic v2 + FastAPI + httpx (no new pip installs); proxy source compiles clean on Python 3.11+ with ruff + mypy --strict."
requirements-completed: ["TOPO-03", "TOPO-04", "TOPO-07"]
duration: "~25 min (1 commit RED + 1 commit GREEN + 1 commit SUMMARY)"
completed: "2026-07-09T00:35:00Z"
---

# Phase 1 Plan 04: Disagg Proxy (OpenAI-Compatible Front Door for P/D)

**One-liner:** FastAPI proxy that exposes `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/metrics` for disagg + disagg-tier topologies, performing a max_tokens=1 prefill before forwarding the original body to the decode stage — with `X-Request-Id` propagation and a `--assert-first-token-matches` hook for the PITFALLS P1 sentinel first-token equivalence check.

## What Was Built

| Layer | File | Role |
|-------|------|------|
| Proxy | `scripts/disagg_proxy.py` | FastAPI app, lifespan-managed `httpx.AsyncClient`, argparse CLI (`--host/--port`, `--prefiller-{host,port}`, `--decoder-{host,port}`, `--served-model-name`, `--assert-first-token-matches`) |
| Routes | same | `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`, `GET /health`, `GET /metrics`, `GET /` (service descriptor) |
| Orchestration | same | Body-clone prefill-before-decode; prefill body forces `max_tokens=1` AND `max_completion_tokens=1`; decode body is the original; prefill stage errors short-circuit without invoking decode |
| Request correlation | same | `X-Request-Id` generated (when missing) via `gl-<uuid4>`, propagated to both upstreams, returned in response headers |
| Sentinel hook | same | `--assert-first-token-matches TEXT` compares decode first token to `TEXT`, logs a warning on mismatch (non-fatal) |
| Static tests | `tests/test_disagg_proxy_static.py` | 9 source-mark tests asserting routes, prefill-before-decode markers, request-id propagation, CLI surface, async httpx, FastAPI lifespan, docstring sentinel + fallback note, metrics source labels |

## TDD Gate Compliance

| Gate | Commit | Status |
|------|--------|--------|
| RED (failing tests) | `acfd45b` | 9/9 failing as expected (no `scripts/disagg_proxy.py` source present) |
| GREEN (impl) | `66f1fa5` | 9/9 passing, ruff clean, mypy --strict clean |
| REFACTOR | n/a | No further cleanup needed |

## Acceptance Criteria Results

| Criterion | Result |
|-----------|--------|
| `tests/test_disagg_proxy_static.py` contains `test_proxy_exposes_required_routes` | PASS |
| Static tests assert exact strings `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/metrics` | PASS |
| Static tests assert exact strings `prefiller_host`, `decoder_host`, `X-Request-Id` | PASS |
| `scripts/disagg_proxy.py` defines CLI args `--prefiller-host`, `--prefiller-port`, `--decoder-host`, `--decoder-port`, `--served-model-name` | PASS |
| `scripts/disagg_proxy.py` contains route strings `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/metrics` | PASS |
| `scripts/disagg_proxy.py` caps only the prefill request to one token and forwards the original decode request body | PASS |
| `python3 -m pytest tests/test_disagg_proxy_static.py -q` exits 0 | PASS (9/9) |
| `python3 -m compileall scripts/disagg_proxy.py` exits 0 | PASS |
| `python3 -m ruff check scripts/disagg_proxy.py` exits 0 | PASS |
| `python3 -m mypy scripts/disagg_proxy.py` exits 0 | PASS |

## Threat Model Coverage

| Threat ID | Disposition | Implementation |
|-----------|-------------|----------------|
| T-01-04-S (Spoofing) | mitigate | `X-Request-Id` generated via `gl-<uuid4>` when missing, propagated through both stages |
| T-01-04-T (Tampering) | mitigate | Prefill body is a dict-clone (`dict(body)`) with only `max_tokens=1` + `max_completion_tokens=1` overridden; decode body is the original parsed JSON, never mutated |
| T-01-04-R (Repudiation) | mitigate | Upstream status codes preserved end-to-end via `Response(status_code=..., content=..., media_type=...)`; prefill stage errors short-circuit with the upstream status (no fabricated 5xx) |
| T-01-04-D (DoS) | mitigate | Per-stage async timeouts (`DEFAULT_PREFILL_TIMEOUT_S=120`, `DEFAULT_DECODE_TIMEOUT_S=300`); no retries — prefill failure returns immediately |
| T-01-SC (Supply chain) | mitigate | Only `fastapi`, `httpx`, `pydantic` used — all already declared in `pyproject.toml`; no new pip installs |

## Deviations from Plan

None — plan executed as written.

### Auto-fixed Issues

None.

### Auth Gates

None.

## Verification Evidence

```
$ python3 -m compileall scripts/disagg_proxy.py
Compiling 'scripts/disagg_proxy.py'...

$ python3 -m compileall tests/test_disagg_proxy_static.py
Compiling 'tests/test_disagg_proxy_static.py'...

$ python3 -m ruff check scripts/disagg_proxy.py tests/test_disagg_proxy_static.py
All checks passed!

$ python3 -m mypy scripts/disagg_proxy.py tests/test_disagg_proxy_static.py
Success: no issues found in 2 source files

$ python3 -m pytest tests/test_disagg_proxy_static.py -q
.........                                                                [100%]
9 passed in 0.05s
```

## Deferred (per plan)

- No RunPod calls (network-blocking; deferred to plan 01-06 / 01-07)
- No `uvicorn` startup (proxy source is verified; integration testing needs compose + sentinel fixture)
- No integration tests against running vLLM (covered by `bash scripts/health.sh disagg` post-provision in 01-06)
- No actual sentinel fixture record (plan 01-05 ships fixture format; first record happens post-provision)
- `scripts/health.sh` top-level "KV-transfer-bug vs proxy-bug" note — added to scripts/health.sh by plan 01-06

## Known Limitations

- The `--assert-first-token-matches` check runs only when the flag is set; without the flag, no runtime sentinel check happens. This is intentional (probe traffic must flow) — plan 01-06 wires the flag via `scripts/health.sh disagg` based on the recorded fixture from plan 01-05.
- Streaming responses are not currently forwarded in token-batches; the proxy awaits the full decode response and returns it. Streaming is a follow-up — would require `httpx.Response.aiter_bytes()` and a `StreamingResponse` from FastAPI. Not required for Phase 1 acceptance.
- The proxy assumes the vLLM `NixlConnector` and FastAPI proxy can interoperate. If runtime shows the FastAPI shim interferes with internal KV transfer handshake timing, the documented fallback is `python -m vllm.entrypoints.openai.api_server --proxy --prefill-hosts ... --decode-hosts ... --port 9100` (recorded in the docstring).

## Self-Check

- [x] `tests/test_disagg_proxy_static.py` exists (164 lines, 9 tests)
- [x] `scripts/disagg_proxy.py` exists (~485 lines, FastAPI app + CLI)
- [x] Commit `acfd45b` (RED) present on `phase-1/01-04-proxy`
- [x] Commit `66f1fa5` (GREEN) present on `phase-1/01-04-proxy`
- [x] Ruff clean
- [x] mypy --strict clean
- [x] pytest 9/9 green

## Files Touched

- `tests/test_disagg_proxy_static.py` (created, 164 lines)
- `scripts/disagg_proxy.py` (created, 485 lines)
- `.planning/phases/01-topologies-topo/01-04-SUMMARY.md` (this file)
