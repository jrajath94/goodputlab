# Phase 1: Topologies (TOPO) - Context

**Gathered:** 2026-07-08
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous)

<domain>
## Phase Boundary

Deploy all 4 serving topologies (colocated, chunked-prefill, disagg, disagg+tiering) on the RunPod H100 NVL pod via docker-compose with profiles. Each topology serves OpenAI-compatible HTTP. P→D KV transfer validated by sentinel-token test (NOT just count increment) — PITFALLS P1 mitigation. Cold-node-to-serving <20min acceptance gate. Health check confirms P→D flow on disagg topologies + decode never runs prefill (via vLLM metrics). All 4 topologies share common OpenAI-compat schema + metrics endpoint. REPRO-01 (docker-compose files) + REPRO-02 (provision.sh) ship with Phase 1.

</domain>

<decisions>
## Implementation Decisions

### Model Selection
- **Qwen 3.6 latest stable** — verify exact HF model id during plan-phase research (Context7 + HuggingFace API). Default to FP8 quant for H100 NVL 94GB fit. If exact 3.6 release not on HF by 2026-07, fall back to most recent Qwen3.x release.

### NIXL KV-transfer Backend
- **UCX only** — pinned via `kv-transfer-config` `backends=["UCX"]`. LIBFABRIC explicitly excluded (PITFALLS P1: vllm #27055 silent garbage). No opt-in toggle in Phase 1.

### Sentinel-Token Validity Test (Layered Defense)
- **All three approaches combined:**
  1. **Standalone script** (`tests/sentinel.py`) — known-prefix request → verify decode first-token logits match expected (L2 < ε). Run post-deploy per topology.
  2. **Built into health-check** (`scripts/health.sh`) — runs sentinel as part of every `make health`. Auto-rollback / exit non-zero on fail.
  3. **Continuous background probe** (`scripts/sentinel_daemon.py`) — runs every 60s, emits `sentinel_drift` Prometheus gauge. Adds ~negligible resource use.

### Docker Compose Layout
- **Single file with profiles** — `docker-compose.yml` w/ profiles: `colocated`, `chunked`, `disagg`, `disagg-tier`. Invoke via `docker compose --profile disagg up`. Cross-topology diff stays in one file.

### Claude's Discretion
- vLLM invocation flags per topology (kv-transfer-config syntax, chunked-prefill on/off, NIXL handshake settings) — verify against live vLLM v0.11.x docs in plan-phase research.
- Exact provision.sh structure (system packages, venv, model download via HF_HUB_ENABLE_HF_TRANSFER).
- Grafana dashboard JSON baseline (deferred to Phase 8 OBS).
- Test fixture data for sentinel (deterministic token sequences).

</decisions>


## Existing Code Insights

### Reusable Assets
- None — greenfield project, no prior code.
- `.planning/PROJECT.md`, `REQUIREMENTS.md`, `research/SUMMARY.md`, `research/PITFALLS.md`, `RUNPOD.md` = planning artifacts to consume.
- RunPod pod `t3son251d5gcvg` (H100 NVL, stopped) — start when ready to deploy.

### Established Patterns
- Conventional Commits (`feat:`, `fix:`, `perf:`, `test:`, `docs:`, `refactor:`, `bench:`, `ci:`)
- `make` targets as primary interface (`make up-disagg`, `make health`, `make sentinel`, `make bench`)
- pytest ≥80% coverage on `core/` + `control/` modules (REPRO-04 gate)
- ruff + mypy on every commit (REPRO-05)

### Integration Points
- RunPod pod SSH + ports documented in `.planning/RUNPOD.md`
- HF model download via `HF_HUB_ENABLE_HF_TRANSFER=1` env var
- vLLM `/metrics` Prometheus endpoint (port 8000 default)
- NIXL UCX on same pod, simulated P→D via localhost (single H100)

</code_context>

<specifics>
## Specific Ideas

- PITFALLS P1 sentinel test is THE load-bearing safety mechanism — must pass before any other Phase measurement. Sentinel FAIL = abort + page operator.
- PITFALLS P2 CVE-2025-25183 — pin vLLM ≥0.11.x in provision.sh; verify version in `make provision` output.
- Cold-to-serving <20min gate (TOPO-06) — measured end-to-end, not per-component. Provision.sh + first `make up-colocated` must complete within budget.
- 4 topologies must serve IDENTICAL OpenAI-compat schema (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/metrics`) so bench scripts work uniformly.

</specifics>

<deferred>
## Deferred Ideas

- Helm charts for multi-node deployment → Phase 8 stretch (REPRO section)
- LMCache prewarming automation → Phase 5
- Spec-decode EAGLE-3 head loading → Phase 6
- Autoscaler role-flip → Phase 7
- Cross-region networking → v2 / out of scope
- Grafana dashboard JSON → Phase 8 OBS (baseline Prom scrape only in Phase 1)
- Production-grade Dockerfile hardening → post-v1

</deferred>