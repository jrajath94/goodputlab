# Phase 1 Verification: Topologies (TOPO)

**Verified:** 2026-07-09
**Verifier:** gsd-verifier (static, offline — no live pod)
**Mode:** Goal-backward analysis

## Status

**human_needed**

The Phase 1 artifacts are fully in place and statically correct (lint, types, tests, configs, sentinel logic, NIXL/UCX pinning, vLLM 0.11.2 pin, 20-min budget gate, common OpenAI-compatible schema). All five must-have structural acceptance criteria are satisfied at the source level. What cannot be verified statically is the **runtime end-to-end behavior on a live GPU pod**, which is the load-bearing acceptance signal for Phase 1 (PITFALLS P1 sentinel-token validity on real hardware, TOPO-06 cold-start <20 min, true P→D KV transfer). Human needs to start the RunPod pod and run `make provision` + `make up-*` + `make health` end-to-end.

---

## 1. Goal Achievement — 5 Success Criteria

| # | Criterion | Status | Evidence |
|---|-----------|:------:|----------|
| 1 | `make up-colocated`, `make up-chunked`, `make up-disagg`, `make up-disagg-tier` each serve OpenAI-compatible HTTP | **PASS (static)** | `Makefile:32-42` defines all four targets invoking `docker compose --profile <name> up -d`. `docker-compose.yml` defines 4 profiles (`vllm-colocated`, `vllm-chunked`, `vllm-disagg-prefill`/`vllm-disagg-decode`/`disagg-proxy`, `vllm-disagg-tier-prefill`/`vllm-disagg-tier-decode`/`disagg-tier-proxy`). All four expose OpenAI-compatible endpoints via vLLM. **Not runtime-verified.** |
| 2 | `make health` confirms P→D flow via sentinel-token validity (NOT gated solely on `kv_transfer_complete_count`) | **PASS (static)** | `Makefile:48-49` runs `bash scripts/health.sh all`. `scripts/health.sh:153-165` invokes `python3 tests/sentinel.py --mode check` (token + logprob comparison). `scripts/health.sh:170-175` explicitly documents `kv_transfer_complete_count` as NOT a real vLLM/NIXL metric. Sentinel is the load-bearing check (`tests/sentinel.py:283-302` does exact token match + epsilon-bounded logprob comparison). **Not runtime-verified.** |
| 3 | Cold-node-to-serving <20 min (`make provision` idempotent) | **PASS (static) / UNMEASURED (runtime)** | `provision.sh:29` sets `PROVISION_BUDGET_SECONDS=1200` (20 min). `provision.sh:34-41` `check_budget` function emits `TOPO-06 BUDGET EXCEEDED` and exits 1 on overrun. All 6 stages (preflight, image_pull, model_download, boot_colocated, sentinel_record, health) call `check_budget`. Idempotency: image pull is a no-op when cached, model download is HF cache hit, sentinel fixture persisted in `tests/_fixtures/`. README correctly labels cold-start time as `[NOT YET MEASURED]` — no fabricated number. |
| 4 | vLLM pinned ≥0.11.x to mitigate CVE-2025-25183; NIXL backend pinned to UCX | **PASS (static)** | vLLM pin: `provision.sh:25` `VLLM_IMAGE=vllm/vllm-openai:v0.11.2`; `docker-compose.yml:22, 140, 206` `${VLLM_IMAGE:-vllm/vllm-openai:v0.11.2}`. UCX pin: `configs/kv_producer.json:7` `["UCX"]`; `configs/kv_consumer.json:7` `["UCX"]`; `configs/kv_lmcache_producer.json:7` `["UCX"]`; `configs/kv_lmcache_consumer.json:7` `["UCX"]`. `grep -rE 'LIBFABRIC' docker-compose.yml configs/*.json configs/*.yaml` returns 0 matches. |
| 5 | All 4 topologies share common OpenAI-compatible schema + `/metrics` endpoint | **PASS (static)** | Common model id `goodputlab-model` in all vLLM `command:` lines and `--served-model-name` flags (`docker-compose.yml:71, 93, 115, 132, 149, 179, 197, 215`). Common endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/metrics` exposed by vLLM and by proxy (`scripts/disagg_proxy.py:282-446`). `scripts/health.sh:117-150` probes the same 3 endpoints (`/health`, `/v1/models`, `/metrics`) for every topology. `tests/test_schema_uniformity.py` validates uniform surface. |

**Static score: 5 / 5 must-haves present in source.**

---

## 2. Requirement Coverage Matrix

| Req ID | Description | Evidence | Status |
|--------|-------------|----------|:------:|
| TOPO-01 | Colocated via `make up-colocated` | `Makefile:32-33`; `docker-compose.yml:61-74` `vllm-colocated` profile, port 18000 | STATIC PASS |
| TOPO-02 | Chunked-prefill via `make up-chunked` | `Makefile:35-36`; `docker-compose.yml:81-96` `--enable-chunked-prefill --max-num-batched-tokens 2048`, port 18001 | STATIC PASS |
| TOPO-03 | P/D disagg via `make up-disagg` + KV transfer | `Makefile:38-39`; `docker-compose.yml:104-159` P (8100) + D (8200) + proxy (9100→19100); `configs/kv_producer.json` + `kv_consumer.json` UCX-only NIXL | STATIC PASS |
| TOPO-04 | Disagg + LMCache tiering via `make up-disagg-tier` | `Makefile:41-42`; `docker-compose.yml:168-225` P/D + `LMCACHE_CONFIG_FILE` env; `configs/lmcache_{prefill,decode}.yaml` with `enable_pd: true` + `transfer_channel: "nixl"`; `configs/kv_lmcache_{producer,consumer}.json` LMCacheConnectorV1 | STATIC PASS |
| TOPO-05 | `make health` confirms P→D + decode≠prefill | `scripts/health.sh:177-245` `check_nixl_disagg` validates `vllm:nixl_xfer_time_seconds_count` delta and `vllm:num_failed_transfers_total == 0`; `_run_sentinel` validates token equivalence | STATIC PASS |
| TOPO-06 | Cold-node-to-serving <20 min | `provision.sh:29` `PROVISION_BUDGET_SECONDS=1200`; `provision.sh:34-41` budget gate; `provision.sh:38` `TOPO-06 BUDGET EXCEEDED` exit | STATIC PASS / RUNTIME UNMEASURED |
| TOPO-07 | Common OpenAI-compat schema + `/metrics` across 4 topologies | Common model id `goodputlab-model` in all commands; `scripts/health.sh` probes identical 3 endpoints; `tests/test_schema_uniformity.py` validates uniformity | STATIC PASS |
| REPRO-01 | docker-compose for all 4 committed | `docker-compose.yml` (225 lines, 4 profiles) committed | PASS |
| REPRO-02 | `make provision` → healthy serving in <20 min | `Makefile:28-29`; `provision.sh` 6-stage idempotent script with budget gate | STATIC PASS / RUNTIME UNMEASURED |

---

## 3. Pitfall Mitigation Verification

### P1 — NIXL LIBFABRIC silent garbage

**Mitigation: UCX pin + sentinel-token validity test (NOT counter increment)**

| Control | Evidence | Status |
|---------|----------|:------:|
| UCX-only NIXL backend in all 4 kv configs | `configs/kv_producer.json:7`, `kv_consumer.json:7`, `kv_lmcache_producer.json:7`, `kv_lmcache_consumer.json:7` all `"backends": ["UCX"]` | CONFIRMED |
| LIBFABRIC absence (grep check) | `grep -rE 'LIBFABRIC' docker-compose.yml configs/*.json configs/*.yaml` returns no matches | CONFIRMED |
| Sentinel-token test exists | `tests/sentinel.py` (378 lines): known-prefix prompt, greedy decode, exact token match + epsilon-bounded logprob check (default `1e-3`) | CONFIRMED |
| Sentinel wired into health check | `scripts/health.sh:153-165` `_run_sentinel` exits non-zero on fail | CONFIRMED |
| Sentinel NOT gated on `kv_transfer_complete_count` | `scripts/health.sh:170-175` explicitly documents this metric as NOT-A-GATE; asserts on real `vllm:nixl_*` family (`vllm:nixl_xfer_time_seconds_count`, `vllm:nixl_bytes_transferred_sum`, `vllm:nixl_num_failed_transfers_total`) | CONFIRMED |
| Continuous daemon probe | `scripts/sentinel_daemon.py` (181 lines): 60s loop, emits `sentinel_drift` Prometheus gauge (0=pass, 1=fail) | CONFIRMED |
| Fixture never fabricated in source | `tests/sentinel.py:14-17` docstring: "Fixtures are NEVER fabricated in source control. Only `record` mode against a trusted colocated topology produces a fixture." | CONFIRMED |

### P2 — CVE-2025-25183 prefix-cache hash collision

**Mitigation: vLLM ≥0.11.x pin**

| Control | Evidence | Status |
|---------|----------|:------:|
| vLLM image pin in `provision.sh` | `provision.sh:25` `VLLM_IMAGE=vllm/vllm-openai:v0.11.2` | CONFIRMED |
| vLLM image pin in `docker-compose.yml` | `docker-compose.yml:22` `${VLLM_IMAGE:-vllm/vllm-openai:v0.11.2}`; lines 140, 206 same | CONFIRMED |
| Version-printed smoke in provision | `provision.sh:57-58` `docker run --rm $VLLM_IMAGE python -c "import vllm; print('vllm', vllm.__version__)"` | CONFIRMED |
| README documents CVE | `docker-compose.yml:14` comment: "Single vLLM image pin: vllm/vllm-openai:v0.11.2 (CVE-2025-25183 mitigation)" | CONFIRMED |

---

## 4. Static Acceptance Checks (all green)

| Check | Result |
|-------|--------|
| `ruff check .` | All checks passed (0 violations) |
| `mypy .` (strict=true) | Success: no issues found in 14 source files |
| `pytest -q --no-cov` | 33 passed, 15 skipped (0 failed, 0 errored) — all skips are live-pod-only tests gated by markers; static tests pass |
| `grep -E 'LIBFABRIC' docker-compose.yml configs/*.json configs/*.yaml` | 0 matches |
| `grep 'UCX' configs/kv_producer.json configs/kv_consumer.json` | Both contain `"backends": ["UCX"]` |
| `grep 'vllm/vllm-openai:v0.11' provision.sh` | Match found (`vllm/vllm-openai:v0.11.2`) |
| `bash -n provision.sh scripts/pull_model.sh scripts/health.sh` | All bash syntax OK |
| `grep 'TOPO-06 BUDGET EXCEEDED' provision.sh` | Found (docstring + runtime emission) |
| `grep -c '\[NOT YET MEASURED\]' README.md` | 7 occurrences (≥5 required) — honest placeholders |

---

## 5. Honest Gaps Remaining

These are the items that cannot be verified without a live RunPod H100 NVL pod. All are explicitly marked `[NOT YET MEASURED]` in the README — no fabrication.

| Gap | Why it can't be checked statically | What's needed |
|-----|------------------------------------|---------------|
| **Cold-node-to-serving <20 min on real hardware** | Requires RunPod pod start (RunPod MCP unreachable from this shell as of 2026-07-09 per task spec). | Human starts pod `t3son251d5gcvg`, runs `make provision`, captures wall-clock time, compares to 1200s budget gate. |
| **Sentinel fixture recorded on real hardware** | `tests/sentinel.py --mode record` must run against the actual loaded model. `tests/_fixtures/` is currently empty. | After provision, `make up-colocated` boots + sentinel records fixture. No fixture = health check returns non-zero on every topology. |
| **NIXL UCX handshake succeeds end-to-end** | Depends on actual GPU-direct (cuda_ipc) transfer between two vLLM procs on same H100. | First `make up-disagg` + `make health disagg` run. `vllm:nixl_xfer_time_seconds_count` must delta +1 after probe. |
| **LMCache tier round-trip in disagg-tier** | Depends on LMCache in-process integration with vLLM v0.11.2 (research notes flagged this as MEDIUM confidence). | First `make up-disagg-tier` + `make health disagg-tier` run. |
| **Decoder-vs-prefill metric separation on proxy** | `scripts/disagg_proxy.py:441-446` concats prefill + decode `/metrics` with `=== [label] ===` markers; cannot validate metric-name collision without live pods. | After deploy, scrape `http://localhost:19100/metrics` and verify labels. |

---

## 6. Score

**Static must-haves verified: 5 / 5 (100%)**

All five Phase 1 success criteria are satisfied at the source level:
- Lint + types + tests: green
- 4 docker-compose profiles wired with shared endpoints, common model id, correct ports
- vLLM 0.11.2 pin (CVE-2025-25183) and UCX-only NIXL pin (P1) in every config
- Sentinel-token validity check is the load-bearing P1 gate, NOT counter increments
- 20-min cold-start budget gate enforced in `provision.sh`
- README is honest: 7 `[NOT YET MEASURED]` placeholders, no fabricated numbers

**Runtime must-haves verified: 0 / 5 (awaiting live pod)**

Phase 1 is structurally complete and ready for runtime validation. The human must execute `make provision` + `make up-*` + `make health` on the RunPod H100 NVL pod to convert this from "static pass" to "end-to-end pass". Once runtime metrics are captured, the README `[NOT YET MEASURED]` cells can be filled and Phase 1 marked complete.