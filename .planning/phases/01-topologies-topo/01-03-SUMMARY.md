---
phase: "01-topologies-topo"
plan: "03"
type: execute
subsystem: topology-deploy
tags: [docker-compose, nixl, lmcache, kv-transfer, ucx, phase-1]
dependency_graph:
  requires: ["01-01"]
  provides: ["TOPO-01", "TOPO-02", "TOPO-03", "TOPO-04", "TOPO-07", "REPRO-01"]
  affects: ["01-04", "01-06", "01-07"]
tech_stack:
  added:
    - docker-compose-v2-profiles
    - vllm/vllm-openai:v0.11.2
    - NIXL NixlConnector (UCX-only)
    - LMCacheConnectorV1 over NIXL PD
  patterns:
    - single-compose-multi-profile
    - UCX-only-NIXL
    - per-pool-side-channel-port
    - 0.45-split-gpu-mem
key_files:
  created:
    - configs/kv_producer.json
    - configs/kv_consumer.json
    - configs/kv_lmcache_producer.json
    - configs/kv_lmcache_consumer.json
    - configs/lmcache_prefill.yaml
    - configs/lmcache_decode.yaml
    - docker-compose.yml
decisions:
  - "D-02: NIXL backend pinned to UCX (alternate backends excluded per PITFALLS P1 vllm #27055)."
  - "D-04: Single docker-compose.yml with profiles colocated|chunked|disagg|disagg-tier."
  - "D-05: All 4 topologies expose common OpenAI-compat schema on served-model-name=goodputlab-model."
  - "Disagg pool gm=0.45 each (sum<=0.90 on 1xH100)."
  - "Per-pool NIXL side channels (5559 P, 5560 D) prevent handshake collision."
  - "LMCache PD transport = NIXL (transfer_channel=nixl), pd_buffer_device=cuda, pd_buffer_size=1GiB."
  - "Healthcheck probes /v1/models + asserts goodputlab-model id (not /health, which returns 200 pre-model-load)."
metrics:
  duration: "00:08:00"
  completed_date: "2026-07-08"
---

# Phase 1 Plan 03: Docker Compose Topologies Summary

One-liner: Single docker-compose file + NIXL/LMCache connector configs enabling all 4 Phase 1 topologies (colocated, chunked, disagg, disagg-tier) with UCX-only KV transfer.

## What Landed

| File | Purpose |
|------|---------|
| `configs/kv_producer.json` | NIXL prefill-side config (NixlConnector, kv_role=kv_producer, UCX) |
| `configs/kv_consumer.json` | NIXL decode-side config (same, kv_role=kv_consumer) |
| `configs/kv_lmcache_producer.json` | LMCache prefill-side config (LMCacheConnectorV1, UCX) |
| `configs/kv_lmcache_consumer.json` | LMCache decode-side config |
| `configs/lmcache_prefill.yaml` | LMCache PD sender (transfer_channel=nixl, pd_role=sender) |
| `configs/lmcache_decode.yaml` | LMCache PD receiver (transfer_channel=nixl, pd_role=receiver) |
| `docker-compose.yml` | 4-profile orchestration, 8 services, single file |

## Topology Mapping

| Profile | Services | External Port | Notes |
|---------|----------|---------------|-------|
| colocated | vllm-colocated | 18000→8000 | No chunked, no kv-transfer |
| chunked | vllm-chunked | 18001→8000 | --enable-chunked-prefill, --max-num-batched-tokens 2048 |
| disagg | vllm-disagg-prefill + vllm-disagg-decode + disagg-proxy | 19100→9100 | NIXL UCX, gm=0.45 each |
| disagg-tier | vllm-disagg-tier-prefill + vllm-disagg-tier-decode + disagg-tier-proxy | 19200→9100 | LMCacheConnectorV1 over NIXL |

## Hard Constraints Encoded

- NIXL backends: `["UCX"]` only — alternate backends excluded (PITFALLS P1, vllm #27055 silent garbage)
- LMCache PD transport: `transfer_channel: "nixl"` (D-03)
- vLLM image: `vllm/vllm-openai:v0.11.2` (CVE-2025-25183 mitigation)
- Model: `Qwen/Qwen3.6-35B-A3B-FP8` (override via `MODEL_ID` env)
- Served model name: `goodputlab-model` (D-05 common endpoint contract)
- Disagg pool: `--gpu-memory-utilization 0.45` each (sum ≤ 0.90 on 1×H100)
- Per-pool side channels: P=5559, D=5560, host=localhost
- UCX transport: `UCX_TLS=cuda_ipc,cuda_copy,tcp` (same-node GPU-direct)
- Healthcheck: probes `/v1/models` + asserts `goodputlab-model` id (not `/health` — returns 200 pre-model-load)
- Mounts: `/workspace/hf`, `./configs:/configs:ro`, `./scripts:/scripts:ro`
- Proxy command: `python /scripts/disagg_proxy.py --host 0.0.0.0 --port 9100 ...` (proxy implementation in plan 01-04)

## Verification

| Check | Result |
|-------|--------|
| `python3 -m json.tool configs/kv_producer.json` | pass |
| `python3 -m json.tool configs/kv_consumer.json` | pass |
| `python3 -m json.tool configs/kv_lmcache_producer.json` | pass |
| `python3 -m json.tool configs/kv_lmcache_consumer.json` | pass |
| `grep 'transfer_channel: "nixl"' configs/lmcache_*.yaml` | pass |
| `grep 'pd_role: "receiver"' configs/lmcache_decode.yaml` | pass |
| `grep -R "LIBFABRIC" docker-compose.yml configs` | none — fully excluded |
| `docker compose --profile colocated config` | resolves |
| `docker compose --profile chunked config` | resolves |
| `docker compose --profile disagg config` | resolves |
| `docker compose --profile disagg-tier config` | resolves |
| ruff/mypy | skip (no Python code in this plan) |

## Commits

| Hash | Subject |
|------|---------|
| ba76f5b | feat(01-03): UCX-only NIXL connector JSONs (producer + consumer) |
| 9e7a318 | feat(01-03): LMCache over NIXL connector JSONs + tier YAMLs |
| 106db2b | feat(01-03): single docker-compose with 4 topology profiles |

## Deviations from Plan

**None.** Plan executed exactly as written.

One incidental note: a sibling agent (01-04-proxy) inadvertently landed the LMCache commit (76ec991) on its branch before I switched back to phase-1/01-03-compose and cherry-picked it as 9e7a318 onto my branch. Both branches now contain identical LMCache content; the original 76ec991 on phase-1/01-04-proxy remains as a duplicate (not harmful; 01-04 will reconcile).

## Acceptance Criteria

| ID | Criterion | Status |
|----|-----------|--------|
| TOPO-01 | Colocated serves OpenAI-compat via `make up-colocated` | wired (profile colocated, port 18000) |
| TOPO-02 | Chunked-prefill via `make up-chunked` | wired (profile chunked, --enable-chunked-prefill) |
| TOPO-03 | P/D disagg via `make up-disagg` + UCX-only NIXL | wired (profile disagg, backends=UCX) |
| TOPO-04 | Disagg + LMCache via `make up-disagg-tier` | wired (profile disagg-tier, LMCacheConnectorV1) |
| TOPO-07 | 4 topologies share OpenAI-compat schema + /metrics | wired (all use goodputlab-model) |
| REPRO-01 | docker-compose for all 4 committed | done (single file, 4 profiles) |

## Deferred (out of scope this plan)

- `make up-*` Makefile targets — added in 01-07 README+tests
- `scripts/disagg_proxy.py` implementation — plan 01-04
- `scripts/health.sh` full implementation — plan 01-06
- Provision.sh + RunPod boot — plan 01-02
- LMCache prewarming automation — Phase 5
- EAGLE-3 speculative decoding — Phase 6
- Grafana dashboard JSON — Phase 8 OBS

## Threat Flags

None new (P1 LIBFABRIC mitigation encoded as exclusion; T-01-03-S served-model-name + T-01-03-D 0.45 split both mitigated).

## Files Touched

7 created (4 JSON, 2 YAML, 1 docker-compose.yml). Total ~310 LOC.