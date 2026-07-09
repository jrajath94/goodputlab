# GoodputLab

[![CI](https://github.com/jrajath94/goodputlab/actions/workflows/ci.yml/badge.svg)](https://github.com/jrajath94/goodputlab/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jrajath94/goodputlab/branch/main/graph/badge.svg)](https://codecov.io/gh/jrajath94/goodputlab)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

SLO-aware control plane for disaggregated prefill and decode LLM serving, with cache-aware routing, admission control, and autoscaling — measured end-to-end on a RunPod H100 NVL pod.

## Status

Phase 1 of 8 (Topologies) is in flight. All four docker-compose profiles
(`colocated`, `chunked`, `disagg`, `disagg-tier`) wire up to a single
vLLM v0.11.x image with a shared OpenAI-compatible HTTP contract. Phase 2
(load generation + metric reconciliation) is the next gate.

## Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11 | `>=3.11` per pyproject.toml |
| Docker Engine | 24.x or newer | Compose v2 plugin (`docker compose ...`) |
| Docker Compose | v2 (`docker compose`) | Single `docker-compose.yml`, profile-based dispatch |
| NVIDIA Container Toolkit | runtime = `nvidia` | Required for GPU passthrough to vLLM containers |
| RunPod pod | `t3son251d5gcvg` | 1x H100 NVL (94 GB HBM3), volume at `/workspace` |

The pod lives in `.planning/RUNPOD.md`. It is shipped STOPPED; bring it
online with `mcp__runpod__start-pod(podId="t3son251d5gcvg")` before
running `make provision` and any `make up-*` target.

## Quickstart

All commands assume you are at the repository root.

```bash
# 1. Install dev dependencies (pytest, ruff, mypy).
make install-dev

# 2. Provision the pod: system packages, venv, HF model cache.
#    Implemented in 01-02 (ships with `provision.sh`).
make provision

# 3. Bring up a topology profile. Pick one. Profiles are mutually
#    exclusive at runtime; tear down between switches with `make down`.
make up-colocated     # vLLM single-process baseline (port 18000)
make up-chunked       # vLLM + chunked prefill, no kv-transfer (port 18001)
make up-disagg        # disagg proxy + prefill + decode, NIXL UCX (port 19100)
make up-disagg-tier   # same as disagg + LMCache tiering (port 19200)

# 4. Confirm health. Runs sentinel-token check + /v1/models probe.
make health
```

End-to-end budget from `make provision` cold start to first `make
up-colocated` healthy is gated at 20 minutes (TOPO-06).

## Topology Table

Every profile serves the same model id `goodputlab-model` and exposes
`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, and
`/metrics` (D-05 Common Endpoint Contract). Ports below are the
host-side external ports published by `docker-compose.yml`.

| Profile | Make target | Compose services | External port | Endpoint base URL |
|---------|-------------|------------------|--------------:|-------------------|
| `colocated` | `make up-colocated` | `vllm-colocated` | 18000 | `http://localhost:18000/v1` |
| `chunked` | `make up-chunked` | `vllm-chunked` | 18001 | `http://localhost:18001/v1` |
| `disagg` | `make up-disagg` | `vllm-disagg-prefill`, `vllm-disagg-decode`, `disagg-proxy` | 19100 | `http://localhost:19100/v1` |
| `disagg-tier` | `make up-disagg-tier` | `vllm-disagg-tier-prefill`, `vllm-disagg-tier-decode`, `disagg-tier-proxy` | 19200 | `http://localhost:19200/v1` |

Direct vLLM ports (8100 prefill, 8200 decode, 8000 single-process) are
exposed only when the matching compose override is in use; the tables
above expose the public, profile-fronted ports.

## Measured Results (Phase 1)

All rows in this table are intentionally placeholders. No benchmark has
been run yet on `t3son251d5gcvg`. Each value will be filled in by the
matching plan and verified against the artifacts on disk before being
promoted out of `[NOT YET MEASURED]`.

| Metric | colocated | chunked | disagg | disagg-tier |
|--------|-----------|---------|--------|-------------|
| Cold-start time (s) | `[NOT YET MEASURED]` | `[NOT YET MEASURED]` | `[NOT YET MEASURED]` | `[NOT YET MEASURED]` |
| Sentinel pass count | `[NOT YET MEASURED]` | `[NOT YET MEASURED]` | `[NOT YET MEASURED]` | `[NOT YET MEASURED]` |
| TTFT p50 (ms) | `[NOT YET MEASURED]` | `[NOT YET MEASURED]` | `[NOT YET MEASURED]` | `[NOT YET MEASURED]` |
| ITL p50 (ms) | `[NOT YET MEASURED]` | `[NOT YET MEASURED]` | `[NOT YET MEASURED]` | `[NOT YET MEASURED]` |
| Cost per 1M output tokens (USD) | `[NOT YET MEASURED]` | `[NOT YET MEASURED]` | `[NOT YET MEASURED]` | `[NOT YET MEASURED]` |

See `.planning/research/SUMMARY.md` for the measurement protocol that
will populate this table.

## Safety

Phase 1 carries two production-critical mitigations that gate every
later measurement:

### UCX-only NIXL backend

NIXL KV-transfer is pinned to the UCX transport. The alternate LIBFABRIC
backend is excluded across every pool because it can deliver
out-of-order, corrupted KV blocks under load (vllm #27055 — silent
garbage). The pin lives in `configs/kv_producer.json` and
`configs/kv_consumer.json` and is asserted by the compose healthchecks.
There is no opt-in toggle in Phase 1: if a profile needs KV transfer,
it gets UCX.

### Sentinel-token validation

A standalone sentinel CLI (`tests/sentinel.py`), a built-in step in
`scripts/health.sh`, and a periodic background daemon
(`scripts/sentinel_daemon.py`) all run the same known-prefix probe:
send a deterministic prompt, verify that the first-token distribution
on the decode side matches the prefill side to within a small epsilon.
Health, not raw transfer counters, is what determines pass/fail.

A "sentinel-token" failure blocks the phase and surfaces an operator
alert. Never disable the sentinel — under disagg it is the only signal
that the prefill → decode handoff is producing correct KV.

## Limitations

- **Single-node.** All four profiles run on the 1x H100 NVL pod. UCX
  `cuda_ipc` is used for GPU-direct transfer within the same box.
  Multi-node disagg is intentionally out of scope; it would need
  additional transport + routing work.
- **Single-tenant.** No request-level multiplexing, no auth, no per-tenant
  rate limiting. The control plane lands in Phase 3.
- **No benchmark claims.** Cold-start, TTFT, ITL, and cost numbers in
  this README are explicitly `[NOT YET MEASURED]` placeholders. Goodput
  curves, crossover points, and failure-mode postmortems live in the
  Phase 8 BENCH report.
- **Phase 1 schema only.** The Makefile + tests in this tree cover
  Phase 1 (Topologies) deliverables. Phase 2-8 plans extend the surface
  in subsequent PRs.

## Layout

```
goodputlab/
├── docker-compose.yml        # 4 profiles, UCX-only NIXL, shared endpoint contract
├── Makefile                  # make install-dev/provision/up-*/down/health
├── configs/                  # kv-transfer + LMCache YAML
├── scripts/                  # health.sh, disagg_proxy.py, sentinel daemon
├── tests/                    # sentinel.py, runtime + static tests
├── control/                  # router, admission, autoscaler (Phase 3+)
├── core/                     # metrics, telemetry, kv-tier (Phase 2+)
└── .planning/                # research, requirements, ROADMAP, STATE
```

## License

MIT
