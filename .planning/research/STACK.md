# Technology Stack

**Project:** GoodputLab
**Researched:** 2026-07-08
**Mode:** Ecosystem (LLM inference serving, P/D disagg)

## Recommended Stack

### Core Serving Engine

| Technology | Version | Purpose | Why |
|---|---|---|---|
| vLLM | v0.11.x (verified: v0.14.0rc2 on dev) | Primary inference; P/D disagg in v1 path; chunked-prefill native | Prod-grade; `NixlConnector` + EAGLE-3 + LMCacheConnector all in v1 |
| SGLang | latest stable | Fallback; RadixAttention shares KV in-process | Cleaner spec×disagg; use only if vLLM flags churn |
| TensorRT-LLM | NOT recommended | — | v1.x prod-stability regressions; brittle H100 spot builds (see DO-NOT-USE) |

### KV Transfer + Tiering

| Technology | Version | Purpose | Why |
|---|---|---|---|
| NIXL | 0.6.x | KV transport P→D | Industry std; UCX backend via `NixlConnector` |
| LMCache | 0.3.x | Shared KV tier (HBM→DRAM→disk) | First-class vLLM; `pd_role: sender/receiver`; UCX default |
| UCX | 1.15+ | NIXL transport (preferred) | Fewer silent-corruption reports than LIBFABRIC |

### Speculative Decoding

| Technology | Version | Purpose | Why |
|---|---|---|---|
| EAGLE-3 | HF pre-trained head | Spec decode on decode pool | `method:eagle3`; verified head: `nvidia/Llama-3.3-70B-Instruct-Eagle3`; Qwen3-32B head TBD |

### Control Plane (our code)

| Technology | Version | Purpose | Why |
|---|---|---|---|
| FastAPI | 0.115+ | Router HTTP front door (OpenAI-compat) | Async-native, Pydantic v2 tight coupling |
| Pydantic | v2.x | Request/response validation | Required by vLLM client types |
| uvicorn | 0.30+ | ASGI server | Standard FastAPI prod |
| prometheus-client | 0.20+ | `/metrics` scrape | Router + per-pool counters/gauges |
| Grafana | 10.x | Dashboard JSON committed | No build code, JSON only |
| pytest + pytest-asyncio | latest | ≥80% coverage `core/`,`control/` | Repo standard |
| ruff + mypy | latest | Lint + types | Repo standard (Py 3.11+) |

### Hardware / Runtime

| Component | Choice | Why |
|---|---|---|
| GPU | 2-4× H100 80GB SXM (spot) | FP8 32-70B single-node; NVLink for intra-node P↔D |
| Base image | `nvcr.io/nvidia/pytorch:24.10-py3` | CUDA 12.6 + NIXL wheels |
| vLLM image | `vllm/vllm-openai:v0.11.x` | Pinned tag, no `:latest` |
| Orchestration | docker-compose (W1-W7) → k8s stretch | Simpler single-node cause isolation |

## Install (Dockerfile pattern)

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.10-py3
RUN pip install --no-cache-dir \
    vllm==0.11.* lmcache==0.3.* nixl==0.6.* ucx-py==0.39.* \
    fastapi==0.115.* pydantic==2.* uvicorn[standard]==0.30.* \
    prometheus-client==0.20.*
# EAGLE-3 head pulled at runtime, not baked
```

## Version Table (HIGH confidence via Context7)

| Lib | Version / Syntax | Source |
|---|---|---|
| vLLM | v0.14.0rc2 dev; pin v0.11.x stable | `/websites/vllm_ai_en_stable` |
| `NixlConnector` | `kv_connector="NixlConnector"`, `kv_role="kv_both"`, `backends:[UCX,GDS]` | verified |
| EAGLE-3 | `--speculative-config '{"method":"eagle3","num_speculative_tokens":3,"model":"nvidia/Llama-3.3-70B-Instruct-Eagle3"}'` | verified |
| LMCache NIXL | `enable_pd:true`, `transfer_channel:"nixl"`, `pd_role:sender\|receiver`, `nixl_backends:[UCX]` | verified |
| LMCache UCX default | `nixl_backends` default = `UCX` | verified |

## Top 5 — DO NOT USE

1. **TensorRT-LLM v1.x for prod bench** — brittle build matrix breaks `<20min` cold-to-serving SLO. Use only for engine-comparison runs.
2. **LIBFABRIC as NIXL backend (v0.6.x)** — silent-corruption on cross-node P→D (vllm #27055). Use UCX.
3. **Bare `hashlib` for prefix routing** — no rolling-window semantics; need content-defined 256-token block hash per worker (RTR-02).
4. **Per-request autoscaler role flips** — thrash; 120s minimum dwell (AUTO-03).
5. **vLLM `:latest` docker tag** — non-reproducible benches; pin to `v0.11.x`.

## Confidence

| Area | Level | Reason |
|---|---|---|
| vLLM disagg flags | HIGH | Context7 official, exact syntax |
| LMCache NIXL config | HIGH | Context7 official |
| EAGLE-3 invocation | HIGH | Context7 official example |
| Qwen3-32B EAGLE-3 HF head | MEDIUM | 70B head verified; Qwen3-32B not yet — verify Phase 6 day 1 |
| NIXL/UCX patch versions | MEDIUM | pin to latest patch on Phase 1 day 1 |
| Spot H100 $3-6/hr | LOW | market-dep; verify on rental day |

## Gaps to validate Phase 1

- `nixl==` wheel availability on `nvcr.io/nvidia/pytorch:24.10-py3`
- Qwen3-32B EAGLE-3 head (search HF; fall back to Llama-3.3-70B-Eagle3)
- Single-engine LMCache + NIXL disagg path (may need both connectors)
- K8s RDMA device-plugin (stretch, flag only)
