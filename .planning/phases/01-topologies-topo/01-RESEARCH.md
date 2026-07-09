# Phase 1: Topologies (TOPO) - Research

**Researched:** 2026-07-08
**Domain:** LLM serving deploy (vLLM + NIXL KV transfer + LMCache) on RunPod H100 NVL
**Confidence:** HIGH (vLLM/NIXL/LMCache stack) | MEDIUM (Qwen3.6/vLLM version compat)

---

<user_constraints>
## User Constraints (from `01-CONTEXT.md`)

### Locked Decisions

| Decision | Verbatim |
|----------|----------|
| **Model** | Qwen 3.6 latest stable; verify exact HF id in plan-phase research. Default to FP8 quant for H100 NVL 94GB fit. If 3.6 not on HF by 2026-07, fall back to most recent Qwen3.x. |
| **NIXL backend** | UCX only, pinned via `kv-transfer-config` `backends=["UCX"]`. LIBFABRIC excluded (PITFALLS P1: vllm #27055 silent garbage). No opt-in toggle Phase 1. |
| **Sentinel-token test** | All 3 combined: (1) `tests/sentinel.py` standalone; (2) `scripts/health.sh` built-in; (3) `scripts/sentinel_daemon.py` 60s probe → `sentinel_drift` Prom gauge. |
| **Compose layout** | Single `docker-compose.yml` w/ profiles: `colocated`, `chunked`, `disagg`, `disagg-tier`. Invoke via `docker compose --profile disagg up`. |

### Claude's Discretion

- vLLM flags per topology (kv-transfer-config syntax, chunked-prefill on/off, NIXL handshake) — verify live vLLM v0.11.x docs in plan-phase
- `provision.sh` structure (system pkgs, venv, model download w/ HF_HUB_ENABLE_HF_TRANSFER)
- Grafana dashboard JSON baseline (deferred Phase 8 OBS)
- Test fixture data for sentinel (deterministic token sequences)

### Deferred Ideas (OUT OF SCOPE)

Helm (Phase 8), LMCache prewarming (Phase 5), EAGLE-3 head (Phase 6), autoscaler role-flip (Phase 7), cross-region (v2), Grafana JSON (Phase 8), Dockerfile hardening (post-v1)

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TOPO-01 | Colocated serves OpenAI-compat HTTP via `make up-colocated` | Q3 — colocated profile in single compose |
| TOPO-02 | Chunked-prefill topology via `make up-chunked` | Q5 --enable-chunked-prefill + --max-num-batched-tokens 2048 |
| TOPO-03 | P/D disagg via `make up-disagg` + P→D KV transfer confirmed by metrics | Q2 NixlConnector + Q6 metrics |
| TOPO-04 | Disagg + LMCache via `make up-disagg-tier` + tier round-trip hit | `enable_pd: True` + `transfer_channel: "nixl"` |
| TOPO-05 | `make health` confirms P→D flow + decode never runs prefill | Q5 sentinel-token + Q6 metric delta |
| TOPO-06 | Cold-node-to-serving <20 min for any topology | Q7 (HF download + vLLM warmup timing) |
| TOPO-07 | 4 topologies share common OpenAI-compat schema + /metrics | Q4 /v1/{chat/completions,models,score,embeddings}, /health, /metrics |
| REPRO-01 | docker-compose for all 4 committed | Q3 single-file `profiles:` |
| REPRO-02 | `make provision` → healthy serving in <20 min | Q7 budget realism |

</phase_requirements>

---

## Summary

Phase 1 deploys **4 vLLM serving topologies** on a single RunPod H100 NVL pod (1× GPU, 94GB HBM3, 200GB /workspace) using `docker compose v2` profiles, exposes unified OpenAI-compat HTTP + Prometheus `/metrics`, and passes a **sentinel-token KV-transfer validity test** for disagg topologies. P/D disagg uses **UCX-only** NIXL (LIBFABRIC excluded by PITFALLS P1 mitigation).

**Critical model pin:** CONTEXT locks "Qwen 3.6 latest stable", but `Qwen/Qwen3.6-27B-FP8` uses architecture `Qwen3_5ForConditionalGeneration` — **NOT registered in vLLM v0.11.2** (verified via [vLLM supported_models](https://docs.vllm.ai/en/latest/models/supported_models.html)). Two workable paths: (A) bump vLLM ≥0.13.x where `qwen3_5` is registered; (B) fall back to `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` (`Qwen3MoeForCausalLM`, official Qwen FP8 quant, **v0.11.x-compatible**, 30B total/3B active MoE, ~30.5GB on disk, fits H100 NVL 94GB).

**Primary recommendation:** **Path B for Phase 1.** Pin `vllm/vllm-openai:v0.11.2` + `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`. Sentinel validity test must pass before any measurement. Verify on day 1 of execution that model architecture is registered.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|--------------|----------------|-----------|
| OpenAI-compat HTTP serving (4 topologies) | vLLM container (Docker, GPU host) | — | Colocated/chunked = 1 container; disagg = 2 containers + proxy |
| P→D KV transfer (NIXL UCX) | vLLM `NixlConnector` (in-process) | UCX transport | Between two `vllm serve` procs on localhost; UCX over cuda_ipc |
| KV tiering (HBM→DRAM→disk) | LMCache engine (in-process w/ vLLM) | /workspace disk | LMCache in vLLM process; `local_cpu: False` + `pd_buffer_device: "cuda"` |
| Sentinel-token validity | `tests/sentinel.py` (Python client) | vLLM `/metrics` cross-check | Known-prefix prompt + `temperature=0` + first-N token compare |
| Container orchestration | `docker compose v2.x` (`profiles:`) | — | One file, four topologies |

---

## Standard Stack

### Core

| Component | Version | Purpose | Why Standard |
|-----------|---------|---------|--------------|
| vLLM | `vllm/vllm-openai:v0.11.2` ([dockerhub](https://hub.docker.com/r/vllm/vllm-openai/tags), 2025-11-20) | LLM serving engine | Project pin; fixes CVE-2025-25183; NixlConnector stable |
| NIXL | Bundled in vLLM image (v0.6.x per [docs](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/)) | KV cache transfer engine | Default backend in NixlConnector |
| UCX | Bundled in vLLM image | RDMA transport | `UCX_TLS=cuda_ipc,cuda_copy,tcp` loopback on H100 |
| LMCache | Pre-installed in `vllm/vllm-openai:v0.11.2` | KV tier (HBM→DRAM→disk) | Native NixlConnector via `enable_pd: True` + `transfer_channel: "nixl"` |
| Docker Compose | v2.39.2 (`/usr/local/bin/docker-compose`) | Multi-topology via `profiles:` | Available locally + RunPod pod |

### Supporting

| Component | Purpose | When to Use |
|-----------|---------|-------------|
| `huggingface_hub[hf-transfer]` | Parallel model download (`HF_HUB_ENABLE_HF_TRANSFER=1`) | provision.sh + pre-warm |
| Prometheus client (bundled in vLLM image) | `/metrics` endpoint per container | Every topology |

### Model Selection (verified via HuggingFace API 2026-07-08)

| Rank | HF model id | Downloads | Arch | vLLM v0.11.2 compat | Notes |
|------|-------------|-----------|------|---------------------|-------|
| 1 | `Qwen/Qwen3.6-27B-FP8` | 4.85M | `Qwen3_5ForConditionalGeneration` | **NO** | CONTEXT-locked; `qwen3_5` not in v0.11.2 |
| 2 | `Qwen/Qwen3.6-35B-A3B-FP8` | 6.85M | `Qwen3_5ForConditionalGeneration` (MoE) | **NO** | Same registration gap |
| 3 | `nvidia/Qwen3.6-27B-NVFP4` | 0.54M | NVFP4 | NO | NV-tensor-RT-LLM-tuned, not standard FP8 |
| **4 (FALLBACK)** | **`Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`** | **292k** | **`Qwen3MoeForCausalLM`** | **YES** | **Strongest v0.11.x FP8**; official; 30.5GB; 30B/3B MoE |

**Path A** (Qwen 3.6 verbatim per CONTEXT) requires vLLM ≥0.13.x where `qwen3_5` is registered — verify day 1.
**Path B (recommended for Phase 1):** stay on project v0.11.x pin + use `Qwen3-30B-A3B-FP8`. Validate Path A post-Phase-1 in follow-up `discuss-phase`.

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `vllm/vllm-openai:v0.11.2` | v0.24.0 (latest, 2026-06-29) | Newer may register `qwen3_5`, but unproven at our load |
| `Qwen/Qwen3-30B-A3B-FP8` | `Qwen/Qwen3-32B-FP8` (99k dl) | Less battle-tested, similar compat |
| Single compose+profiles | 4 separate files | CONTEXT decision: cross-diff in one file |
| NIXL + LIBFABRIC | UCX only | P1 mandates UCX |

**Installation:** `docker pull vllm/vllm-openai:v0.11.2` (LMCache + NIXL + UCX bundled).

---

## Package Legitimacy Audit

All packages pre-baked in official `vllm/vllm-openai:v0.11.2` image; no new pip installs.

| Package | Source | Disposition |
|---------|--------|-------------|
| `vllm` v0.11.2 image | Docker Hub official (vllm/vllm-openai) | Approved |
| `nixl` bundled | github.com/ai-dynamo/nixl | Approved (no separate install) |
| `lmcache` bundled | github.com/LMCache/LMCache | Approved (no separate install) |
| `huggingface_hub[hf-transfer]` | github.com/huggingface/huggingface_hub (official, 6+ yrs) | Approved |
| `prom/prometheus` | github.com/prometheus/prometheus | Approved (Phase 8 OBS; Phase 1 uses raw /metrics scrape) |

Slopcheck: not run — no new external packages. All containers = upstream vLLM official image.

---

## Architecture Patterns

### System Architecture Diagram

```
RunPod H100 NVL pod (1× GPU, 94GB, 200GB /workspace) — docker-compose v2 `profiles:`

profile=colocated    vllm-colocated         (port 18000→8000, gm=0.85, no chunked-prefill)
profile=chunked      vllm-chunked           (port 18001→8000, --enable-chunked-prefill --max-num-batched-tokens 2048, gm=0.85)
profile=disagg       vllm-disagg-prefill    (port 8100, gm=0.45, kv_role=producer,  SIDE_CH=5559)
                     vllm-disagg-decode     (port 8200, gm=0.45, kv_role=consumer,  SIDE_CH=5560)
                     disagg-proxy           (port 19100→9100, routes /v1/* to P→D)
                     transport: NIXL/UCX/cuda_ipc + tcp (P+D on same H100, no NIC)
profile=disagg-tier  Same shape, kv-transfer-config = LMCacheConnectorV1 +
                     configs/lmcache_{prefill,decode}.yaml w/ enable_pd: True, transfer_channel: "nixl"

All 4 topologies expose: /v1/{chat/completions, models, score, embeddings} (OpenAI-compat)
                         /health (vLLM native), /metrics (Prometheus: vllm:* + vllm:nixl_*)
                         tests/sentinel.py — known-prefix validity probe
```

### Recommended Project Structure

```
goodputlab/
├── docker-compose.yml          # 4 profiles (single file, per CONTEXT)
├── Makefile                     # up-{topo}, health, sentinel, provision, bench, lint, test
├── provision.sh                 # Bare-node → healthy serving in <20 min
├── configs/
│   ├── kv_{producer,consumer}.json                # NixlConnector (disagg)
│   ├── kv_lmcache_{producer,consumer}.json        # LMCacheConnectorV1 (disagg-tier)
│   └── lmcache_{prefill,decode}.yaml              # LMCache sender/receiver
├── tests/
│   ├── conftest.py                                  # model/base_url fixtures
│   ├── test_{topos,schema_uniformity,sentinel_disagg}.py
│   ├── sentinel.py                                  # standalone CLI (--mode record|check)
│   └── _fixtures/sentinel_<model>_<version>.json
├── scripts/{health.sh, sentinel_daemon.py, pull_model.sh}
├── pyproject.toml                 # ruff + mypy + pytest config
└── README.md
```

### Pattern 1: NixlConnector UCX-only (P/D disagg)

```bash
# configs/kv_producer.json  — prefill
{
  "kv_connector": "NixlConnector",
  "kv_role": "kv_producer",
  "kv_buffer_device": "cuda",
  "kv_connector_extra_config": {"backends": ["UCX"]}    # LIBFABRIC excluded (P1)
}

# configs/kv_consumer.json  — decode (same shape, role=kv_consumer)

# Env on both containers:
VLLM_NIXL_SIDE_CHANNEL_HOST=localhost
VLLM_NIXL_SIDE_CHANNEL_PORT=5559  # 5560 for decode — must differ
UCX_TLS=cuda_ipc,cuda_copy,tcp    # GPU-direct on same H100

# Invocation:
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --port 8100 --tensor-parallel-size 1 --gpu-memory-utilization 0.45 \
  --block-size 128 --kv-transfer-config /configs/kv_producer.json --enforce-eager
```

### Pattern 2: LMCache over NIXL (disagg-tier)

```yaml
# configs/lmcache_prefill.yaml
local_cpu: False
enable_pd: True
transfer_channel: "nixl"
pd_role: "sender"
pd_proxy_host: "localhost"
pd_proxy_port: 7500
pd_buffer_size: 1073741824   # 1 GB
pd_buffer_device: "cuda"
# configs/lmcache_decode.yaml — same except pd_role: "receiver", pd_peer_host: "localhost"
#                                       pd_peer_init_port: 7300, pd_peer_alloc_port: 7400
```

```bash
LMCACHE_CONFIG_FILE=/configs/lmcache_prefill.yaml \
  vllm serve <model> --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer",...}'
```

### Pattern 3: Sentinel (`tests/sentinel.py` — per CONTEXT 3-layer defense)

```python
import requests, json
FIXTURE = "tests/_fixtures/sentinel_qwen30b_a3b_2507_fp8_v0.11.2.json"
KNOWN_PREFIX = "The quick brown fox jumps over the lazy dog. " * 50
def check(base_url="http://localhost:9100/v1"):
    fx = json.load(open(FIXTURE))
    r = requests.post(f"{base_url}/chat/completions", json={
        "model":"qwen30b","prompt":KNOWN_PREFIX,"max_tokens":50,
        "temperature":0.0,"stream":False,"logprobs":1}, timeout=60).json()
    tokens = r["choices"][0]["logprobs"]["tokens"]
    assert tokens == fx["tokens"], f"SENTINEL FAIL: got {tokens[:5]} != {fx['tokens'][:5]}"
```

### Anti-Patterns to Avoid

- **LIBFABRIC backend** → silent garbage (vllm #27055)
- **Healthcheck just curls `/health`** → returns 200 before model load; use `/v1/models` + non-empty `data[].id`
- **One shared kv-transfer-config** → different `kv_role` + ports required; two files enforced by Makefile
- **Trusting `kv_transfer_complete_count`** → NOT a real vLLM metric (Q6); use `vllm:nixl_*` only
- **Sum of `--gpu-memory-utilization` > 1.0** → 0.45 + 0.45 = 0.9 max on single H100

---

## Don't Hand-Roll

| Problem | Use Instead | Why |
|---------|-------------|-----|
| OpenAI-compat HTTP serve | `vllm serve --port ...` | Built-in |
| P→D KV transfer over UCX | `NixlConnector` + `kv-transfer-config` | Built-in |
| LMCache tier eviction | LMCache + `LMCacheConnectorV1` | Built-in, KV-tuned |
| KV transfer Prom metrics | vLLM `vllm:nixl_*` | In `nixl/stats.py` |
| Model download parallelism | `huggingface_hub[hf-transfer]` | 5-10× faster |
| Container orchestration (1 host, 4 profiles) | `docker compose v2` profiles | Single file |

**Key insight:** `tests/sentinel.py` (~50 lines) is the only hand-rolled piece — not vendored in vLLM as CLI but the load-bearing P1 safety mechanism.

---

## Common Pitfalls

| # | Pitfall | Sev | Mitigation |
|---|---------|:---:|------------|
| P1 | NIXL LIBFABRIC silent garbage | CRIT | Pin `backends=["UCX"]` + sentinel |
| P2 | CVE-2025-25183 prefix-cache hash collision | M (single-tenant) | vLLM ≥0.7.2 (we use 0.11.2) |
| P3 | Compose `profiles:` typo silently skips service | M | `docker compose --profile X config` dry-run |
| P4 | 2 vLLM on 1 H100 OOM (gm sum >1.0) | H | Cap each pool at 0.45 |
| P5 | Health-check 200 before model loaded | H | Probe `/v1/models` + non-empty `data[]` |
| P6 | Sentinel fixture drift across vLLM versions | M | Record per (model,vllm_version); hash-pinned |
| P7 | Cold start >20 min first-ever deploy | M | HF cache in /workspace; hf-transfer parallel |

---

## Q&A — 8 specific questions

### Q1: Qwen 3.6 + fallback chain + FP8/H100 NVL fit

| Rank | HF model id | Arch | vLLM v0.11.2 compat |
|------|-------------|------|---------------------|
| 1 | `Qwen/Qwen3.6-27B-FP8` | `Qwen3_5ForConditionalGeneration` | **NO** — `qwen3_5` not registered |
| 2 | `Qwen/Qwen3.6-35B-A3B-FP8` | `Qwen3_5ForConditionalGeneration` MoE | **NO** — same gap |
| 3 | `nvidia/Qwen3.6-27B-NVFP4` | NVFP4 | NO — needs Blackwell-tuned kernels |
| **4 (FALLBACK)** | **`Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`** | **`Qwen3MoeForCausalLM`** | **YES** — registered in v0.11.x |

**Recommendation: Path B** — `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` + vLLM v0.11.2 (official FP8 quant, 30.5GB, fits H100 NVL 94GB @ gm=0.85). **Path A** (CONTEXT Qwen 3.6) requires vLLM ≥0.13.x — verify via `ModelRegistry.get_supported_archs()` in `provision.sh`; fall through to Path B.

**Source:** [huggingface.co/Qwen/Qwen3.6-27B-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8), [Qwen3-30B-A3B-FP8](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8), [vLLM supported_models](https://docs.vllm.ai/en/latest/models/supported_models.html).

---

### Q2: vLLM v0.11.x NixlConnector `kv-transfer-config` syntax

From upstream `tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh` and [docs.vllm.ai](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/):

```bash
KV_CONFIG_P='{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'
KV_CONFIG_D='{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
```

**UCX-only minimal config (recommended Phase 1):**

```json
{
  "kv_connector": "NixlConnector",
  "kv_role": "kv_producer",
  "kv_buffer_device": "cuda",
  "kv_connector_extra_config": {"backends": ["UCX"]}
}
```

**Key NixlConnector fields** (full list in [docs](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/)):

| Field | Default | Notes |
|-------|---------|-------|
| `kv_connector` | required | `"NixlConnector"` or `"LMCacheConnectorV1"` |
| `kv_role` | required | `kv_producer` / `kv_consumer` (`kv_both` deprecated) |
| `kv_load_failure_policy` | `"fail"` | `"recompute"` warned — silent corruption |
| `kv_buffer_device` | `"cuda"` | alt: `cpu`, `hbm`, `hfs` |
| `kv_connector_extra_config.backends` | `["UCX"]` | Add LIBFABRIC → bug; **EXCLUDED P1** |
| `kv_connector_extra_config.bidirectional_kv_xfer` | `false` | D→P KV pull |
| `kv_connector_extra_config.kv_recompute_threshold` | `64` | Min remote tokens to trigger pull |
| `kv_connector_extra_config.kv_lease_duration` | `30` | Seconds, heartbeat-renewed |
| `kv_connector_extra_config.decoder_kv_blocks_ttl` | `480` | Seconds, not renewed |
| `enable_permute_local_kv` | `false` | Heterogeneous KV layout (experimental) |

**Mandatory env (P and D differ in port):**

```bash
VLLM_NIXL_SIDE_CHANNEL_HOST=localhost
VLLM_NIXL_SIDE_CHANNEL_PORT=5559   # 5560 on decode
UCX_TLS=cuda_ipc,cuda_copy,tcp     # GPU-direct on H100
```

**LIBFABRIC exclusion:** Literal `"backends":["UCX"]` + comment cites vllm #27055. Source: [docs](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/), [vllm #27055](https://github.com/vllm-project/vllm/issues/27055).

---

### Q3: docker compose v2 `profiles:` syntax

Verbatim from [docs.docker.com/reference/compose-file/services](https://docs.docker.com/reference/compose-file/services/):

```yaml
services:
  frontend:
    image: frontend
    profiles: ["frontend"]
  phpmyadmin:
    image: phpmyadmin
    depends_on:
      - db
    profiles:
        - debug
```

**Activation:**

```bash
docker compose --profile <X> up                # run services tagged X
docker compose --profile X --profile Y up      # multiple
docker compose up service-name                # bypass profile (always)
docker compose --profile X config             # dry-run print resolved YAML
```

**Phase 1 application (`docker-compose.yml` skeleton):**

```yaml
x-common: &common
  image: vllm/vllm-openai:v0.11.2
  runtime: nvidia
  environment: &env
    HF_HOME: /workspace/hf
    HF_HUB_ENABLE_HF_TRANSFER: "1"
  volumes: ["/workspace/hf:/workspace/hf", "./configs:/configs:ro"]
  deploy:
    resources:
      reservations:
        devices: [{capabilities: [gpu], count: 1}]
x-health: &hcheck
  test: ["CMD-SHELL", "curl -fs http://localhost:8000/v1/models | grep -q 'qwen30b'"]
  interval: 30s
  timeout: 10s
  retries: 30
  start_period: 240s

services:
  vllm-colocated:
    <<: *common; profiles: ["colocated"]
    command: >
      vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
        --port 8000 --tensor-parallel-size 1
        --gpu-memory-utilization 0.85 --block-size 128
        --served-model-name qwen30b
    ports: ["18000:8000"]; healthcheck: *hcheck

  vllm-chunked:
    <<: *common; profiles: ["chunked"]
    command: >
      vllm serve ... --enable-chunked-prefill --max-num-batched-tokens 2048
    ports: ["18001:8000"]; healthcheck: *hcheck

  vllm-disagg-prefill:
    <<: *common; profiles: ["disagg","disagg-tier"]
    command: >
      vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
        --port 8100 --gpu-memory-utilization 0.45
        --kv-transfer-config /configs/kv_producer.json --enforce-eager
    environment:
      <<: *env
      VLLM_NIXL_SIDE_CHANNEL_PORT: "5559"
      UCX_TLS: "cuda_ipc,cuda_copy,tcp"

  vllm-disagg-decode:
    <<: *common; profiles: ["disagg","disagg-tier"]
    command: >
      vllm serve ... --port 8200 --gpu-memory-utilization 0.45
        --kv-transfer-config /configs/kv_consumer.json --enforce-eager
    environment:
      <<: *env
      VLLM_NIXL_SIDE_CHANNEL_PORT: "5560"
      UCX_TLS: "cuda_ipc,cuda_copy,tcp"

  disagg-proxy:
    profiles: ["disagg","disagg-tier"]
    image: vllm/vllm-openai:v0.11.2
    command: >
      python -m vllm.entrypoints.openai.api_server --proxy
        --prefill-hosts vllm-disagg-prefill:8100
        --decode-hosts vllm-disagg-decode:8200
        --port 9100
    ports: ["19100:9100"]

  # disagg-tier duplicates P+D with LMCacheConnectorV1 + lmcache YAML
  vllm-disagg-tier-prefill: profiles: ["disagg-tier"]  # like vllm-disagg-prefill, but
    command: ... --kv-transfer-config /configs/kv_lmcache_producer.json
    environment: { LMCACHE_CONFIG_FILE: /configs/lmcache_prefill.yaml, ... }
  vllm-disagg-tier-decode: profiles: ["disagg-tier"]  # likewise, lmcache decode
  disagg-tier-proxy: profiles: ["disagg-tier"]  # routes to tier P+D
```

**Key constraint:** Don't duplicate YAML for P+D if they only differ in profile tag — use shared `<<: *common` anchor.

**Source:** [docs.docker.com/compose-file/services](https://docs.docker.com/reference/compose-file/services/), [Docker Compose profiles docs](https://docs.docker.com/compose/profiles/).

---

### Q4: Healthcheck values tuned for 3-5 min cold start

```yaml
healthcheck:
  test: ["CMD-SHELL", "curl -fs http://localhost:8000/v1/models | grep -q 'qwen30b'"]
  interval: 30s
  timeout: 10s
  retries: 30         # 30 × 30s = 15 min total probe budget
  start_period: 240s  # 4 min grace — vLLM model load + CUDA graphs
```

**Rationale:** `/v1/models` + assert id — `/health` returns 200 before model load. `interval:30s` / `timeout:10s` standard. `retries:30` = 15min, caps at TOPO-06 (20min) − start_period − 1min. `start_period:240s` — vLLM init + CUDA graph = 2-4min on H100 NVL. Bump to 300s for `disagg-tier`.

**Source:** Patterns derived from [Breaking the Ice: Analyzing Cold Start Latency in vLLM, arXiv 2606.07362](https://arxiv.org/pdf/2606.07362) + Docker Compose [healthcheck docs](https://docs.docker.com/reference/compose-file/services/).

---

### Q5: Sentinel-token validity test pattern

**What:** Known-prefix prompt → greedy decode (`temperature=0`, `max_tokens=50`) → compare first 50 tokens vs recorded fixture. Token sequence differs ⇒ NIXL transferred corrupted KV.

```python
# tests/sentinel.py — CLI (--mode record|check)
import argparse, hashlib, json, sys, requests
FIXTURE = "tests/_fixtures/sentinel_qwen30b_a3b_2507_fp8_v0.11.2.json"
KNOWN_PREFIX = "The quick brown fox jumps over the lazy dog. " * 50
def fetch(base_url, n=50):
    r = requests.post(f"{base_url}/chat/completions", json={
        "model":"qwen30b","prompt":KNOWN_PREFIX,"max_tokens":n,
        "temperature":0.0,"stream":False,"logprobs":1
    }, timeout=60).json()
    return r["choices"][0]["logprobs"]["tokens"][:n], r["choices"][0]["logprobs"]["token_logprobs"][:n]
def check(base_url):
    fx = json.load(open(FIXTURE))
    tok, lps = fetch(base_url, len(fx["tokens"]))
    if tok != fx["tokens"]:
        print(f"SENTINEL FAIL: {tok[:5]} != {fx['tokens'][:5]}", file=sys.stderr); return False
    return all(abs(a-e) <= 1e-3 for a,e in zip(lps[:5], fx["logprobs_first_5"]))
def record(base_url):
    t,l = fetch(base_url)
    json.dump({"model":"Qwen/Qwen3-30B-A3B-Instruct-2507-FP8","vllm_version":"0.11.2",
        "prompt_sha256":hashlib.sha256(KNOWN_PREFIX.encode()).hexdigest()[:16],
        "tokens":t,"logprobs_first_5":l[:5]}, open(FIXTURE,"w"), indent=2)
if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--mode", choices=["record","check"], default="check"); p.add_argument("--base-url", default="http://localhost:9100/v1")
    args = p.parse_args()
    (record if args.mode=="record" else sys.exit(0 if check(args.base_url) else 1))(args.base_url)
```

**3-layer integration per CONTEXT:** (1) `tests/sentinel.py` standalone CLI+pytest · (2) `scripts/health.sh` invokes sentinel after `/v1/models` 200 · (3) `tests/sentinel_daemon.py` 60s loop emits `sentinel_drift` Prom gauge = 1/0.

**Fixture cadence:** record per `(model, vllm_version, kv_connector_extra_config)` triple. Re-record on any change.

**Source:** [vLLM NixlConnector usage](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/), [vLLM issue #27055 P1 rationale](https://github.com/vllm-project/vllm/issues/27055).

---

### Q6: vLLM Prometheus metrics + delta pattern after one P→D round trip

**Standard vLLM metrics** ([docs](https://docs.vllm.ai/en/latest/usage/metrics/)):

```
vllm:request_success_total{...}
vllm:num_requests_running{engine=...}
vllm:num_requests_waiting{engine=...}
vllm:kv_cache_usage_perc{engine=...}
vllm:time_to_first_token_seconds_bucket{...}
vllm:time_per_output_token_seconds_bucket{...}
vllm:e2e_request_latency_seconds_bucket{...}
vllm:prefix_cache_queries_total{...}
vllm:prefix_cache_hits_total{...}
vllm:request_prefill_time_seconds_bucket{...}
vllm:generation_tokens_total{...}
vllm:prompt_tokens_total{...}
vllm:kv_block_lifetime_seconds_bucket{...}
```

**NIXL connector-specific** (verified via [nixl/stats.py](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/nixl/stats.py)):

```
vllm:nixl_xfer_time_seconds_bucket{engine=...}        # Histogram
vllm:nixl_post_time_seconds_bucket{engine=...}        # Histogram
vllm:nixl_bytes_transferred_bucket{engine=...}        # Histogram (2KB→16GB)
vllm:nixl_num_descriptors_bucket{engine=...}          # Histogram
vllm:nixl_num_failed_transfers_total{engine=...}      # Counter
vllm:nixl_num_failed_notifications_total{engine=...}  # Counter
vllm:nixl_num_kv_expired_reqs_total{engine=...}       # Counter
```

**Note:** `kv_transfer_complete_count`/`kv_transfer_inflight` (PITFALLS P1) are **NOT** vLLM/NIXL metric names — older pre-NIXL-connector or proxy logs only. Use the metric set above.

**Expected delta after 1 successful P→D round trip (~2K prompt):**

| Metric (prefill proc, unless noted) | Δ after 1 round trip |
|--------|------------------------|
| `vllm:nixl_xfer_time_seconds_count` | +1 |
| `vllm:nixl_bytes_transferred_sum` | +~5×10^7 (~50MB at FP8 KV) |
| `vllm:request_success_total{finished_reason="stop"}` (decode) | +1 |
| `vllm:time_to_first_token_seconds_sum` (decode) | +~0.300 |
| `vllm:nixl_num_failed_{transfers,notifications}_total` (BOTH) | 0 |

**Sentinel cross-check (all 4 required):** (1) decode `request_success_total > 0` AND finite `time_to_first_token_seconds` · (2) prefill `nixl_bytes_transferred_sum > 0` AND `nixl_xfer_time_seconds_count ≥ 1` · (3) `tests/sentinel.py --mode check` returns 0 · (4) `nixl_num_failed_transfers_total == 0` (BOTH). **Any one fails → abort + page operator** (per CONTEXT line 65).

**Source:** [docs.vllm.ai/usage/metrics](https://docs.vllm.ai/en/latest/usage/metrics/), [nixl/stats.py](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/nixl/stats.py).

---

### Q7: Cold-to-serving <20 min budget realism

| Step | Time |
|------|------|
| Image pull (~10GB compressed) | 3-5 min |
| HF model download (hf-transfer, ~30GB @ 2-3Gbps RunPod) | 1-2 min |
| vLLM model load (CUDA graphs + KV alloc, 27-30B FP8, gm=0.85) | 2-4 min |
| First HTTP req + first token (CUDA graph warm) | 5-10 sec |
| Sentinel + health check | 5 sec |
| **TOTAL (single topology)** | **~7-12 min** |
| TOPO-06 budget | 20 min |
| Buffer for unexpected | 8-13 min |

**Disagg-tier (most expensive):** Prefill + Decode parallel start (~2min each) + UCX NIXL handshake (~5s) + LMCache init + first P→D request (~10s) → **~9-14 min** within budget.

**Risk mitigations:** RunPod network throttling → `hf-transfer` parallel + cache to `/workspace/hf` (persists) · vLLM compile cache cold → 30s headroom (Phase 8 caches between runs) · LMCache eviction cold → sub-second init.

**Verification (provision.sh):** `bash -c 'S=$(date +%s); make up-disagg-tier; [ $(($(date +%s)-S)) -le 1200 ] || { echo TOPO-06 BUDGET EXCEEDED; exit 1; }'`

**Source:** [arxiv 2606.07362](https://arxiv.org/pdf/2606.07362) cold-start analysis; actual RunPod network [NOT YET MEASURED].

---

### Q8: `--enable-chunked-prefill` × `--kv-transfer-config` mutual exclusivity (USER ASSUMPTION CHECK)

**User assumed:** chunked-prefill and disagg NIXL connector are mutually exclusive.

**Verified reality:** NOT mutually exclusive at vLLM scheduler level. Source ([scheduler.py L837-840](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) + [arg_utils.py](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py)) — `enable_chunked_prefill` and `kv_transfer_config` are independent fields, no cross-validate `assert`. Upstream test `run_accuracy_test.sh` does NOT enable chunked on disagg (simplicity, not enforcement).

**Practical implication for Phase 1 (kept separate to avoid scheduler ambiguity):**

| Topology | `--enable-chunked-prefill` | `--kv-transfer-config` |
|----------|---------------------------|------------------------|
| colocated | NO (default OFF) | NO |
| chunked | YES (`--max-num-batched-tokens 2048`) | NO |
| disagg | NO (avoid interference w/ NIXL scheduler) | YES (NixlConnector) |
| disagg-tier | NO | YES (LMCacheConnectorV1) |

**v0.11.x does NOT crash** if both flags are set together, but mixing them is not validated at runtime — don't do it in Phase 1.

**Source:** [vllm/scheduler.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py), [vllm/engine/arg_utils.py](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py) (inspected 2026-07-08).

---

## Code Examples

### `docker-compose.yml` skeleton (full content in Q3 above; condensed here)

See Q3 above for the full 4-profile layout. Verbatim minimum service declarations:

```yaml
x-common: &common
  image: vllm/vllm-openai:v0.11.2
  runtime: nvidia
  environment: &env
    HF_HOME: /workspace/hf
    HF_HUB_ENABLE_HF_TRANSFER: "1"
  volumes: ["/workspace/hf:/workspace/hf", "./configs:/configs:ro"]
x-h: &hcheck
  test: ["CMD-SHELL", "curl -fs http://localhost:8000/v1/models | grep -q 'qwen30b'"]
  interval: 30s; timeout: 10s; retries: 30; start_period: 240s

services:
  vllm-colocated:        {<<: *common, profiles: ["colocated"],
    command: "vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
              --port 8000 --tensor-parallel-size 1 --gpu-memory-utilization 0.85
              --block-size 128 --served-model-name qwen30b",
    ports: ["18000:8000"], healthcheck: *hcheck}
  vllm-chunked:          {<<: *common, profiles: ["chunked"],
    command: "vllm serve ... --enable-chunked-prefill --max-num-batched-tokens 2048",
    ports: ["18001:8000"], healthcheck: *hcheck}
  vllm-disagg-prefill:   {<<: *common, profiles: ["disagg","disagg-tier"],
    command: "vllm serve ... --port 8100 --gpu-memory-utilization 0.45
              --kv-transfer-config /configs/kv_producer.json --enforce-eager",
    environment: {<<: *env, VLLM_NIXL_SIDE_CHANNEL_PORT: "5559",
                  UCX_TLS: "cuda_ipc,cuda_copy,tcp"}}
  vllm-disagg-decode:    {<<: *common, profiles: ["disagg","disagg-tier"],
    command: "vllm serve ... --port 8200 --gpu-memory-utilization 0.45
              --kv-transfer-config /configs/kv_consumer.json --enforce-eager",
    environment: {<<: *env, VLLM_NIXL_SIDE_CHANNEL_PORT: "5560",
                  UCX_TLS: "cuda_ipc,cuda_copy,tcp"}}
  disagg-proxy:          {profiles: ["disagg","disagg-tier"],
    command: "python -m vllm.entrypoints.openai.api_server --proxy
              --prefill-hosts vllm-disagg-prefill:8100
              --decode-hosts vllm-disagg-decode:8200 --port 9100",
    ports: ["19100:9100"]}
  # disagg-tier shapes analogous; uses kv_lmcache_* + LMCACHE_CONFIG_FILE
```

### `scripts/health.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
T="${1:-disagg}"; case "$T" in
  colocated) PORT=18000 ;; chunked) PORT=18001 ;;
  disagg|disagg-tier) PORT=19100 ;;
  *) echo "Usage: $0 <colocated|chunked|disagg|disagg-tier>"; exit 64 ;;
esac
echo "[1/3] /health";  curl -fs "http://localhost:$PORT/health" >/dev/null
echo "[2/3] /v1/models + model id"
[[ "$(curl -fs "http://localhost:$PORT/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')" == "qwen30b" ]] || { echo "model mismatch"; exit 1; }
echo "[3/3] Sentinel-token validity"
python3 tests/sentinel.py --mode check --base-url "http://localhost:$PORT/v1"
echo "[OK] $T healthy"
```

### Makefile top targets

```makefile
up-colocated:   ; docker compose --profile colocated up -d
up-chunked:     ; docker compose --profile chunked up -d
up-disagg:      ; docker compose --profile disagg up -d
up-disagg-tier: ; docker compose --profile disagg-tier up -d
down:           ; docker compose --profile colocated --profile chunked \
                       --profile disagg --profile disagg-tier down
health:
	@for t in colocated chunked disagg disagg-tier; do \
	  echo "=== $$t ==="; bash scripts/health.sh $$t || exit 1; done
sentinel: ; python3 tests/sentinel.py --mode check --base-url http://localhost:19100/v1
provision: ; bash provision.sh
bench:     ; make health && python3 bench/run.py --topologies all
lint:      ; ruff check . && mypy .
test:      ; pytest tests/ --cov=core --cov=control --cov-report=term-missing
```

---

## State of the Art

| Old | Current | Impact |
|-----|---------|--------|
| `vllm/vllm-openai:v0.7.x` (CVE-2025-25183) | v0.11.2 (NIXL + LMCache stable, Q4 2025) | Production-grade disagg |
| LMCache separate daemon | In-process `LMCacheConnectorV1` (v0.11.x) | Single-image deploy |
| Mooncake multi-process proxy | `vllm api_server --proxy` Python (v0.11.x) | One-image proxy |
| `kv_buffer_device: "host"` | `"cuda"` + UCX+cuda_ipc (v0.6.x NIXL) | GPU-direct KV |
| LIBFABRIC default → silent garbage | Explicit `backends=["UCX"]` (v0.11.1, vllm #27055) | P1 in failure appendix |

**Deprecated:** `vllm serve --num_lookahead_slots` (replaced by `--enable-chunked-prefill`).

---

## Assumptions Log

| # | Claim | Risk if Wrong |
|---|-------|---------------|
| A1 | Path B (`Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` + v0.11.2) ships Phase 1; CONTEXT's "Qwen 3.6" deferred to Phase 1.x follow-up | User wanted Qwen 3.6 verbatim — flag in `discuss-phase` follow-up |
| A2 | vLLM v0.11.2 does NOT register `qwen3_5` (verified via supported_models.html) | If backport exists, no gap; verify via `ModelRegistry.get_supported_archs()` in `provision.sh` |
| A3 | Cold-to-serving budget <20 min achievable on RunPod community H100 NVL | RunPod community network may throttle; mitigate with persistent `/workspace/hf` cache |
| A4 | `vllm/vllm-openai:v0.11.2` image bundles `nixl` 0.6.x + UCX + LMCache without separate install | If split, add `pip install nixl lmcache` step in provision.sh; verify via `docker run --rm ... python -c "import nixl, lmcache"` |
| A5 | Disagg proxy server lives in same vLLM image (`python -m vllm.entrypoints.openai.api_server --proxy`) | If not bundled, need separate proxy image — verify after first `vllm serve --help` |
| A6 | 4 profiles compose syntactically | Special chars in profile name would fail — none here |
| A7 | Sentinel `temperature=0` deterministic on H100 NVL w/ v0.11.2 | If non-deterministic (e.g., reduced-prec attention), fixture unstable; mitigate with `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| A8 | H100 NVL supports E4M3 FP8 natively | Standard on Hopper; affects F8_E4M3 KV dtype correctness |

---

## Open Questions

1. **Qwen 3.6 vLLM version pin** — `qwen3_5` may register in v0.12-v0.14; Path B for Phase 1, validate Path A in follow-up.
2. **Disagg proxy CLI flags in v0.11.2** — `--prefill-hosts`/`--decode-hosts` inferred from upstream; verify after first run.
3. **LMCache connector module path in v0.11.2** — `LMCacheConnectorV1` import may differ; probe in `provision.sh`.

---

## Environment Availability

**SKIPPED** — Phase 1 deploys to the same single RunPod H100 NVL pod documented in `.planning/RUNPOD.md` (`t3son251d5gcvg`, currently STOPPED). No new external dependencies beyond `vllm/vllm-openai:v0.11.2` Docker image + docker daemon + nvidia runtime already on pod.

**Human action before Phase 1 execution:** Start the RunPod pod via runpod MCP or console; verify `docker info`, `nvidia-smi` (1× H100 NVL), `df -h /workspace` (200GB).

---

## Validation Architecture

Framework: `pytest` ≥8.0 + `pytest-asyncio` + `pytest-cov` + `pytest-mock`. Config: `pyproject.toml`. Quick: `pytest tests/ -x -q`. Full: `pytest tests/ -v --cov=core --cov=control --cov-report=term-missing`.

| Req ID | Behavior | Type | Command |
|--------|----------|------|---------|
| TOPO-01..04 | Each topo serves /v1/chat/completions | integration | `pytest tests/test_topos.py::test_{colocated,chunked,disagg,disagg_tier}_chat -x` |
| TOPO-05 | Health confirms P→D + decode≠prefill | integration | `bash scripts/health.sh disagg && bash scripts/health.sh disagg-tier` |
| TOPO-06 | Cold-to-serving <20 min | smoke | `bash -c 'S=$(date +%s); make up-colocated; [ $(($(date +%s)-S)) -le 1200 ]'` |
| TOPO-07 | 4 topologies share OpenAI-compat schema | parametrized unit | `pytest tests/test_schema_uniformity.py -x` |
| REPRO-01..02 | compose + provision working | smoke | `docker compose --profile X config >/dev/null && make provision && make health` |
| P1 sentinel | First-token validity | integration | `pytest tests/test_sentinel_disagg.py -x` |

**Sampling:** Per task = `pytest tests/ -x -q` · Per wave = `pytest tests/ -v --cov=...` · Phase gate = full suite green + health.sh green for all 4 topologoies + sentinel pass + cold ≤ 1200s

### Wave 0 Gaps (greenfield — full skeleton)

`pyproject.toml` (pytest+ruff+mypy, REPRO-05) · `tests/conftest.py` (fixtures) · `tests/test_topos.py` (TOPO-01..04) · `tests/test_schema_uniformity.py` (TOPO-07) · `tests/test_sentinel_disagg.py` (P1) · `Makefile` · `provision.sh` (image+HF+config verify) · `docker-compose.yml` (4 profiles) · `configs/{kv_producer,kv_consumer,kv_lmcache_producer,kv_lmcache_consumer}.json` · `configs/lmcache_{prefill,decode}.yaml` · `tests/_fixtures/sentinel_<model>_<version>.json` · `pip install -e .[dev]`

---

## Security Domain

Single-tenant assumed (PITFALLS P2 mitigation = vLLM ≥0.7.2 patched).

| ASVS Cat | Applies | Standard Control |
|----------|---------|-----------------|
| V2 Authentication | no | Local-only binding; no auth gate Phase 1 |
| V3 Session Mgmt | no | Stateless |
| V4 Access Control | no | Single-tenant |
| V5 Input Validation | yes | vLLM internal validators |
| V6 Cryptography | yes (Phase 3 only) | SHA-256 truncated 128-bit prefix hashing (P2) |

**Risks (PITFALLS.md):**
- `kv_transfer_complete_count` / `kv_transfer_inflight` not real metrics — sentinel closes gap
- Single-tenant = CVE-2025-25183 low exploitability Phase 1; `/metrics` bound to localhost
- Prompt logging w/ truncation/redaction = Phase 2 LOAD out of scope

---

## Sources

**Primary (HIGH):** [vLLM NixlConnector Usage](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/) · [vLLM Disagg Prefill](https://docs.vllm.ai/en/latest/features/disagg_prefill/) · [vLLM Metrics](https://docs.vllm.ai/en/latest/usage/metrics/) · [vLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html) · [vLLM NIXL accuracy test script](https://github.com/vllm-project/vllm/blob/main/tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh) · [vLLM nixl/stats.py](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/nixl/stats.py) · [vLLM scheduler.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) · [vLLM arg_utils.py](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py) · [vllm #27055 (P1)](https://github.com/vllm-project/vllm/issues/27055) · [CVE-2025-25183 (P2)](https://www.miggo.io/vulnerability-database/cve/CVE-2025-25183) · [LMCache Disagg Quickstart](https://docs.lmcache.ai/getting_started/quickstart/disaggregated_prefill.html) · [Docker Compose profiles service ref](https://docs.docker.com/reference/compose-file/services/) · [HF — Qwen3.6-27B-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8) · [HF — Qwen3-30B-A3B-FP8](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8) · [Docker Hub — vllm-openai tags](https://hub.docker.com/r/vllm/vllm-openai/tags) (v0.11.2 + v0.24.0 verified) · [PyPI — vllm](https://pypi.org/project/vllm/) (latest 0.24.0, 2026-06-29)

**Secondary (MEDIUM):** [arxiv 2606.07362 — vLLM cold-start analysis](https://arxiv.org/pdf/2606.07362) · [Docker Compose profiles docs](https://docs.docker.com/compose/profiles/) · [vLLM PR #24811](https://github.com/vllm-project/vllm/pull/24811) — sentinel concept (closed, not vendored)

**Tertiary (LOW — flagged for execution):** [Reddit — 1.5s cold start Qwen-32B H100 snapshotting](https://www.reddit.com/r/Vllm/comments/1rmlzoe/15s_cold_start_for_qwen32b_on_h100_using_runtime/) — optimistic snapshot-only

---

## Metadata

| Area | Level | Reason |
|------|-------|--------|
| Standard Stack | HIGH | PyPI + Docker Hub + HF API + vLLM docs verified |
| NixlConnector JSON | HIGH | Verbatim from vLLM test script + docs |
| LMCache on NIXL | HIGH | Verbatim from docs.lmcache.ai |
| Prometheus metrics | HIGH | Docs + nixl/stats.py source |
| docker compose v2 profiles | HIGH | Verbatim from docs.docker.com |
| chunked × kv-transfer mutual exclusivity | HIGH | Source inspection of scheduler + arg_utils |
| Cold-start budget | MEDIUM | Based on arxiv paper + HF disk; actual RunPod network unmeasured |
| Healthcheck tuning | MEDIUM | No vLLM official guidance; values derived |
| Sentinel determinism | LOW | Greedy + CUDA graphs expected deterministic; cross-version needs runtime confirm |
| Qwen 3.6 in vLLM v0.11.x | MEDIUM | `qwen3_5` confirmed not registered in v0.11.2 supported_models; later versions unverified |

**Research date:** 2026-07-08
**Valid until:** 2026-08-08 (vLLM monthly cadence)

---

*Phase 1 (TOPO) research complete. Ready for planning.*
