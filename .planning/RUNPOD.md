# GoodputLab — RunPod Configuration

**Project:** GoodputLab — SLO-aware disagg LLM serving
**Pod ID:** `t3son251d5gcvg`
**Name:** goodputlab-dev
**Status:** STOPPED (resume when ready to install + test)

## Spec

| Field | Value |
|-------|-------|
| GPU | 1× NVIDIA H100 NVL (94GB HBM3) |
| Cost | $2.59/hr community cloud |
| Image | vllm/vllm-openai:latest |
| Container disk | 100 GB |
| Volume | 200 GB @ /workspace (preserved across stop/start) |
| Region | (assigned by runpod; community pool) |

## Public Endpoints (when running)

| Service | Private | Public |
|---------|--------:|-------:|
| SSH | 22 | `38.143.35.131:16778` |
| Jupyter | 8888 | (port not exposed — add if needed) |
| vLLM API | 8000 | `100.65.25.97:60040` |
| Router (FastAPI) | 8080 | `100.65.25.97:60041` |
| Prometheus | 9090 | `100.65.25.97:60042` |
| Grafana | 3000 | `100.65.25.97:60039` |

## Env Vars Set

- `VLLM_LOGGING_LEVEL=INFO`
- `HF_HOME=/workspace/hf`
- `GOODPUTLAB_HOME=/workspace/goodputlab`

## SSH Access

```bash
ssh -p 16778 root@38.143.35.131   # if SSH key registered
# OR via runpod web terminal (https://runpod.io/console/pods)
```

If SSH key not registered, use runpod web terminal — no key required.

## Lifecycle

- **Stop:** `mcp__runpod__stop-pod(podId="t3son251d5gcvg")` — billing stops, disk+volume preserved
- **Start:** `mcp__runpod__start-pod(podId="t3son251d5gcvg")` — resumes same pod, billing resumes
- **Destroy:** `mcp__runpod__delete-pod(podId="t3son251d5gcvg")` — permanent, volume also lost unless detached first

## Cost Discipline

- Total budget: $600-1200 across all 8 phases
- Phase 1 dev: ~$50 target (24h)
- Phase 8 bench: $400-800 (4-8× H100, 24-48h)
- **Always stop pod between sessions** — never leave idle
- Snapshot results to S3/parquet immediately after each bench

---
*Pod created: 2026-07-08 22:44 UTC*
*Stopped: 2026-07-08 22:44 UTC (idle, no code yet)*