#!/usr/bin/env bash
# GoodputLab RunPod provisioning with single 1200s budget gate (TOPO-06).
#
# Stages, each tracked against the shared deadline:
#   1. Pre-flight (docker, compose, nvidia-smi, HF cache writability)
#   2. vLLM image pull + import smoke
#   3. Model resolution and download (preferred + fallback probe)
#   4. Colocated topology boot to /health HTTP 200
#   5. Sentinel fixture record (tests/sentinel.py --mode record)
#   6. Health check (scripts/health.sh colocated)
#
# Any stage that overflows the 1200s budget exits non-zero with
# "TOPO-06 BUDGET EXCEEDED". Re-runs are idempotent (compose up, model already
# cached, sentinel already recorded unless --force-fixture is passed).
#
# Reference: .planning/phases/01-topologies-topo/01-02-PLAN.md,
#            .planning/research/PITFALLS.md (P1, P2)
set -euo pipefail

# ─── Configuration ─────────────────────────────────────────────────────────
export HF_HOME=${HF_HOME:-/workspace/hf}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export GOODPUTLAB_HOME=${GOODPUTLAB_HOME:-/workspace/goodputlab}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-goodputlab-model}
export VLLM_IMAGE=${VLLM_IMAGE:-vllm/vllm-openai:v0.11.2}
export PREFERRED_MODEL_ID=${PREFERRED_MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}
export FALLBACK_MODEL_ID=${FALLBACK_MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}

PROVISION_BUDGET_SECONDS=${PROVISION_BUDGET_SECONDS:-1200}
STAGE_LOG_PREFIX="[provision]"

log() { printf '%s %s\n' "$STAGE_LOG_PREFIX" "$*"; }
elapsed() { echo "$(( $(date +%s) - START ))"; }
check_budget() {
  local secs
  secs=$(elapsed)
  if [ "$secs" -gt "$PROVISION_BUDGET_SECONDS" ]; then
    log "TOPO-06 BUDGET EXCEEDED at stage=$1 elapsed=${secs}s budget=${PROVISION_BUDGET_SECONDS}s"
    exit 1
  fi
}

log "starting provision; budget=${PROVISION_BUDGET_SECONDS}s"
START=$(date +%s)

# ─── 1. Pre-flight ────────────────────────────────────────────────────────
log "stage=preflight"
command -v docker >/dev/null || { log "FATAL: docker not found"; exit 1; }
docker compose version >/dev/null || { log "FATAL: docker compose not found"; exit 1; }
nvidia-smi -L >/dev/null || { log "FATAL: nvidia-smi not found (GPU required)"; exit 1; }
mkdir -p "${HF_HOME}" || { log "FATAL: cannot write to HF_HOME=${HF_HOME}"; exit 1; }
check_budget preflight

# ─── 2. vLLM image pull + import smoke (P2: pinned v0.11.2) ───────────────
log "stage=image_pull image=${VLLM_IMAGE}"
docker pull "$VLLM_IMAGE" || { log "FATAL: docker pull failed"; exit 1; }
docker run --rm "$VLLM_IMAGE" python -c "import vllm; print('vllm', vllm.__version__)" \
  || { log "FATAL: vLLM import failed inside image"; exit 1; }
check_budget image_pull

# ─── 3. Model resolution + download (preferred → fallback) ────────────────
log "stage=model_download preferred=${PREFERRED_MODEL_ID}"
# pull_model.sh always downloads the resolved id and prints MODEL_ID=<id> on
# its last line. Capture that line so compose env gets the right value.
PULL_OUTPUT=$(bash "${GOODPUTLAB_HOME}/scripts/pull_model.sh") \
  || { log "FATAL: model pull failed (preferred + fallback)"; echo "$PULL_OUTPUT"; exit 1; }
echo "$PULL_OUTPUT"
MODEL_ID=$(printf '%s\n' "$PULL_OUTPUT" | awk -F= '/^MODEL_ID=/{print $2; exit}')
if [ -z "${MODEL_ID:-}" ]; then
  log "FATAL: pull_model.sh did not emit MODEL_ID=<value>"
  exit 1
fi
export MODEL_ID
log "stage=model_download selected=${MODEL_ID}"
check_budget model_download

# ─── 4. Colocated topology boot ───────────────────────────────────────────
log "stage=boot_colocated"
docker compose --profile colocated up -d \
  || { log "FATAL: docker compose up colocated failed"; exit 1; }
# Curl /health with retry until ready or budget exhaustion.
HEALTH_URL="http://localhost:18000/health"
for attempt in $(seq 1 60); do
  if curl -fsS --max-time 5 "$HEALTH_URL" >/dev/null 2>&1; then
    log "stage=boot_colocated health=ok attempt=${attempt}"
    break
  fi
  check_budget boot_colocated
  sleep 5
done
curl -fsS --max-time 5 "$HEALTH_URL" >/dev/null \
  || { log "FATAL: /health never returned 200"; exit 1; }
check_budget boot_colocated

# ─── 5. Sentinel fixture record (P1: trusted KV-transfer validator) ──────
log "stage=sentinel_record"
python3 tests/sentinel.py \
  --mode record \
  --base-url http://localhost:18000/v1 \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --fixture-dir tests/_fixtures \
  || { log "FATAL: sentinel record failed"; exit 1; }
check_budget sentinel_record

# ─── 6. Health check (sentinel run + topology health gate) ───────────────
log "stage=health"
bash scripts/health.sh colocated \
  || { log "FATAL: health.sh colocated reported failures"; exit 1; }
check_budget health

log "done total_elapsed=$(elapsed)s budget=${PROVISION_BUDGET_SECONDS}s"
