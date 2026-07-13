#!/usr/bin/env bash
# GoodputLab model compatibility probe + download.
#
# Probes the preferred model id against the pinned vLLM image; if the image
# cannot construct a tokenizer (typical of unsupported architectures under
# v0.11.2), prints a single MODEL_FALLBACK_USED line and retries against the
# fallback. Fails non-zero if both probes fail. Last line of stdout always
# prints the selected MODEL_ID so upstream callers can capture it.
#
# Refs: .planning/phases/01-topologies-topo/01-02-PLAN.md,
#       .planning/PROJECT.md D-01 (model selection)
#
# Exit codes:
#   0  - preferred resolved and downloaded (or already cached)
#   1  - fallback also failed (no model usable)
#   2  - vLLM import itself failed inside the image
set -euo pipefail

VLLM_IMAGE=${VLLM_IMAGE:-vllm/vllm-openai:v0.11.2}
HF_HOME=${HF_HOME:-/workspace/hf}
HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
PREFERRED_MODEL_ID=${PREFERRED_MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}
FALLBACK_MODEL_ID=${FALLBACK_MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}

# When --probe-only is passed, skip the download step (used by health probes
# where the model is already cached and we only need to verify resolvability).
PROBE_ONLY=0
if [ "${1:-}" = "--probe-only" ]; then
  PROBE_ONLY=1
fi

probe_model() {
  local model_id="$1"
  docker run --rm \
    -e HF_HOME="$HF_HOME" \
    -e HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER" \
    -v "$HF_HOME:$HF_HOME" \
    "$VLLM_IMAGE" \
    python -c "
from huggingface_hub import model_info
mi = model_info('${model_id}')
print('MODEL_OK', '${model_id}', getattr(mi, 'pipeline_tag', 'unknown'))
"
}

download_model() {
  local model_id="$1"
  docker run --rm \
    -e HF_HOME="$HF_HOME" \
    -e HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER" \
    -v "$HF_HOME:$HF_HOME" \
    "$VLLM_IMAGE" \
    python -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('${model_id}')
print('TOKENIZER_LOADED', '${model_id}', getattr(t, 'vocab_size', 'n/a'))
"
}

# Sanity: the vLLM image must be importable before we probe models.
if ! docker run --rm "$VLLM_IMAGE" python -c "import vllm" >/dev/null 2>&1; then
  echo "FATAL_VLLM_IMPORT image=${VLLM_IMAGE}" >&2
  exit 2
fi

# Prefer first.
if probe_model "$PREFERRED_MODEL_ID" >/dev/null 2>&1; then
  MODEL_ID="$PREFERRED_MODEL_ID"
  REASON="preferred_probe_ok"
else
  echo "MODEL_FALLBACK_USED preferred=${PREFERRED_MODEL_ID} fallback=${FALLBACK_MODEL_ID} reason=unsupported_architecture"
  if ! probe_model "$FALLBACK_MODEL_ID" >/dev/null 2>&1; then
    echo "FATAL_MODEL_PROBE preferred=${PREFERRED_MODEL_ID} fallback=${FALLBACK_MODEL_ID}" >&2
    exit 1
  fi
  MODEL_ID="$FALLBACK_MODEL_ID"
  REASON="fallback_probe_ok"
fi

if [ "$PROBE_ONLY" -eq 1 ]; then
  printf 'MODEL_ID=%s\n' "$MODEL_ID"
  exit 0
fi

# Always download (idempotent — HF cache layer skips if present).
download_model "$MODEL_ID" || {
  echo "FATAL_MODEL_DOWNLOAD model=${MODEL_ID}" >&2
  exit 1
}

# Final line for downstream capture (provision.sh reads it).
printf 'MODEL_ID=%s\n' "$MODEL_ID"
