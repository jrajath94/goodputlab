#!/bin/bash
# GoodputLab pod-side vLLM launcher — one honest serving config per mode.
#
#   launch_vllm.sh colocated    single process, chunked prefill OFF, :8000
#   launch_vllm.sh chunked      single process, chunked prefill ON,  :8000
#   launch_vllm.sh disagg-2gpu  prefill on GPU0 :8100 + decode on GPU1 :8200
#                               (NIXL) + disagg_proxy :9100 — requires 2 GPUs
#
# Ops notes (learned on paid pods, 2026-07-17):
# - Kill and launch in SEPARATE ssh sessions: a pkill pattern that also
#   appears in the launch text matches the session's own bash and kills it.
# - P/D pairs on a shared GPU must start SEQUENTIALLY; simultaneous
#   startup makes both engines profile memory mid-load and abort. With
#   one GPU per process (this script) parallel start is safe, but we
#   keep sequential startup anyway — it costs ~90 s and removes a race.
# - setsid + </dev/null so servers survive ssh session close.
set -euo pipefail

MODE="${1:?usage: launch_vllm.sh colocated|chunked|disagg-2gpu}"
MAX_LEN="${MAX_MODEL_LEN:-20480}"
MODEL_DIR=$(ls -d /workspace/hf/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/)
mkdir -p /workspace/logs

wait_ready () { # port logfile
  for _ in $(seq 1 90); do
    curl -sf "http://127.0.0.1:$1/v1/models" > /dev/null && return 0
    sleep 5
  done
  echo "PORT $1 NOT READY"; tail -5 "$2"; exit 1
}

case "$MODE" in
  colocated|chunked)
    if [ "$MODE" = "chunked" ]; then FLAG="--enable-chunked-prefill"; else FLAG="--no-enable-chunked-prefill"; fi
    setsid nohup vllm serve "$MODEL_DIR" \
      --host 0.0.0.0 --port 8000 --max-model-len "$MAX_LEN" $FLAG \
      --served-model-name goodputlab-model \
      > "/workspace/logs/vllm_${MODE}.log" 2>&1 < /dev/null &
    wait_ready 8000 "/workspace/logs/vllm_${MODE}.log"
    echo "${MODE} READY :8000 (max-model-len ${MAX_LEN})"
    ;;
  disagg-2gpu)
    CUDA_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_PORT=5601 setsid nohup \
      vllm serve "$MODEL_DIR" \
      --host 0.0.0.0 --port 8100 --max-model-len "$MAX_LEN" \
      --gpu-memory-utilization 0.90 --served-model-name goodputlab-model \
      --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' \
      > /workspace/logs/vllm_prefill.log 2>&1 < /dev/null &
    wait_ready 8100 /workspace/logs/vllm_prefill.log
    echo "prefill READY :8100 (GPU0)"

    CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_PORT=5602 setsid nohup \
      vllm serve "$MODEL_DIR" \
      --host 0.0.0.0 --port 8200 --max-model-len "$MAX_LEN" \
      --gpu-memory-utilization 0.90 --served-model-name goodputlab-model \
      --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}' \
      > /workspace/logs/vllm_decode.log 2>&1 < /dev/null &
    wait_ready 8200 /workspace/logs/vllm_decode.log
    echo "decode READY :8200 (GPU1)"

    setsid nohup python3 /workspace/disagg_proxy.py \
      --host 0.0.0.0 --port 9100 \
      --prefiller-host 127.0.0.1 --prefiller-port 8100 \
      --decoder-host 127.0.0.1 --decoder-port 8200 \
      > /workspace/logs/proxy.log 2>&1 < /dev/null &
    sleep 8
    curl -sf http://127.0.0.1:9100/health && echo "DISAGG_2GPU_READY :9100"
    ;;
  *)
    echo "unknown mode: $MODE"; exit 2
    ;;
esac
