#!/usr/bin/env bash
set -eu

MODEL_PATH="${MODEL_PATH:-/mnt/Model-Weights/gemma-4-31B-it}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gemma4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-/mnt/foxbrain-omni-physical/stage1/CoRL-evaluation}"

uv run --extra serve vllm serve \
  --model "${MODEL_PATH}" \
  --trust-remote-code \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len 32768 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs 8 \
  --max-num-batched-tokens 65536 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --async-scheduling \
  --mm-processor-cache-gb 0 \
  --allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}"
