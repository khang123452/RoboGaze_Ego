#!/usr/bin/env bash
set -eu

export LOCAL_BASE_URL="${LOCAL_BASE_URL:-http://localhost:8000/v1}"
export LOCAL_MODEL="${LOCAL_MODEL:-gemma4}"

uv run python scripts/run_robogaze_datasets.py \
  --input-root "${INPUT_ROOT:-inputs/robogaze_dataset}" \
  --output-dir "${OUTPUT_DIR:-outputs/robogaze_dataset}" \
  --cache-dir "${CACHE_DIR:-cache/robogaze_dataset}" \
  --vlm-concurrency "${VLM_CONCURRENCY:-8}" \
  "$@"
