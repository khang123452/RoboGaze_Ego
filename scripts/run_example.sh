#!/usr/bin/env bash
set -eu

export LOCAL_BASE_URL="${LOCAL_BASE_URL:-http://localhost:8000/v1}"
export LOCAL_MODEL="${LOCAL_MODEL:-gemma4}"

uv run robogaze \
  --task-instruction "The robot arm is performing a task. Use the left hand to pick up dragonfruit from pink plate to teal plate." \
  --initial-frame "inputs/robogaze_dataset/gr1_real/conditioning_frame/4.jpg" \
  --video "inputs/robogaze_dataset/gr1_real/videos/4.mp4" \
  --output-dir "${OUTPUT_DIR:-outputs/robogaze}" \
  --video-id "${VIDEO_ID:-example}" \
  --vlm-concurrency "${VLM_CONCURRENCY:-4}"
