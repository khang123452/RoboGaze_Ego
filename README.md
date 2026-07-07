# RoboGaze

RoboGaze is a local VLM inference pipeline for detecting execution glitches in robot videos.

This branch is the supplementary-code version of the project. It keeps the inference code and the local vLLM serving command, and removes the paper/evaluation/annotation workspace.

## Install

```bash
uv sync
```

For the GPU server that also serves the VLM:

```bash
uv sync --extra serve
```

The pipeline also expects `ffmpeg` and `ffprobe` on the system path for video subclips and media metadata.

## Serve Gemma

```bash
bash scripts/serve_vlm.sh
```

The script serves the model as `gemma4` on `http://localhost:8000/v1`.
Additional presets are available for throughput experiments:

- `scripts/serve_vlm_safe.sh`: `--max-num-seqs 4`, `--max-num-batched-tokens 32768`, `--max-model-len 40960`
- `scripts/serve_vlm_balanced.sh`: `--max-num-seqs 8`, `--max-num-batched-tokens 49152`, `--max-model-len 40960`
- `scripts/serve_vlm_throughput.sh`: `--max-num-seqs 16`, `--max-num-batched-tokens 65536`, `--max-model-len 32768`

The scripts use these environment-variable defaults, which can be overridden:
`MODEL_PATH`, `SERVED_MODEL_NAME`, `HOST`, `PORT`, `TP_SIZE`,
`GPU_MEMORY_UTILIZATION`, and `ALLOWED_LOCAL_MEDIA_PATH`.

## Run Inference

```bash
export LOCAL_BASE_URL=http://localhost:8000/v1
export LOCAL_MODEL=gemma4

uv run robogaze \
  --task-instruction "Use the right hand to pick up the red cup and place it on the plate." \
  --initial-frame /path/to/initial_frame.jpg \
  --video /path/to/execution.mp4 \
  --output-dir outputs/robogaze \
  --video-id example_001 \
  --vlm-concurrency 8
```

Set `--vlm-concurrency` close to the server preset's `--max-num-seqs` value.
The pipeline parallelizes independent VLM calls for window states, subgoal
routing, specialist agents, and boundary refinements while preserving
deterministic output ordering.

Main outputs are written under `outputs/robogaze/<video-id>/`:

- `report.json`: final structured glitch report
- `video_meta.json`: video metadata and detected layout
- `view_bank.json`: generated medium-window and subgoal video clips
- `task_memory.json`, `scene_memory.json`, `window_states.json`, `subgoal_segments.json`: intermediate model state
- `coarse_candidates.json`, `hypotheses.json`, `verifier.json`, `verified_event_groups.json`: inference trace
- `cache_keys.json`: request cache trace

Generated cache files go to `cache/robogaze/` by default.

## Notes

- The default local model name is `gemma4`, matching `scripts/serve_vlm.sh`.
- Video inputs are passed to the OpenAI-compatible server as local `file://` media paths, so vLLM must be started with an `--allowed-local-media-path` that covers your data.
- This is research code. The implementation is intentionally direct and keeps intermediate JSON files for debugging failed runs.
