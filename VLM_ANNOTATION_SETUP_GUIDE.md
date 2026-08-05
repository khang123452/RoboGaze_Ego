# Running RoboGaze-Ego with a VLM Backend — Setup Guide

**Goal:** Serve a vision-language model (Qwen2.5-VL / Qwen3-VL / Gemma 4, etc.) locally and run it through the `RoboGaze-Ego` pipeline against a private egocentric human video dataset to produce per-video annotations (`report.json`).

This guide assumes a fresh machine with one or more NVIDIA GPUs and a reasonably current driver/CUDA stack. It does not assume anything about a specific cluster setup — adapt paths/GPU counts to your actual hardware.

---

## 1. Prerequisites

- Python 3.10–3.12
- One or more NVIDIA GPUs with enough VRAM for your chosen model (see sizing notes in §4)
- [`uv`](https://docs.astral.sh/uv/) for Python environment/dependency management (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- `ffmpeg` **and** `ffprobe` on `PATH`, built with AV1 decode support (`libdav1d`). Many egocentric-capture devices (smart glasses, action cams) encode in AV1/HEVC, and OpenCV's bundled decoder cannot read AV1 — RoboGaze-Ego needs a real `ffmpeg`/`ffprobe` for this (see §7 known issue).
  - Verify: `ffmpeg -version` should list `--enable-libdav1d` in its build config.

## 2. Get the repo and install dependencies

```bash
git clone <robogaze-ego-repo-url> RoboGaze_Ego
cd RoboGaze_Ego
uv sync --extra serve
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`torch.cuda.is_available()` must print `True` before continuing. If it doesn't, the GPU driver/CUDA setup needs fixing first — that's environment-specific to whatever machine you're on.

## 3. Pick a VLM backend

RoboGaze-Ego needs a model that can accept **image and video input** — it attaches frames to almost every prompt (task grounding, subgoal detection, verification). A text-only model will fail immediately with errors like `Unknown model type: <arch>` the moment an image is attached.

Reasonable options, roughly in order of how well-established their vLLM support is:

| Model | Notes |
|---|---|
| Qwen2.5-VL (7B/32B/72B) | Solid, mature vLLM support. Good default choice. |
| Qwen3-VL | Newer, larger context/video handling. Needs a recent vLLM. |
| Gemma 4 (E2B/E4B/31B) | Native vision+audio. Needs vLLM ≥0.19.0 and a matching `transformers` build. |

**Before serving, confirm your installed vLLM actually supports the architecture** — this saves a lot of wasted time chasing unrelated errors:

```bash
python -c "
import json
from vllm import ModelRegistry
cfg = json.load(open('<path-to-model>/config.json'))
arch = cfg['architectures'][0]
print(arch, '-> supported:', arch in ModelRegistry.get_supported_archs())
"
```

If `False`, upgrade `vllm` (`uv pip install -U vllm`) and re-check, or pick a different model your current vLLM version supports.

## 4. Serve the model

```bash
uv run --extra serve vllm serve <path-or-hf-id-to-model> \
  --served-model-name vlm \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size <N_GPUS> \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --allowed-local-media-path <dir-containing-your-videos>
```

Sizing notes (rough, bf16 weights):
- A ~27B dense model needs ~50GB VRAM for weights alone — fits on one 80GB GPU with headroom for KV cache.
- A ~35B MoE model can be ~65-70GB on disk — plan for 2 GPUs (`--tensor-parallel-size 2`) rather than one.
- Leave real headroom beyond raw weight size for KV cache/activations — don't run right at the edge of VRAM.

Wait for the server to log `Uvicorn running on http://0.0.0.0:8000` and start printing idle throughput heartbeats before moving on. Leave this running in its own terminal/session.

## 5. Point RoboGaze-Ego at the server

In a separate terminal (with the venv activated):

```bash
export LOCAL_BASE_URL=http://localhost:8000/v1
export LOCAL_MODEL=vlm
```

## 6. Quick single-episode test

Run one real video through the pipeline before doing anything at scale:

```bash
ffmpeg -y -loglevel error -i /path/to/episode.mp4 -frames:v 1 /tmp/test_frame.jpg

uv run robogaze \
  --task-instruction "Describe what the person is doing." \
  --initial-frame /tmp/test_frame.jpg \
  --video /path/to/episode.mp4 \
  --output-dir outputs/robogaze \
  --video-id test_episode_001
```

Check `outputs/robogaze/test_episode_001/report.json` — if it's populated with a real annotation, the full chain (server, ffmpeg, pipeline) is working.

## 7. Known issue: AV1-encoded video + OpenCV

`src/robogaze/views.py`'s `save_single_frame()` function uses `cv2.VideoCapture` to grab specific frames from the *original* source video. OpenCV's bundled decoder cannot decode AV1 — if your dataset is AV1-encoded (common for egocentric capture devices), you'll hit:

```
RuntimeError: Could not read frame <N> from <video_path>
```

Fix: patch `save_single_frame` to decode via `ffmpeg` (which handles AV1 fine via `libdav1d`) instead of `cv2`, falling back to `cv2` only if `ffmpeg` isn't available. Add near the top of `views.py`:

```python
import tempfile  # add to imports

def _read_frame_ffmpeg(video_path, frame_index, fps):
    """Decode a single frame via ffmpeg. Handles codecs cv2 can't (e.g. AV1)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    fd, tmp_name = tempfile.mkstemp(suffix=".png")
    import os
    os.close(fd)
    tmp = Path(tmp_name)
    timestamp = frame_index / fps if fps > 0 else 0.0
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path), "-ss", f"{timestamp:.6f}",
        "-frames:v", "1", str(tmp),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return None
    img = Image.open(tmp).convert("RGB")
    img.load()
    tmp.unlink(missing_ok=True)
    return img
```

Then in `save_single_frame`, try `_read_frame_ffmpeg(...)` first and only fall back to the existing `cv2.VideoCapture` path if it returns `None`. (`write_window_videos`, which cuts the per-window subclips, already shells out to real `ffmpeg` and is unaffected by this issue.)

If you'd rather not patch the code, an alternative is transcoding your dataset to H.264 once upfront with `ffmpeg -c:v libx264` before running the pipeline.

## 8. Annotating a full dataset

For a handful of videos, loop the single-episode command from §6 over your files:

```bash
for video in /path/to/dataset/*.mp4; do
  id=$(basename "$video" .mp4)
  ffmpeg -y -loglevel error -i "$video" -frames:v 1 "/tmp/${id}_frame.jpg"
  uv run robogaze \
    --task-instruction "<your task description or a per-video lookup>" \
    --initial-frame "/tmp/${id}_frame.jpg" \
    --video "$video" \
    --output-dir outputs/robogaze \
    --video-id "$id"
done
```

For larger datasets, `scripts/run_robogaze_datasets.py` is a proper batch runner with resumability, concurrency control, and a summary JSONL — but it expects a specific input layout:

```
inputs/robogaze_dataset/<dataset_name>/
  metadata.csv          # columns: file_name, text (task instruction)
  videos/<file_name>
  conditioning_frame/<stem>.<jpg|jpeg|png>
  video_layout.txt      # optional, for multi-view videos
```

It only accepts dataset names already in its `DATASETS` tuple — add your dataset name there:

```bash
sed -i 's/DATASETS = ("gr1_real", "gr1_sim", "droid_mv")/DATASETS = ("gr1_real", "gr1_sim", "droid_mv", "<your_dataset_name>")/' scripts/run_robogaze_datasets.py
```

Then:

```bash
uv run python scripts/run_robogaze_datasets.py \
  --datasets <your_dataset_name> \
  --input-root inputs/robogaze_dataset \
  --output-dir outputs/robogaze_dataset \
  --vlm-concurrency 8 \
  --limit 1   # drop this once a small test batch looks right
```

## 9. Output

Each processed video produces `outputs/.../<video_id>/report.json` containing the pipeline's structured annotation (subgoals, detected glitches/anomalies, per-window analysis, etc. — see `src/robogaze/schemas.py` for the full schema).

## 10. Other troubleshooting notes

- **`Unknown model type: <arch>` on the first API call**: the served model doesn't have vision support registered in this vLLM version, or isn't a VLM at all. Re-check §3.
- **`response_format`/JSON-mode errors**: `client.py`'s `call_json()` requests `response_format={"type": "json_object"}`, which vLLM implements via a guided-decoding backend (`outlines` or `lm-format-enforcer` — set with `--guided-decoding-backend` on newer vLLM, or it's on by default). If a specific model architecture isn't recognized by the backend's tokenizer integration, try switching backends via that flag before assuming the model itself is unsupported.
- **Multiple GPUs / OOM on load**: check `nvidia-smi` for stale processes still holding VRAM from a previous aborted server before assuming you need more GPUs.
