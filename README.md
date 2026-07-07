# RoboGaze: Evaluating Robot World Models via Structured Vision-Language Analysis

<p align="center">
  <img src="assets/robogaze_overview.jpg" alt="RoboGaze overview" width="100%">
</p>

<p align="center">
  <a href="https://robogaze-eval.github.io/"><img src="https://img.shields.io/badge/Project%20Page-RoboGaze-2f855a" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2606.28385"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b" alt="arXiv"></a>
  <img src="https://img.shields.io/badge/Code-Released-orange" alt="Code Released">
</p>

RoboGaze is a training-free, multi-agent VLM framework for diagnosing failures in generated robot-manipulation videos. Given a task instruction, an initial frame, and a generated execution video, RoboGaze produces a structured report that explains what failed, when it failed, why it failed, and how severe the failure is.

This repository contains the official RoboGaze code release, including the local inference pipeline, OpenAI-compatible VLM client, batch runners, evaluation utilities, and static project page assets.

## Overview

Robot world models can generate visually plausible manipulation rollouts that still violate task logic, physical plausibility, robot-body consistency, or object-scene consistency. Scalar metrics and monolithic VLM judges often miss these errors or over-report failures in clean clips.

RoboGaze addresses this with a three-stage diagnostic pipeline:

1. **Task-scene grounding**: parses the instruction and initial frame into task memory, scene memory, expected subgoals, visible objects, robot parts, and layout information.
2. **Specialist routing**: identifies suspicious temporal spans and dispatches them to dimension-specific agents over a robotics failure taxonomy.
3. **Critic verification**: re-examines candidate glitches, rejects weak hypotheses, merges duplicates, refines temporal boundaries, and emits a final structured report.

<p align="center">
  <img src="assets/pipeline.png" alt="RoboGaze three-stage pipeline" width="100%">
</p>

The released implementation is intentionally transparent: intermediate JSON files and generated view clips are written to disk so that failed or ambiguous runs can be inspected.

## Table of Contents

- [Installation](#installation)
- [Serve a Local VLM](#serve-a-local-vlm)
- [Run Single-Video Inference](#run-single-video-inference)
- [Run Batch Inference](#run-batch-inference)
- [Evaluate Predictions](#evaluate-predictions)
- [Citation](#citation)

## Installation

### 1. Clone and enter the repository

```bash
git clone https://github.com/cair-vinuni/RoboGaze.git
cd RoboGaze
```

### 2. Install core dependencies

RoboGaze uses `uv` for reproducible Python environment management.

```bash
uv sync
```

To serve the local VLM from the same environment, install the optional serving dependencies:

```bash
uv sync --extra serve
```

### 3. Install system media tools

RoboGaze expects `ffmpeg` and `ffprobe` on the system path. They are used to extract video windows, frame strips, refined clips, and media metadata.

```bash
ffmpeg -version
ffprobe -version
```

## Serve a Local VLM

The pipeline talks to an OpenAI-compatible chat-completions endpoint. The provided script starts a local vLLM server and exposes the model as `gemma4` at `http://localhost:8000/v1`.

```bash
bash scripts/serve_vlm.sh
```


## Run Single-Video Inference

Set the endpoint and model name, then run `robogaze` on one video:

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

You can also use the example wrapper:

```bash
bash scripts/run_example.sh
```

## Run Batch Inference

For prepared benchmark folders, use:

```bash
bash scripts/run_all_datasets.sh
```


Useful batch options:

```bash
uv run python scripts/run_robogaze_datasets.py \
  --input-root inputs/robogaze_dataset \
  --datasets gr1_real gr1_sim droid_mv \
  --output-dir outputs/robogaze_dataset \
  --cache-dir cache/robogaze_dataset \
  --vlm-concurrency 8 \
  --limit 10
```

The batch runner writes a JSONL summary and continues after per-sample failures unless `--fail-fast` is passed.


## Evaluate Predictions

The evaluation script compares RoboGaze reports against canonical ground-truth JSON files. It computes per-clip metrics, per-dimension metrics, temporal IoU, text-similarity matching, coverage, and clean-clip detection summaries.

```bash
export GEMINI_API_KEY=<your_key>

uv run python scripts/evaluate.py \
  --pred-dir outputs/robogaze_dataset \
  --gt-dir ground_truth \
  --out-dir report/robogaze_eval \
  --model gemini-3.1-flash-lite
```


## Citation

If you find RoboGaze useful for your research, please cite:

```bibtex
@misc{nguyen2026robogaze,
  title   = {RoboGaze: Evaluating Robot World Models via
             Structured Vision-Language Analysis},
  author  = {Nguyen, Minh-Loi and Diep, Nghiem Tuong and Nguyen, Hung Khang and
             Le, Minh and Le Thien, Doanh and Tran, Hoang H. and Le, Dung D. and
             Duong, Vu N. and Sonntag, Daniel and Le, An Thai and Nguyen, Duy Minh Ho and
             Ngo, Vien Anh and Nhiem, Tran Van},
  year    = {2026},
  eprint  = {2606.28385},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  doi     = {10.48550/arXiv.2606.28385},
  url     = {https://arxiv.org/abs/2606.28385}
}
```
