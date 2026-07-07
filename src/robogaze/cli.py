"""Command line entry point for RoboGaze."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import RoboGaze


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RoboGaze local inference.")
    parser.add_argument("--task-instruction", required=True)
    parser.add_argument("--initial-frame", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", default="outputs/robogaze")
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--cache-dir", default="cache/robogaze")
    parser.add_argument("--max-views-per-agent", type=int, default=6)
    parser.add_argument(
        "--vlm-concurrency",
        type=int,
        default=None,
        help="Max concurrent independent VLM requests. Defaults to ROBOGAZE_VLM_CONCURRENCY or 4.",
    )
    parser.add_argument("--no-refinement", action="store_true")
    parser.add_argument(
        "--video-layout-file",
        default=None,
        help="Optional text file describing a dataset-level video layout.",
    )
    args = parser.parse_args()

    pipeline = RoboGaze(
        cache_dir=args.cache_dir,
        max_views_per_agent=args.max_views_per_agent,
        enable_refinement=not args.no_refinement,
        vlm_concurrency=args.vlm_concurrency,
    )
    report = pipeline.run(
        task_instruction=args.task_instruction,
        initial_frame=args.initial_frame,
        video=args.video,
        output_dir=args.output_dir,
        video_id=args.video_id,
        video_layout_description=_read_optional_text(args.video_layout_file),
    )
    print(report.model_dump_json(indent=2))


def _read_optional_text(path: str | None) -> str | None:
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8").strip()
    return text or None


if __name__ == "__main__":
    main()
