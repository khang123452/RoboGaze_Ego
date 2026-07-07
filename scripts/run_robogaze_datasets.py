#!/usr/bin/env python3
"""Run RoboGaze over prepared dataset folders."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robogaze.pipeline import RoboGaze


DATASETS = ("gr1_real", "gr1_sim", "droid_mv")
FRAME_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class Sample:
    dataset: str
    file_name: str
    task_instruction: str
    video_path: Path
    initial_frame_path: Path
    metadata: dict[str, str]

    @property
    def video_id(self) -> str:
        return Path(self.file_name).stem


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser()
    output_root = args.output_dir.expanduser()
    cache_root = args.cache_dir.expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_jsonl or output_root / "batch_summary.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    datasets = DATASETS if args.datasets == ["all"] else tuple(args.datasets)
    only = _parse_only(args.only)
    samples = []
    for dataset in datasets:
        rows = load_samples(input_root, dataset)
        rows = filter_samples(rows, only=only, start_index=args.start_index, limit=args.limit)
        samples.extend(rows)

    print(
        f"[batch] datasets={','.join(datasets)} samples={len(samples)} "
        f"output={output_root} cache={cache_root}",
        flush=True,
    )
    pipeline = RoboGaze(
        cache_dir=cache_root,
        max_views_per_agent=args.max_views_per_agent,
        enable_refinement=not args.no_refinement,
        min_hypothesis_confidence=args.min_hypothesis_confidence,
        min_subgoal_duration_s=args.min_subgoal_duration_s,
        vlm_concurrency=args.vlm_concurrency,
    )

    failures = 0
    completed = 0
    skipped = 0
    started_at = time.time()
    for index, sample in enumerate(samples, start=1):
        dataset_output_dir = output_root / sample.dataset
        run_dir = dataset_output_dir / sample.video_id
        report_path = run_dir / "report.json"
        if report_path.exists() and not args.overwrite:
            skipped += 1
            print(
                f"[skip] {index}/{len(samples)} {sample.dataset}/{sample.video_id} "
                f"report={report_path}",
                flush=True,
            )
            write_summary(
                summary_path,
                sample,
                "skipped",
                elapsed_s=0.0,
                output_dir=run_dir,
                report_path=report_path,
            )
            continue

        t0 = time.time()
        print(
            f"[run] {index}/{len(samples)} {sample.dataset}/{sample.video_id} "
            f"video={sample.video_path}",
            flush=True,
        )
        try:
            report = pipeline.run(
                task_instruction=sample.task_instruction,
                initial_frame=sample.initial_frame_path,
                video=sample.video_path,
                output_dir=dataset_output_dir,
                video_id=sample.video_id,
                video_layout_description=read_layout(input_root, sample.dataset),
            )
            elapsed_s = round(time.time() - t0, 2)
            completed += 1
            print(
                f"[done] {sample.dataset}/{sample.video_id} "
                f"elapsed_s={elapsed_s} glitches={len(report.glitches)}",
                flush=True,
            )
            write_summary(
                summary_path,
                sample,
                "completed",
                elapsed_s=elapsed_s,
                output_dir=run_dir,
                report_path=report_path,
                extra={"glitch_count": len(report.glitches)},
            )
        except Exception as exc:  # noqa: BLE001 - batch runner should log and continue.
            failures += 1
            elapsed_s = round(time.time() - t0, 2)
            error_path = run_dir / "batch_error.txt"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(
                "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                encoding="utf-8",
            )
            print(
                f"[failed] {sample.dataset}/{sample.video_id} "
                f"elapsed_s={elapsed_s} error={exc} trace={error_path}",
                flush=True,
            )
            write_summary(
                summary_path,
                sample,
                "failed",
                elapsed_s=elapsed_s,
                output_dir=run_dir,
                error=str(exc),
                error_path=error_path,
            )
            if args.fail_fast:
                break

    total_s = round(time.time() - started_at, 2)
    print(
        f"[batch-done] completed={completed} skipped={skipped} "
        f"failed={failures} elapsed_s={total_s} summary={summary_path}",
        flush=True,
    )
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RoboGaze over prepared datasets.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("inputs/robogaze_dataset"),
        help="Root containing gr1_real, gr1_sim, and droid_mv folders.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=("all", *DATASETS),
        help="Datasets to run. Default: all.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/robogaze_dataset"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/robogaze_dataset"))
    parser.add_argument(
        "--summary-jsonl",
        type=Path,
        default=None,
        help="Batch status output. Default: <output-dir>/batch_summary.jsonl.",
    )
    parser.add_argument("--vlm-concurrency", type=int, default=int(os.environ.get("VLM_CONCURRENCY", "8")))
    parser.add_argument("--max-views-per-agent", type=int, default=6)
    parser.add_argument("--min-hypothesis-confidence", type=float, default=0.2)
    parser.add_argument("--min-subgoal-duration-s", type=float, default=1.0)
    parser.add_argument("--no-refinement", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Rerun samples with existing report.json.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit per selected dataset.")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based start index per selected dataset.")
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated sample ids or file names to run, e.g. 4,kn_000000,episode_000255.mp4.",
    )
    return parser.parse_args()


def load_samples(input_root: Path, dataset: str) -> list[Sample]:
    dataset_root = input_root / dataset
    metadata_path = dataset_root / "metadata.csv"
    videos_dir = dataset_root / "videos"
    frames_dir = dataset_root / "conditioning_frame"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    samples: list[Sample] = []
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"file_name", "text"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{metadata_path} missing columns: {sorted(missing)}")
        for row in reader:
            file_name = (row.get("file_name") or "").strip()
            text = (row.get("text") or "").strip()
            if not file_name or not text:
                continue
            video_path = videos_dir / file_name
            initial_frame_path = find_initial_frame(frames_dir, Path(file_name).stem)
            if not video_path.exists():
                raise FileNotFoundError(f"Missing video for {dataset}/{file_name}: {video_path}")
            if initial_frame_path is None:
                raise FileNotFoundError(
                    f"Missing initial frame for {dataset}/{file_name} under {frames_dir}"
                )
            samples.append(
                Sample(
                    dataset=dataset,
                    file_name=file_name,
                    task_instruction=text,
                    video_path=video_path,
                    initial_frame_path=initial_frame_path,
                    metadata={k: v for k, v in row.items() if k is not None},
                )
            )
    return samples


def find_initial_frame(frames_dir: Path, stem: str) -> Path | None:
    for ext in FRAME_EXTENSIONS:
        path = frames_dir / f"{stem}{ext}"
        if path.exists():
            return path
    matches = sorted(frames_dir.glob(f"{stem}.*"))
    return matches[0] if matches else None


def filter_samples(
    samples: list[Sample],
    *,
    only: set[str],
    start_index: int,
    limit: int | None,
) -> list[Sample]:
    if only:
        samples = [
            sample
            for sample in samples
            if sample.video_id in only or sample.file_name in only
        ]
    if start_index:
        samples = samples[start_index:]
    if limit is not None:
        samples = samples[:limit]
    return samples


def read_layout(input_root: Path, dataset: str) -> str | None:
    path = input_root / dataset / "video_layout.txt"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def write_summary(
    path: Path,
    sample: Sample,
    status: str,
    *,
    elapsed_s: float,
    output_dir: Path,
    report_path: Path | None = None,
    error: str | None = None,
    error_path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": sample.dataset,
        "video_id": sample.video_id,
        "file_name": sample.file_name,
        "status": status,
        "elapsed_s": elapsed_s,
        "video_path": str(sample.video_path),
        "initial_frame_path": str(sample.initial_frame_path),
        "output_dir": str(output_dir),
    }
    if report_path is not None:
        payload["report_path"] = str(report_path)
    if error is not None:
        payload["error"] = error
    if error_path is not None:
        payload["error_path"] = str(error_path)
    if extra:
        payload.update(extra)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _parse_only(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


if __name__ == "__main__":
    sys.exit(main())
