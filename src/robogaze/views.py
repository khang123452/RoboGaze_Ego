"""Video metadata, subgoal clips, and deterministic view routing."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageDraw, ImageFont

from .schemas import CoarseCandidate, SubgoalSegment, VideoMeta, ViewBank, ViewWindow


DROID_MV_LAYOUT_DESCRIPTION = (
    "The video is a 2x2 synchronized multi-view grid. Top-left shows the "
    "robotic arm from the left side. Top-right shows another exterior "
    "left-side viewing angle. Bottom-left shows a first-person end-effector/"
    "gripper view. Bottom-right is inactive black and should be ignored."
)


def video_meta(video_path: str | Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    layout = _detect_multiview_grid(cap, width, height)
    cap.release()
    return VideoMeta(
        video_path=str(video_path),
        fps=fps,
        total_frames=total,
        duration_s=total / fps if fps else 0.0,
        width=width,
        height=height,
        **layout,
    )


def apply_layout_metadata(
    meta: VideoMeta,
    *,
    layout_description: str | None = None,
) -> VideoMeta:
    """Merge dataset-level layout metadata with pixel fallback detection.

    Pixel detection wins on a positive multi-view finding because pixels are the
    execution-video ground truth. Dataset metadata only supplies the semantic
    layout description when present.
    """
    pixel_detected = meta.is_multiview_grid and meta.layout_source == "pixel"
    metadata_detected = bool(layout_description)
    if pixel_detected:
        return meta.model_copy(
            update={
                "layout_type": meta.layout_type or "2x2_grid",
                "layout_description": layout_description or meta.layout_description,
                "inactive_regions": meta.inactive_regions,
                "layout_source": "pixel+metadata" if metadata_detected else "pixel",
            }
        )
    if metadata_detected:
        return meta.model_copy(
            update={
                "is_multiview_grid": True,
                "layout_type": "dataset_layout",
                "layout_description": layout_description,
                "inactive_regions": [],
                "layout_source": "metadata",
            }
        )
    return meta


def _detect_multiview_grid(cap: cv2.VideoCapture, width: int, height: int) -> dict[str, Any]:
    if width <= 0 or height <= 0:
        return {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return {}
    h, w = frame.shape[:2]
    if h < 2 or w < 2:
        return {}
    br = frame[h // 2 :, w // 2 :]
    if br.size == 0:
        return {}
    gray = cv2.cvtColor(br, cv2.COLOR_BGR2GRAY)
    black_ratio = float((gray < 16).mean())
    mean_value = float(gray.mean())
    aspect = width / height if height else 0.0
    if black_ratio >= 0.85 and mean_value <= 24.0 and 1.2 <= aspect <= 2.4:
        return {
            "is_multiview_grid": True,
            "layout_type": "2x2_droid_mv",
            "layout_description": DROID_MV_LAYOUT_DESCRIPTION,
            "inactive_regions": ["bottom-right"],
            "layout_source": "pixel",
        }
    return {}


def build_view_bank(
    video_path: str | Path,
    initial_frame_path: str | Path,
    out_dir: str | Path,
    *,
    target_fps: float = 4.0,
) -> tuple[VideoMeta, ViewBank]:
    meta = video_meta(video_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    global_strip = make_global_window(meta)
    medium_windows = make_windows(meta.total_frames, meta.fps, target_fps=target_fps)
    initial_frame = make_initial_frame_window(initial_frame_path, out)
    write_window_videos(video_path, [global_strip] + medium_windows, out)

    final_frame_index = max(0, meta.total_frames - 1)
    final_frame = save_single_frame(
        video_path,
        final_frame_index,
        out / "final_frame.jpg",
        view_id="final_frame",
        scale="frame",
    )

    return meta, ViewBank(
        global_strip=global_strip,
        medium_windows=medium_windows,
        initial_frame=initial_frame,
        final_frame=final_frame,
    )


def make_initial_frame_window(
    initial_frame_path: str | Path,
    out_dir: Path,
) -> ViewWindow:
    image_path = Path(initial_frame_path).expanduser().resolve()
    suffix = image_path.suffix or ".jpg"
    copied = out_dir / f"initial_frame{suffix}"
    if not copied.exists() or not _same_file(image_path, copied):
        shutil.copy2(image_path, copied)
    image_path = copied.resolve()
    return ViewWindow(
        id="initial_frame",
        scale="frame",
        start_frame=0,
        end_frame=0,
        start_time=0.0,
        end_time=0.0,
        frame_indices=[0],
        rows=1,
        cols=1,
        image_path=str(image_path),
    )


def make_global_window(
    meta: VideoMeta,
) -> ViewWindow:
    end_frame = max(0, meta.total_frames - 1)
    return ViewWindow(
        id="global_strip",
        scale="global",
        start_frame=0,
        end_frame=end_frame,
        start_time=0.0,
        end_time=_frame_time(end_frame, meta.fps),
        frame_indices=[],
        image_path="",
    )


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.samefile(b)
    except FileNotFoundError:
        return False


def make_windows(
    total_frames: int,
    fps: float,
    *,
    target_fps: float = 4.0,
) -> list[ViewWindow]:
    if total_frames <= 0:
        return []
    if fps <= 0 or target_fps <= 0:
        raise ValueError("fps and target_fps must be positive")
    step = max(1, int(round(fps / target_fps)))
    sampled = list(range(0, total_frames, step))
    return _make_scale(sampled, fps, scale="medium", prefix="M", k=8, stride=4, rows=2, cols=4)


def make_refined_windows(
    meta: VideoMeta,
    start_time: float,
    end_time: float,
    *,
    margin_s: float = 0.5,
    target_fps: float = 8.0,
    k: int = 8,
    stride: int = 3,
) -> list[ViewWindow]:
    if meta.total_frames <= 0 or meta.fps <= 0:
        return []
    lo = max(0.0, start_time - margin_s)
    hi = min(meta.duration_s, end_time + margin_s)
    start_frame = max(0, min(meta.total_frames - 1, int(round(lo * meta.fps))))
    end_frame = max(start_frame, min(meta.total_frames - 1, int(round(hi * meta.fps))))
    step = max(1, int(round(meta.fps / target_fps)))
    sampled = list(range(start_frame, end_frame + 1, step))
    if sampled[-1] != end_frame:
        sampled.append(end_frame)
    return _make_scale(
        sampled,
        meta.fps,
        scale="refined",
        prefix="R",
        k=k,
        stride=stride,
        rows=2,
        cols=4,
    )


def build_refined_views(
    video_path: str | Path,
    meta: VideoMeta,
    start_time: float,
    end_time: float,
    out_dir: str | Path,
    *,
    margin_s: float = 0.5,
    target_fps: float = 8.0,
) -> list[ViewWindow]:
    windows = make_refined_windows(
        meta,
        start_time,
        end_time,
        margin_s=margin_s,
        target_fps=target_fps,
    )
    write_window_videos(video_path, windows, out_dir)
    return windows


def make_subgoal_windows(
    meta: VideoMeta,
    segments: list[SubgoalSegment],
    *,
    min_duration_s: float = 1.0,
) -> list[ViewWindow]:
    """Create temporal windows for started/visible subgoal phases.

    Segments marked not_started intentionally do not get visual clips; downstream
    routing still receives their metadata and can route task_progress.
    """
    windows: list[ViewWindow] = []
    if meta.total_frames <= 0 or meta.fps <= 0:
        return windows
    for segment in segments:
        if segment.status == "not_started":
            continue
        start_time = max(0.0, min(meta.duration_s, segment.start_time))
        end_time = max(start_time, min(meta.duration_s, segment.end_time))
        original = (start_time, end_time)
        duration = end_time - start_time
        if duration < min_duration_s:
            pad = (min_duration_s - duration) / 2.0
            start_time = max(0.0, start_time - pad)
            end_time = min(meta.duration_s, end_time + pad)
            if end_time - start_time < min_duration_s:
                if start_time <= 0.0:
                    end_time = min(meta.duration_s, min_duration_s)
                elif end_time >= meta.duration_s:
                    start_time = max(0.0, meta.duration_s - min_duration_s)

        start_frame = _time_to_frame(start_time, meta)
        end_frame = max(start_frame, _time_to_frame(end_time, meta))
        if end_frame <= start_frame:
            frame_indices = [start_frame]
        else:
            n = min(12, end_frame - start_frame + 1)
            frame_indices = sorted(
                {
                    min(
                        end_frame,
                        int(round(start_frame + i * (end_frame - start_frame) / max(1, n - 1))),
                    )
                    for i in range(n)
                }
            )
        rows, cols = (2, 4) if len(frame_indices) <= 8 else (3, 4)
        windows.append(
            ViewWindow(
                id=f"SG_{segment.subgoal_id}",
                scale="subgoal",
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=round(start_time, 4),
                end_time=round(_frame_time(end_frame, meta.fps), 4),
                frame_indices=frame_indices,
                rows=rows,
                cols=cols,
                image_path="",
                subgoal_id=segment.subgoal_id,
                segment_status=segment.status,
                unpadded_time_range=original,
            )
        )
    return windows


def build_subgoal_views(
    video_path: str | Path,
    meta: VideoMeta,
    segments: list[SubgoalSegment],
    out_dir: str | Path,
    *,
    min_duration_s: float = 1.0,
) -> list[ViewWindow]:
    windows = make_subgoal_windows(
        meta,
        segments,
        min_duration_s=min_duration_s,
    )
    write_window_videos(video_path, windows, out_dir)
    return windows


def _make_scale(
    sampled: list[int],
    fps: float,
    *,
    scale: str,
    prefix: str,
    k: int,
    stride: int,
    rows: int,
    cols: int,
) -> list[ViewWindow]:
    if not sampled:
        return []
    if len(sampled) < k:
        starts = [0]
    else:
        starts = list(range(0, len(sampled) - k + 1, stride))
        if starts[-1] != len(sampled) - k:
            starts.append(len(sampled) - k)

    out: list[ViewWindow] = []
    for i, start in enumerate(starts):
        chunk = sampled[start : start + k]
        if len(chunk) < 2:
            continue
        out.append(
            ViewWindow(
                id=f"{prefix}{i:02d}",
                scale=scale,  # type: ignore[arg-type]
                start_frame=chunk[0],
                end_frame=chunk[-1],
                start_time=round(chunk[0] / fps, 4),
                end_time=round(chunk[-1] / fps, 4),
                frame_indices=[int(x) for x in chunk],
                rows=rows,
                cols=cols,
                image_path="",
            )
        )
    return out


def write_window_videos(
    video_path: str | Path,
    windows: list[ViewWindow],
    out_dir: str | Path,
) -> None:
    """Write one H.264 MP4 subclip per view window for video VLM input."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ffmpeg = _require_ffmpeg_with_encoder("libx264")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0 or total <= 0:
        raise RuntimeError(f"Invalid video metadata for clip export: {video_path}")

    for window in windows:
        start = max(0, min(window.start_frame, total - 1))
        end = max(start, min(window.end_frame, total - 1))
        path = out / f"{window.id}.mp4"
        _write_clip_ffmpeg(ffmpeg, video_path, path, start, end, fps)
        _validate_written_video(path, expected_frames=end - start + 1, ffmpeg=ffmpeg)
        window.video_path = str(path.resolve())


def _write_clip_ffmpeg(
    ffmpeg: str,
    video_path: str | Path,
    out_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
) -> None:
    fps_arg = _fps_arg(fps)
    vf = (
        f"trim=start_frame={start_frame}:end_frame={end_frame + 1},"
        "setpts=PTS-STARTPTS,"
        "pad=ceil(iw/2)*2:ceil(ih/2)*2,"
        "format=yuv420p"
    )
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-an",
        "-r",
        fps_arg,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed while writing {out_path}: {result.stderr.strip()}"
        )


def _validate_written_video(path: Path, *, expected_frames: int, ffmpeg: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Encoded video is missing or empty: {path}")
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Encoded video failed ffmpeg decode check: {path}: {result.stderr.strip()}"
        )

    stream = _probe_video_stream(path)
    codec = stream.get("codec_name")
    pix_fmt = stream.get("pix_fmt")
    if codec != "h264" or pix_fmt != "yuv420p":
        raise RuntimeError(
            f"Encoded video must be h264/yuv420p for VLM input: "
            f"{path} got codec={codec!r} pix_fmt={pix_fmt!r}"
        )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Encoded video cannot be opened by OpenCV: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Encoded video has no readable frames: {path}")

    sample_indices = sorted({0, max(0, total // 2), max(0, total - 1)})
    green_like = 0
    readable = 0
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        readable += 1
        if _looks_like_solid_green_frame(frame):
            green_like += 1
    cap.release()

    if readable == 0:
        raise RuntimeError(f"Encoded video has no readable sample frames: {path}")
    if green_like == readable:
        raise RuntimeError(f"Encoded video appears to decode as solid green: {path}")
    if expected_frames > 1 and total < max(1, expected_frames - 1):
        raise RuntimeError(
            f"Encoded video has too few frames: {path} got={total} expected~={expected_frames}"
        )


def _require_ffmpeg_with_encoder(encoder: str) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "RoboGaze requires ffmpeg on PATH. "
            "Install ffmpeg with libx264 support before running inference."
        )
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not inspect ffmpeg encoders: {result.stderr.strip()}")
    if encoder not in result.stdout:
        raise RuntimeError(
            f"RoboGaze requires ffmpeg encoder {encoder!r}. "
            "Install ffmpeg/x264, for example from conda-forge or an OS package "
            "built with libx264."
        )
    return ffmpeg


def _probe_video_stream(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is required to validate generated VLM subclips.")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found in generated clip: {path}")
    return streams[0]


def _looks_like_solid_green_frame(frame) -> bool:
    b, g, r = [float(x) for x in frame.reshape(-1, 3).mean(axis=0)]
    spread = float(frame.std())
    return g > 80 and g > r * 4 and g > b * 4 and spread < 35


def _fps_arg(fps: float) -> str:
    rounded = round(fps)
    if abs(fps - rounded) < 1e-6:
        return str(int(rounded))
    return f"{fps:.6f}".rstrip("0").rstrip(".")


def save_single_frame(
    video_path: str | Path,
    frame_index: int,
    out_path: str | Path,
    *,
    view_id: str,
    scale: str,
) -> ViewWindow:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 1.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_index = max(0, min(frame_index, max(0, total - 1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = _label_image(img, f"f{frame_index} {frame_index / fps:.2f}s")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=92)
    return ViewWindow(
        id=view_id,
        scale=scale,  # type: ignore[arg-type]
        start_frame=frame_index,
        end_frame=frame_index,
        start_time=round(frame_index / fps, 4),
        end_time=round(frame_index / fps, 4),
        frame_indices=[frame_index],
        rows=1,
        cols=1,
        image_path=str(out.resolve()),
    )


def select_views_for_agent(
    agent_name: str,
    candidate: CoarseCandidate,
    view_bank: ViewBank,
    *,
    max_views: int = 6,
) -> list[ViewWindow]:
    return _select_subgoal_views_for_agent(
        agent_name,
        candidate,
        view_bank,
        max_views=max_views,
    )


def _select_subgoal_views_for_agent(
    agent_name: str,
    candidate: CoarseCandidate,
    view_bank: ViewBank,
    *,
    max_views: int,
) -> list[ViewWindow]:
    pattern = SUBGOAL_MODE_VIEWS.get(agent_name, ("sg_current",))
    by_subgoal = {v.subgoal_id: v for v in view_bank.subgoal_windows if v.subgoal_id}
    ordered = [v for v in view_bank.subgoal_windows if v.subgoal_id]
    current = by_subgoal.get(candidate.subgoal_id or "")
    prior = _adjacent_subgoal_window(ordered, current, direction=-1)
    next_view = _adjacent_subgoal_window(ordered, current, direction=1)

    lookup = {
        "initial_frame": view_bank.initial_frame,
        "final_frame": view_bank.final_frame,
        "sg_prior": prior,
        "sg_current": current,
        "sg_next": next_view,
    }
    views = [lookup[name] for name in pattern]

    compact = [v for v in views if v is not None]
    if not compact:
        compact = [view_bank.global_strip]
    return _dedupe(compact)[:max_views]


SUBGOAL_MODE_VIEWS = {
    "task_progress": ("sg_current", "sg_next", "final_frame"),
    "instruction_consistency": ("initial_frame", "sg_prior", "sg_current", "sg_next"),
    "object_scene_consistency": ("initial_frame", "sg_current"),
    "physical_plausibility": ("initial_frame", "sg_current"),
    "robot_body_consistency": ("initial_frame", "sg_current"),
    "visual_quality": ("sg_current",),
}


def _adjacent_subgoal_window(
    ordered: list[ViewWindow],
    current: ViewWindow | None,
    *,
    direction: int,
) -> ViewWindow | None:
    if not ordered or current is None:
        return None
    try:
        index = next(i for i, view in enumerate(ordered) if view.id == current.id)
    except StopIteration:
        return None
    adjacent_index = index + direction
    if 0 <= adjacent_index < len(ordered):
        return ordered[adjacent_index]
    return None


def view_lookup(view_bank: ViewBank, extra_views: list[ViewWindow] | None = None) -> dict[str, ViewWindow]:
    views = [view_bank.global_strip]
    views += view_bank.medium_windows
    views += view_bank.subgoal_windows
    if view_bank.initial_frame is not None:
        views.append(view_bank.initial_frame)
    if view_bank.final_frame is not None:
        views.append(view_bank.final_frame)
    if extra_views:
        views += extra_views
    return {v.id: v for v in views}


def temporal_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def _dedupe(views: list[ViewWindow]) -> list[ViewWindow]:
    out: list[ViewWindow] = []
    seen: set[str] = set()
    for view in views:
        if view.id in seen:
            continue
        seen.add(view.id)
        out.append(view)
    return out


def _time_to_frame(time_s: float, meta: VideoMeta) -> int:
    if meta.total_frames <= 0 or meta.fps <= 0:
        return 0
    return max(0, min(meta.total_frames - 1, int(round(time_s * meta.fps))))


def _frame_time(frame: int, fps: float) -> float:
    return round(frame / fps, 4) if fps > 0 else 0.0


def _label_image(img: Image.Image, text: str) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("Arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    pad = 5
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle((0, 0, tw + 2 * pad, th + 2 * pad), fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return out


def window_summary(views: list[ViewWindow]) -> list[dict[str, Any]]:
    return [
        {
            "view_id": v.id,
            "scale": v.scale,
            "start_time": v.start_time,
            "end_time": v.end_time,
            "start_frame": v.start_frame,
            "end_frame": v.end_frame,
            "frame_indices": v.frame_indices,
            "subgoal_id": v.subgoal_id,
            "segment_status": v.segment_status,
            "unpadded_time_range": v.unpadded_time_range,
        }
        for v in views
    ]
