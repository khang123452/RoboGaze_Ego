#!/usr/bin/env python3
"""Evaluate RoboGaze predictions against canonical ground-truth JSON files.

Protocol:
  GT events + predicted events
    -> LLM judge description similarity S_ij
    -> cost = S_ij * temporal IoU * (1 + lambda_dim * same_dimension)
    -> Hungarian one-to-one matching
    -> per-clip metrics, dataset macro metrics, and clean-clip detection

This is intentionally a single-file research script. It supports only canonical
ground-truth files with a top-level ``glitches`` list.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from robogaze.taxonomy import DIMENSIONS, GLITCH_TYPE_TO_DIMENSION

try:
    from tqdm import tqdm  # type: ignore
except ImportError:

    def tqdm(items, **_kwargs):
        return items


PROMPT_VERSION = "v1"
DATASET_NAMES = {"gr1_real", "gr1_sim", "droid_mv"}

PROMPT_HEADER = """SYSTEM:
You are a strict scoring judge for an egocentric hand-manipulation video glitch evaluation. You compare a *predicted* glitch description against a *reference* (ground-truth) glitch description. Both descriptions refer to the same video. You do NOT see the video. You score based on the textual content alone and treat the reference as authoritative.

Return STRICT JSON only, no markdown fences, no preamble. Schema:
{
  "event_faithfulness": <int 0-5>,
  "specificity": <int 0-5>,
  "causal_correctness": <int 0-5>,
  "rationale": "<short string>"
}

Scoring axes:

1) event_faithfulness - does the prediction describe the same event as the reference?
   5 = same event, same affected objects/parts, same phase.
   3 = related event at similar moment but a partially different aspect.
   1 = prediction names an unrelated event.
   0 = prediction describes a different phenomenon entirely.

2) specificity - does the prediction name the correct entities, effector, and phase?
   5 = names the exact object, gripper/effector side, and phase from the reference.
   3 = names the correct broad object or phase but vague on the rest.
   1 = generic mentions only, no concrete entities.
   0 = contradicts the reference entities.

3) causal_correctness - does the prediction correctly attribute the underlying cause?
   5 = same cause attribution as the reference.
   3 = causally adjacent (e.g., reference says "object teleports", prediction says "object reappears in new position").
   1 = attributes to a different cause.
   0 = contradicts the reference cause.

Rules:
- Be strict. If unsure, score lower.
- Use integer scores only.
- Do not reward verbosity; reward correctness and specificity.
- The dimension labels are hints, not part of the score. Score the descriptions.

EXAMPLES:

Example 1 (near-perfect match):
reference_description: "The Rubik's cube undergoes severe visual corruption during the transport phase. It flickers, changes shape, and appears to teleport."
predicted_description: "The Rubik's cube exhibits visual corruption during transport. Geometry and texture flicker, and the cube morphs into a small green sphere."
JSON: {"event_faithfulness": 5, "specificity": 4, "causal_correctness": 4, "rationale": "Same event, same object, same phase; prediction adds a sphere detail consistent with reference morph."}

Example 2 (partial credit):
reference_description: "The orange ball levitates above the hand for ~1 second before falling."
predicted_description: "The hand picks up the orange ball but it is unclear whether the ball is fully grasped before lifting."
JSON: {"event_faithfulness": 3, "specificity": 3, "causal_correctness": 2, "rationale": "Same object and similar moment but reference is about levitation/no-support while prediction is about grasp ambiguity."}

Example 3 (wrong event):
reference_description: "The demonstrator uses the wrong (right) hand instead of the instructed left hand."
predicted_description: "The dragonfruit appears to penetrate the hand geometry briefly."
JSON: {"event_faithfulness": 0, "specificity": 0, "causal_correctness": 0, "rationale": "Different phenomena."}

NOW SCORE THE FOLLOWING PAIR.
"""


def main() -> int:
    args = parse_args()
    gt_dir = args.gt_dir.expanduser()
    pred_dir = args.pred_dir.expanduser()
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not gt_dir.exists():
        raise SystemExit(f"ground-truth directory not found: {gt_dir}")
    if not pred_dir.exists():
        raise SystemExit(f"prediction directory not found: {pred_dir}")

    samples, coverage_rows = load_samples(gt_dir, pred_dir)
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        samples = samples[: args.limit]

    print(f"[eval] paired {len(samples)} clips from GT={gt_dir} pred={pred_dir}")
    judge = make_judge(args) if needs_judge(samples) else None

    clip_rows = []
    dim_rows = []
    match_rows = []
    judge_rows = []

    for sample in tqdm(samples, desc="scoring"):
        C, S, iou, judge_scores = build_cost_matrix(
            sample["pred_events"],
            sample["gt_events"],
            judge,
            task_instruction=sample["task"],
            lambda_dim=args.lambda_dim,
            skip_no_overlap=not args.judge_all_pairs,
        )
        matches = hungarian_match(C)
        metrics = compute_clip_metrics(
            sample["video_id"],
            sample["pred_events"],
            sample["gt_events"],
            S,
            iou,
            matches,
        )

        clip_rows.append(make_clip_row(sample, metrics))
        for row in make_dim_rows(metrics):
            dim_rows.append(row)
        for row in make_match_rows(sample, matches, S, iou):
            match_rows.append(row)
        for row in make_judge_rows(sample, judge_scores, C, S, iou, matches):
            judge_rows.append(row)

    summary = summarize(clip_rows, dim_rows, samples, args, gt_dir, pred_dir)

    write_csv(out_dir / "coverage.csv", coverage_rows)
    write_csv(out_dir / "per_clip.csv", clip_rows)
    write_csv(out_dir / "per_dim.csv", dim_rows)
    write_csv(out_dir / "matches.csv", match_rows)
    write_csv(out_dir / "judge_log.csv", judge_rows)
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "report.md", summary)

    print(f"[eval] wrote {out_dir / 'summary.json'}")
    print(json.dumps(summary, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RoboGaze output reports.")
    parser.add_argument("--gt-dir", "--ground-truth", dest="gt_dir", type=Path, required=True)
    parser.add_argument("--pred-dir", "--outputs", dest="pred_dir", type=Path, required=True)
    parser.add_argument("--out-dir", "--out", dest="out_dir", type=Path, required=True)
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument(
        "--base-url",
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--cache-dir", type=Path, default=Path("evaluation/cache/judge"))
    parser.add_argument("--lambda-dim", type=float, default=0.25)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-ping", action="store_true")
    parser.add_argument(
        "--judge-all-pairs",
        action="store_true",
        help="Call the judge even when temporal IoU is zero.",
    )
    return parser.parse_args()


def load_samples(gt_dir: Path, pred_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gt_samples = []
    for path in sorted(gt_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            gt = load_gt(path)
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[skip-gt] {path}: {exc}", flush=True)
            continue
        gt_samples.append(gt)

    pred_by_id = {}
    pred_alias_to_id = {}
    for path in sorted(pred_dir.rglob("report.json")):
        pred = load_prediction(path, pred_dir)
        pred_id = display_prediction_id(path, pred_dir, pred["video_id"])
        pred_by_id[pred_id] = pred
        for alias in pred_aliases(path, pred_dir, pred["video_id"]):
            pred_alias_to_id.setdefault(alias, pred_id)

    samples = []
    coverage_rows = []
    used_pred_ids = set()

    for gt in sorted(gt_samples, key=lambda item: natural_key(item["video_id"])):
        pred_id = first_match(gt_aliases(gt), pred_alias_to_id)
        if pred_id is None:
            coverage_rows.append(
                coverage_row(gt["video_id"], gt["path"], None, len(gt["events"]), "")
            )
            continue

        pred = pred_by_id[pred_id]
        used_pred_ids.add(pred_id)
        samples.append(
            {
                "video_id": gt["video_id"],
                "task": gt["task"] or pred["task"],
                "gt_events": gt["events"],
                "pred_events": pred["events"],
                "gt_path": gt["path"],
                "pred_path": pred["path"],
            }
        )
        coverage_rows.append(
            coverage_row(
                gt["video_id"],
                gt["path"],
                pred["path"],
                len(gt["events"]),
                len(pred["events"]),
            )
        )

    for pred_id, pred in sorted(pred_by_id.items()):
        if pred_id in used_pred_ids:
            continue
        coverage_rows.append(
            coverage_row(pred_id, None, pred["path"], "", len(pred["events"]))
        )

    return samples, coverage_rows


def load_gt(path: Path) -> dict[str, Any]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    if "glitches" not in data:
        raise ValueError("missing top-level 'glitches' list")
    if not isinstance(data["glitches"], list):
        raise ValueError("'glitches' must be a list")

    video_id = str(data.get("video_id") or path.stem)
    events = []
    for idx, item in enumerate(data.get("glitches") or [], start=1):
        if isinstance(item, dict):
            events.append(gt_event(item, idx))

    aliases = [path.stem, video_id]
    if data.get("source_dataset") and data.get("source_video_id"):
        aliases.append(f"{data['source_dataset']}__{data['source_video_id']}")
        aliases.append(str(data["source_video_id"]))
    if "__" in video_id:
        aliases.append(video_id.split("__", 1)[1])

    return {
        "video_id": video_id,
        "task": str(data.get("task") or data.get("task_instruction") or ""),
        "events": events,
        "path": path,
        "aliases": unique(aliases),
    }


def load_prediction(path: Path, pred_dir: Path) -> dict[str, Any]:
    data = read_json(path)
    if isinstance(data, list):
        data = {"glitches": data}
    if not isinstance(data, dict):
        data = {}

    video_id = str(data.get("video_id") or video_id_from_report_path(path, pred_dir))
    meta = data.get("video_meta") or read_json_if_exists(path.parent / "video_meta.json") or {}
    events = []
    for idx, item in enumerate(data.get("glitches") or [], start=1):
        if isinstance(item, dict):
            events.append(pred_event(item, idx, meta))

    return {
        "video_id": video_id,
        "task": str(data.get("task_instruction") or data.get("task") or ""),
        "events": events,
        "path": path,
    }


def gt_event(item: dict[str, Any], idx: int) -> dict[str, Any]:
    glitch_type = str(item.get("glitch_type") or "unknown")
    return {
        "id": str(item.get("glitch_id") or item.get("event_id") or f"G{idx}"),
        "dimension": dimension_for(glitch_type, item.get("dimension")),
        "glitch_type": glitch_type,
        "start_frame": as_int(item.get("start_frame"), 0),
        "end_frame": as_int(item.get("end_frame"), 0),
        "severity": as_int(item.get("severity") or item.get("human_severity"), 0),
        "description": clean_text(item.get("note") or item.get("description")),
    }


def pred_event(item: dict[str, Any], idx: int, meta: dict[str, Any]) -> dict[str, Any]:
    glitch_type = str(item.get("glitch_type") or "unknown")
    return {
        "id": str(item.get("event_id") or f"P{idx}"),
        "dimension": dimension_for(glitch_type, item.get("dimension")),
        "glitch_type": glitch_type,
        "start_frame": event_frame(item, "start", meta),
        "end_frame": event_frame(item, "end", meta),
        "severity": as_int(item.get("severity"), 0),
        "description": clean_text(item.get("description") or item.get("note")),
    }


def make_judge(args: argparse.Namespace):
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing judge API key: set {args.api_key_env}")

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    prompt_sha = hashlib.sha256(PROMPT_HEADER.encode()).hexdigest()[:16]

    if not args.skip_ping:
        ping_judge(client, args.model)

    def score(gt: str, pred: str, task: str, gt_dim: str, pred_dim: str) -> dict[str, Any]:
        cache_key = judge_cache_key(args.model, prompt_sha, gt, pred, task, gt_dim, pred_dim)
        cache_path = cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            return read_json(cache_path)

        result = call_judge(client, args.model, gt, pred, task, gt_dim, pred_dim)
        cache_path.write_text(json.dumps(result, indent=2))
        return result

    return score


def needs_judge(samples: list[dict[str, Any]]) -> bool:
    return any(sample["pred_events"] and sample["gt_events"] for sample in samples)


def ping_judge(client: OpenAI, model: str) -> None:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return only the JSON: {\"ok\": 1}"},
                {"role": "user", "content": "ping"},
            ],
            temperature=0.0,
        )
    except Exception as exc:
        raise SystemExit(f"[judge] ping failed for model={model!r}: {exc!r}")
    if not (response.choices[0].message.content or "").strip():
        raise SystemExit(f"[judge] ping returned empty content for model={model!r}")
    print(f"[judge] ping OK model={model}")


def call_judge(
    client: OpenAI,
    model: str,
    gt_desc: str,
    pred_desc: str,
    task_instruction: str,
    gt_dim: str,
    pred_dim: str,
) -> dict[str, Any]:
    prompt = render_prompt(gt_desc, pred_desc, task_instruction, gt_dim, pred_dim)
    last_error = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a strict scoring judge. Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            parsed = extract_json(response.choices[0].message.content or "")
            return normalize_judge_score(parsed)
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"judge call failed after retries: {last_error!r}")


def build_cost_matrix(
    pred_events: list[dict[str, Any]],
    gt_events: list[dict[str, Any]],
    judge,
    *,
    task_instruction: str,
    lambda_dim: float,
    skip_no_overlap: bool,
) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[list[dict[str, Any] | None]]]:
    n_pred = len(pred_events)
    n_gt = len(gt_events)
    C = [[0.0 for _ in range(n_gt)] for _ in range(n_pred)]
    S = [[0.0 for _ in range(n_gt)] for _ in range(n_pred)]
    iou = [[0.0 for _ in range(n_gt)] for _ in range(n_pred)]
    judge_scores = [[None for _ in range(n_gt)] for _ in range(n_pred)]

    for i, pred in enumerate(pred_events):
        for j, gt in enumerate(gt_events):
            iou_ij = temporal_iou(pred, gt)
            iou[i][j] = iou_ij
            if skip_no_overlap and iou_ij == 0.0:
                continue
            score = judge(
                gt["description"],
                pred["description"],
                task_instruction,
                gt["dimension"],
                pred["dimension"],
            )
            judge_scores[i][j] = score
            S[i][j] = judge_similarity(score)
            dim_bonus = 1.0 + lambda_dim * (1.0 if pred["dimension"] == gt["dimension"] else 0.0)
            C[i][j] = S[i][j] * iou_ij * dim_bonus

    return C, S, iou, judge_scores


def hungarian_match(C: list[list[float]], eps: float = 1e-9) -> list[tuple[int, int, float]]:
    if not C or not C[0]:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise SystemExit("scipy is required for Hungarian matching. Run `uv sync`.") from exc
    row_ind, col_ind = linear_sum_assignment(C, maximize=True)
    return [
        (int(i), int(j), float(C[int(i)][int(j)]))
        for i, j in zip(row_ind, col_ind)
        if C[int(i)][int(j)] > eps
    ]


def compute_clip_metrics(
    video_id: str,
    pred_events: list[dict[str, Any]],
    gt_events: list[dict[str, Any]],
    S: list[list[float]],
    iou: list[list[float]],
    matches: list[tuple[int, int, float]],
) -> dict[str, Any]:
    n_pred = len(pred_events)
    n_gt = len(gt_events)
    n_match = len(matches)

    sum_s = sum(S[i][j] for i, j, _ in matches)
    p_desc = sum_s / n_pred if n_pred else 0.0
    r_desc = sum_s / n_gt if n_gt else 0.0
    sum_si = sum(S[i][j] * iou[i][j] for i, j, _ in matches)
    p_fi = sum_si / n_pred if n_pred else 0.0
    r_fi = sum_si / n_gt if n_gt else 0.0

    rec_num = sum(S[i][j] * gt_events[j]["severity"] for i, j, _ in matches)
    rec_den = sum(gt["severity"] for gt in gt_events)
    prec_num = sum(S[i][j] * pred_events[i]["severity"] for i, j, _ in matches)
    prec_den = sum(pred["severity"] for pred in pred_events)
    sev_p = prec_num / prec_den if prec_den else 0.0
    sev_r = rec_num / rec_den if rec_den else 0.0

    per_dim = {}
    for dim in DIMENSIONS:
        n_pred_dim = sum(1 for pred in pred_events if pred["dimension"] == dim)
        n_gt_dim = sum(1 for gt in gt_events if gt["dimension"] == dim)
        if n_pred_dim == 0 and n_gt_dim == 0:
            per_dim[dim] = math.nan
            continue
        sum_dim = sum(
            S[i][j]
            for i, j, _ in matches
            if pred_events[i]["dimension"] == dim and gt_events[j]["dimension"] == dim
        )
        p_dim = sum_dim / n_pred_dim if n_pred_dim else 0.0
        r_dim = sum_dim / n_gt_dim if n_gt_dim else 0.0
        per_dim[dim] = f1(p_dim, r_dim)

    return {
        "video_id": video_id,
        "N_pred": n_pred,
        "M_gt": n_gt,
        "A_matched": n_match,
        "P_desc": p_desc,
        "R_desc": r_desc,
        "F1_desc": f1(p_desc, r_desc),
        "mIoU": sum(iou[i][j] for i, j, _ in matches) / n_match if n_match else 0.0,
        "P_F1xIoU": p_fi,
        "R_F1xIoU": r_fi,
        "F1xIoU": f1(p_fi, r_fi),
        "sev_within_1": (
            sum(
                1
                for i, j, _ in matches
                if abs(pred_events[i]["severity"] - gt_events[j]["severity"]) <= 1
            )
            / n_match
            if n_match
            else 0.0
        ),
        "sev_weighted_P": sev_p,
        "sev_weighted_R": sev_r,
        "sev_weighted_F1": f1(sev_p, sev_r),
        "per_dim_F1_desc": per_dim,
    }


def make_clip_row(sample: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    row = {
        "video_id": sample["video_id"],
        "gt_path": str(sample["gt_path"]),
        "pred_path": str(sample["pred_path"]),
    }
    for key in [
        "N_pred",
        "M_gt",
        "A_matched",
        "P_desc",
        "R_desc",
        "F1_desc",
        "mIoU",
        "P_F1xIoU",
        "R_F1xIoU",
        "F1xIoU",
        "sev_within_1",
        "sev_weighted_P",
        "sev_weighted_R",
        "sev_weighted_F1",
    ]:
        row[key] = rounded(metrics[key])
    return row


def make_dim_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dim, value in metrics["per_dim_F1_desc"].items():
        rows.append(
            {
                "video_id": metrics["video_id"],
                "dimension": dim,
                "F1_desc": "" if is_nan(value) else round(value, 4),
            }
        )
    return rows


def make_match_rows(
    sample: dict[str, Any],
    matches: list[tuple[int, int, float]],
    S: list[list[float]],
    iou: list[list[float]],
) -> list[dict[str, Any]]:
    rows = []
    for i, j, cost in matches:
        pred = sample["pred_events"][i]
        gt = sample["gt_events"][j]
        rows.append(
            {
                "video_id": sample["video_id"],
                "pred_event_id": pred["id"],
                "gt_event_id": gt["id"],
                "pred_dimension": pred["dimension"],
                "gt_dimension": gt["dimension"],
                "pred_glitch_type": pred["glitch_type"],
                "gt_glitch_type": gt["glitch_type"],
                "pred_start_frame": pred["start_frame"],
                "pred_end_frame": pred["end_frame"],
                "gt_start_frame": gt["start_frame"],
                "gt_end_frame": gt["end_frame"],
                "pred_severity": pred["severity"],
                "gt_severity": gt["severity"],
                "S": round(S[i][j], 4),
                "IoU": round(iou[i][j], 4),
                "C": round(cost, 4),
                "pred_description": pred["description"],
                "gt_description": gt["description"],
            }
        )
    return rows


def make_judge_rows(
    sample: dict[str, Any],
    judge_scores: list[list[dict[str, Any] | None]],
    C: list[list[float]],
    S: list[list[float]],
    iou: list[list[float]],
    matches: list[tuple[int, int, float]],
) -> list[dict[str, Any]]:
    matched = {(i, j) for i, j, _ in matches}
    rows = []
    for i, pred in enumerate(sample["pred_events"]):
        for j, gt in enumerate(sample["gt_events"]):
            score = judge_scores[i][j]
            if score is None:
                continue
            rows.append(
                {
                    "video_id": sample["video_id"],
                    "pred_event_id": pred["id"],
                    "gt_event_id": gt["id"],
                    "pred_dimension": pred["dimension"],
                    "gt_dimension": gt["dimension"],
                    "pred_glitch_type": pred["glitch_type"],
                    "gt_glitch_type": gt["glitch_type"],
                    "IoU": round(iou[i][j], 4),
                    "S": round(S[i][j], 4),
                    "C": round(C[i][j], 4),
                    "event_faithfulness": score["event_faithfulness"],
                    "specificity": score["specificity"],
                    "causal_correctness": score["causal_correctness"],
                    "rationale": score.get("rationale", ""),
                    "matched": (i, j) in matched,
                }
            )
    return rows


def summarize(
    clip_rows: list[dict[str, Any]],
    dim_rows: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    gt_dir: Path,
    pred_dir: Path,
) -> dict[str, Any]:
    event_rows = [row for row in clip_rows if row["M_gt"] > 0]
    clip_det = clip_detection(samples)
    return {
        "n_clips": len(samples),
        "n_event_eval_clips": len(event_rows),
        "macro_F1_desc": rounded(mean([row["F1_desc"] for row in event_rows])),
        "macro_F1_desc_CI95": rounded_list(bootstrap_ci([row["F1_desc"] for row in event_rows])),
        "macro_mIoU": rounded(mean([row["mIoU"] for row in event_rows])),
        "macro_F1xIoU": rounded(mean([row["F1xIoU"] for row in event_rows])),
        "macro_F1xIoU_CI95": rounded_list(bootstrap_ci([row["F1xIoU"] for row in event_rows])),
        "macro_sev_within_1": rounded(mean([row["sev_within_1"] for row in event_rows])),
        "macro_sev_weighted_F1": rounded(mean([row["sev_weighted_F1"] for row in event_rows])),
        "per_dim_F1_desc": {
            dim: rounded(
                mean(
                    [
                        row["F1_desc"]
                        for row in dim_rows
                        if row["dimension"] == dim and row["F1_desc"] != ""
                    ]
                )
            )
            for dim in DIMENSIONS
        },
        "clip_detection": clip_det,
        "config": {
            "gt_dir": str(gt_dir),
            "pred_dir": str(pred_dir),
            "model": args.model,
            "lambda_dim": args.lambda_dim,
            "limit": args.limit,
            "judge_all_pairs": args.judge_all_pairs,
            "event_metric_scope": "paired clips with at least one GT event",
            "clean_clip_scope": "all paired clips",
        },
    }


def clip_detection(samples: list[dict[str, Any]]) -> dict[str, Any]:
    tn = fp = fn = tp = 0
    for sample in samples:
        gt_clean = len(sample["gt_events"]) == 0
        pred_clean = len(sample["pred_events"]) == 0
        if gt_clean and pred_clean:
            tn += 1
        elif gt_clean and not pred_clean:
            fp += 1
        elif not gt_clean and pred_clean:
            fn += 1
        else:
            tp += 1
    n_clean = tn + fp
    n_glitchy = fn + tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n_clips": len(samples),
        "n_clean_gt": n_clean,
        "n_glitchy_gt": n_glitchy,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "clean_clip_accuracy": rounded(tn / n_clean if n_clean else 0.0),
        "glitchy_clip_recall": rounded(tp / n_glitchy if n_glitchy else 0.0),
        "precision": rounded(precision),
        "recall": rounded(recall),
        "F1": rounded(f1(precision, recall)),
    }


def render_prompt(gt_desc: str, pred_desc: str, task: str, gt_dim: str, pred_dim: str) -> str:
    return (
        PROMPT_HEADER
        + "\nContext:\n"
        + f"- task_instruction: {task or '(not provided)'}\n"
        + f"- reference_dimension: {gt_dim or '(not provided)'}\n"
        + f"- predicted_dimension: {pred_dim or '(not provided)'}\n"
        + "\nreference_description:\n"
        + f"{gt_desc or '(empty)'}\n"
        + "\npredicted_description:\n"
        + f"{pred_desc or '(empty)'}\n"
        + "\nReturn JSON only.\n"
    )


def temporal_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    start = max(a["start_frame"], b["start_frame"])
    end = min(a["end_frame"], b["end_frame"])
    inter = max(0, end - start)
    union = max(a["end_frame"], b["end_frame"]) - min(a["start_frame"], b["start_frame"])
    return inter / union if union > 0 else 0.0


def display_prediction_id(path: Path, pred_dir: Path, video_id: str) -> str:
    dataset = infer_dataset(path, pred_dir)
    if dataset and not video_id.startswith(f"{dataset}__"):
        return f"{dataset}__{video_id}"
    return video_id


def video_id_from_report_path(path: Path, pred_dir: Path) -> str:
    try:
        rel_parent = path.parent.relative_to(pred_dir)
    except ValueError:
        return path.parent.name
    if len(rel_parent.parts) >= 2 and rel_parent.parts[-2] in DATASET_NAMES:
        return f"{rel_parent.parts[-2]}__{rel_parent.parts[-1]}"
    return rel_parent.parts[-1] if rel_parent.parts else path.parent.name


def pred_aliases(path: Path, pred_dir: Path, video_id: str) -> list[str]:
    dataset = infer_dataset(path, pred_dir)
    aliases = [video_id, path.parent.name, video_id_from_report_path(path, pred_dir)]
    if dataset:
        aliases.extend([f"{dataset}__{video_id}", f"{dataset}__{path.parent.name}"])
    return unique(aliases)


def gt_aliases(gt: dict[str, Any]) -> list[str]:
    return gt["aliases"]


def infer_dataset(path: Path, pred_dir: Path) -> str:
    try:
        parts = path.relative_to(pred_dir).parts
    except ValueError:
        return ""
    for part in parts[:-1]:
        if part in DATASET_NAMES:
            return part
    return pred_dir.name if pred_dir.name in DATASET_NAMES else ""


def first_match(aliases: list[str], lookup: dict[str, str]) -> str | None:
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    return None


def event_frame(item: dict[str, Any], prefix: str, meta: dict[str, Any]) -> int:
    frame = as_int(item.get(f"{prefix}_frame"), None)
    if frame is not None:
        return frame
    time_s = as_float(item.get(f"{prefix}_time"))
    fps = as_float(meta.get("fps"))
    if time_s is not None and fps is not None:
        return int(round(time_s * fps))
    return 0


def judge_similarity(score: dict[str, Any]) -> float:
    return (
        score["event_faithfulness"]
        + score["specificity"]
        + score["causal_correctness"]
    ) / 15.0


def normalize_judge_score(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_faithfulness": clamp_int(data.get("event_faithfulness"), 0, 5),
        "specificity": clamp_int(data.get("specificity"), 0, 5),
        "causal_correctness": clamp_int(data.get("causal_correctness"), 0, 5),
        "rationale": str(data.get("rationale") or ""),
    }


def judge_cache_key(model: str, prompt_sha: str, *parts: str) -> str:
    h = hashlib.sha256()
    h.update(PROMPT_VERSION.encode())
    h.update(b"|")
    h.update(prompt_sha.encode())
    h.update(b"|")
    h.update(model.encode())
    for part in parts:
        h.update(b"\x1f")
        h.update((part or "").encode("utf-8", errors="replace"))
    return h.hexdigest()


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"could not extract JSON from judge response: {text!r}")


def coverage_row(
    video_id: str,
    gt_path: Path | None,
    pred_path: Path | None,
    gt_count: int | str,
    pred_count: int | str,
) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "has_ground_truth": gt_path is not None,
        "has_prediction": pred_path is not None,
        "ground_truth_path": str(gt_path or ""),
        "prediction_path": str(pred_path or ""),
        "ground_truth_count": gt_count,
        "prediction_count": pred_count,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2))


def write_report(path: Path, summary: dict[str, Any]) -> None:
    clip = summary["clip_detection"]
    lines = [
        "# RoboGaze Evaluation",
        "",
        f"- Clips: {summary['n_clips']}",
        f"- Event-eval clips: {summary['n_event_eval_clips']}",
        f"- Macro F1_desc: {summary['macro_F1_desc']} [{summary['macro_F1_desc_CI95'][0]}, {summary['macro_F1_desc_CI95'][1]}]",
        f"- Macro mIoU: {summary['macro_mIoU']}",
        f"- Macro F1xIoU: {summary['macro_F1xIoU']} [{summary['macro_F1xIoU_CI95'][0]}, {summary['macro_F1xIoU_CI95'][1]}]",
        f"- Macro severity within 1: {summary['macro_sev_within_1']}",
        f"- Macro severity-weighted F1: {summary['macro_sev_weighted_F1']}",
        "",
        "## Clip Detection",
        "",
        f"- TN/FP/FN/TP: {clip['TN']}/{clip['FP']}/{clip['FN']}/{clip['TP']}",
        f"- Clean clip accuracy: {clip['clean_clip_accuracy']}",
        f"- Glitchy clip recall: {clip['glitchy_clip_recall']}",
        f"- Clip F1: {clip['F1']}",
    ]
    path.write_text("\n".join(lines) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return read_json(path)
    except json.JSONDecodeError:
        return None


def dimension_for(glitch_type: str, explicit: Any = None) -> str:
    return str(explicit or GLITCH_TYPE_TO_DIMENSION.get(glitch_type, "unknown"))


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def as_int(value: Any, default: int | None = 0) -> int | None:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def clamp_int(value: Any, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return lo


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def mean(values: list[float]) -> float:
    clean = [v for v in values if not is_nan(v)]
    return sum(clean) / len(clean) if clean else 0.0


def bootstrap_ci(values: list[float], n: int = 1000, seed: int = 0) -> tuple[float, float]:
    clean = [v for v in values if not is_nan(v)]
    if not clean:
        return (0.0, 0.0)
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        draw = [clean[rng.randrange(len(clean))] for _ in clean]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    return (samples[int(0.025 * n)], samples[int(0.975 * n)])


def is_nan(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def rounded(value: Any) -> Any:
    return round(value, 4) if isinstance(value, float) else value


def rounded_list(values: tuple[float, float]) -> list[float]:
    return [round(v, 4) for v in values]


def unique(values: list[Any]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


if __name__ == "__main__":
    raise SystemExit(main())
