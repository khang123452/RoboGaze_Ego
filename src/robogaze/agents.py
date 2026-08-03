"""VLM-backed task memory, scene memory, routing, agents, and verifier calls."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from PIL import Image
from pydantic import BaseModel

from .client import ModelClient, image_block, text_block, video_block
from . import prompts
from .schemas import (
    ActionType,
    BoundaryRefinement,
    CoarseCandidate,
    GlitchHypothesis,
    HandEntity,
    SceneMemory,
    SceneEntity,
    SubgoalSegment,
    SubgoalCheck,
    TaskMemory,
    VerifierConflict,
    VerifierMerge,
    VerifierRefinementRequest,
    VerifierRejection,
    VerifierResult,
    VideoMeta,
    ViewBank,
    ViewWindow,
    WindowEntityState,
    WindowHandState,
    WindowStateMemory,
    WorkspaceLayout,
)
from .taxonomy import (
    DIMENSIONS,
    GLITCH_TYPES,
    dimension_for_glitch_type,
    valid_glitch_type_for_dimension,
)
from .views import window_summary


def visual_layout_note(video_meta: VideoMeta) -> str:
    if not video_meta.is_multiview_grid or not video_meta.layout_description:
        return ""
    return (
        "MULTI-VIEW VIDEO LAYOUT:\n"
        f"{video_meta.layout_description}\n"
        "Reason across all active views. Ignore inactive black regions. "
        "If contact, placement, object identity, or left/right is ambiguous "
        "in one view, consult the other active views before concluding.\n"
    )


def _system_with_layout(system: str, video_meta: VideoMeta) -> str:
    note = visual_layout_note(video_meta)
    return f"{note}\n{system}" if note else system


def build_task_memory(
    task_instruction: str,
    *,
    client: ModelClient | None = None,
    initial_frame_path: str | Path | None = None,
) -> TaskMemory:
    fallback = heuristic_task_memory(task_instruction)
    if client is None:
        return fallback

    blocks: list[dict] = [
        text_block(
            "TASK INSTRUCTION:\n"
            + task_instruction
            + "\n\nExtract TaskMemory. If an initial frame is attached, use it only to ground object names."
        )
    ]
    if initial_frame_path is not None:
        _append_image(blocks, initial_frame_path, "INITIAL_FRAME")
    raw = client.call_json(
        system=prompts.TASK_MEMORY_SYSTEM,
        user_blocks=blocks,
        model=client.model,
    )
    return _task_memory_from_raw(raw, fallback)


def heuristic_task_memory(task_instruction: str) -> TaskMemory:
    text = task_instruction.strip()
    lower = text.lower()
    required_effector = None
    effector_match = re.search(r"\b(left|right)\s+(hand|gripper|arm)\b", lower)
    if effector_match:
        required_effector = f"{effector_match.group(1)} {effector_match.group(2)}"

    target_objects: list[str] = []
    object_patterns = [
        r"\bpick up\s+(?:the |a |an )?(.+?)(?:\s+and\b|\s+then\b|,|\.|$)",
        r"\bgrasp\s+(?:the |a |an )?(.+?)(?:\s+and\b|\s+then\b|,|\.|$)",
        r"\bmove\s+(?:the |a |an )?(.+?)(?:\s+to\b|\s+onto\b|\s+on\b|\s+in\b|,|\.|$)",
        r"\bplace\s+(?:the |a |an )?(.+?)\s+(?:on|onto|in|into|to)\b",
    ]
    for pattern in object_patterns:
        match = re.search(pattern, lower)
        if match:
            obj = _clean_phrase(match.group(1))
            if obj and obj not in target_objects and obj not in {"it", "them"}:
                target_objects.append(obj)
            break

    target_receptacles: list[str] = []
    receptacle_match = re.search(
        r"\bplace\s+(?:it|them|the .+?)\s+(?:on|onto|in|into|to)\s+(?:the |a |an )?([^,.]+)",
        lower,
    )
    if receptacle_match:
        receptacle = _clean_phrase(receptacle_match.group(1))
        if receptacle:
            target_receptacles.append(receptacle)

    explicit_constraints: list[str] = []
    if required_effector:
        explicit_constraints.append(f"{required_effector} should be used")
    for obj in target_objects:
        explicit_constraints.append(f"{obj} should be manipulated")
    for receptacle in target_receptacles:
        obj = target_objects[0] if target_objects else "target object"
        explicit_constraints.append(f"{obj} should end on {receptacle}")

    obj = target_objects[0] if target_objects else "target object"
    eff = required_effector or "hand"
    goal_condition = None
    if target_receptacles:
        goal_condition = f"{obj} should be placed on {target_receptacles[0]}"
    expected_subgoals = [
        f"{eff} approaches {obj}",
        f"{eff} grasps {obj}",
        f"{obj} is moved toward {target_receptacles[0] if target_receptacles else 'target location'}",
        f"{obj} is released on {target_receptacles[0] if target_receptacles else 'target location'}",
    ]

    return TaskMemory(
        raw_instruction=text,
        target_objects=target_objects,
        target_receptacles=target_receptacles,
        required_effector=required_effector,
        goal_condition=goal_condition,
        explicit_constraints=explicit_constraints,
        expected_subgoals=expected_subgoals,
        subgoal_checks=_default_subgoal_checks(obj),
    )


def build_scene_memory(
    initial_frame_path: str | Path,
    task_memory: TaskMemory,
    *,
    client: ModelClient | None = None,
) -> SceneMemory:
    path = str(Path(initial_frame_path).resolve())
    fallback_entities = _fallback_scene_entities(task_memory)
    fallback = SceneMemory(
        initial_frame_path=path,
        visible_objects=[entity.name for entity in fallback_entities],
        visible_hand_parts=[],
        workspace_description="",
        initial_scene_description="",
        scene_entities=fallback_entities,
        hand_entities=[],
        workspace_layout=WorkspaceLayout(),
    )
    if client is None:
        return fallback

    blocks: list[dict] = [
        text_block(
            "TASK MEMORY:\n"
            + _json(task_memory)
            + "\n\nINITIAL FRAME PATH:\n"
            + path
            + "\n\nBuild SceneMemory from the attached initial frame."
        )
    ]
    _append_image(blocks, initial_frame_path, "INITIAL_FRAME")
    raw = client.call_json(
        system=prompts.SCENE_MEMORY_SYSTEM,
        user_blocks=blocks,
        model=client.model,
    )
    return _scene_memory_from_raw(raw, fallback)


def build_window_state_memory(
    client: ModelClient | None,
    task_memory: TaskMemory,
    scene_memory: SceneMemory,
    video_meta: VideoMeta,
    window: ViewWindow,
) -> WindowStateMemory:
    fallback = _fallback_window_state(window, task_memory, scene_memory)
    if client is None:
        return fallback

    payload = {
        "task_memory": task_memory.model_dump(),
        "scene_memory": scene_memory.model_dump(),
        "video_meta": video_meta.model_dump(),
        "window": window_summary([window])[0],
    }
    layout_note = visual_layout_note(video_meta)
    media_note = "\n\nAttached video subclip follows. Use only this window for observations."
    if layout_note:
        media_note += "\n\n" + layout_note
    blocks: list[dict] = [
        text_block(
            "WINDOW STATE MEMORY INPUT:\n"
            + json.dumps(payload, indent=2)
            + media_note
        )
    ]
    _append_view(blocks, window)
    raw = client.call_json(
        system=_system_with_layout(prompts.WINDOW_STATE_MEMORY_SYSTEM, video_meta),
        user_blocks=blocks,
        model=client.model,
    )
    return _window_state_from_raw(
        raw,
        fallback,
        task_memory,
        scene_memory,
    )


def run_coarse_router(
    client: ModelClient,
    task_memory: TaskMemory,
    scene_memory: SceneMemory,
    video_meta: VideoMeta,
    view_bank: ViewBank,
    *,
    subgoal_segments: list[SubgoalSegment],
) -> list[CoarseCandidate]:
    by_subgoal = {
        view.subgoal_id: view
        for view in view_bank.subgoal_windows
        if view.subgoal_id
    }
    candidates: list[CoarseCandidate] = []
    for segment in subgoal_segments:
        candidates.extend(
            run_coarse_router_for_segment(
                client,
                task_memory,
                scene_memory,
                video_meta,
                view_bank,
                segment,
                by_subgoal=by_subgoal,
            )
        )
    return candidates


def run_coarse_router_for_segment(
    client: ModelClient,
    task_memory: TaskMemory,
    scene_memory: SceneMemory,
    video_meta: VideoMeta,
    view_bank: ViewBank,
    segment: SubgoalSegment,
    *,
    by_subgoal: dict[str | None, ViewWindow] | None = None,
) -> list[CoarseCandidate]:
    by_subgoal = by_subgoal or {
        view.subgoal_id: view
        for view in view_bank.subgoal_windows
        if view.subgoal_id
    }
    subgoal_view = by_subgoal.get(segment.subgoal_id)
    if subgoal_view is None:
        return [_coarse_candidate_from_subgoal_segment(segment)]

    views = _subgoal_router_views(view_bank, subgoal_view, segment)
    text = {
        "task_memory": task_memory.model_dump(),
        "scene_memory": scene_memory.model_dump(),
        "video_meta": video_meta.model_dump(),
        "subgoal_segment": segment.model_dump(),
        "views": window_summary(views),
    }
    blocks: list[dict] = [
        text_block("COARSE ROUTER SUBGOAL INPUT:\n" + json.dumps(text, indent=2))
    ]
    for view in views:
        _append_view(blocks, view)

    raw = client.call_json(
        system=_system_with_layout(prompts.COARSE_ROUTER_SUBGOAL_SYSTEM, video_meta),
        user_blocks=blocks,
        model=client.model,
    )
    parsed = _coarse_candidates_from_raw(raw, video_meta, allow_empty_dimensions=True)
    return _normalize_subgoal_router_candidates(parsed, segment, subgoal_view)


def _subgoal_router_views(
    view_bank: ViewBank,
    subgoal_view: ViewWindow,
    segment: SubgoalSegment,
) -> list[ViewWindow]:
    views: list[ViewWindow] = []
    subgoals = [view for view in view_bank.subgoal_windows if view.subgoal_id]
    index = next(
        (i for i, view in enumerate(subgoals) if view.subgoal_id == segment.subgoal_id),
        0,
    )
    if index == 0 and view_bank.initial_frame is not None:
        views.append(view_bank.initial_frame)
    elif index > 0:
        prior = subgoals[index - 1]
        views.append(prior)
    views.append(subgoal_view)
    if index == len(subgoals) - 1 and view_bank.final_frame is not None:
        views.append(view_bank.final_frame)
    return views


def _normalize_subgoal_router_candidates(
    candidates: list[CoarseCandidate],
    segment: SubgoalSegment,
    subgoal_view: ViewWindow,
) -> list[CoarseCandidate]:
    normalized: list[CoarseCandidate] = []
    for index, candidate in enumerate(candidates, start=1):
        if candidate.subgoal_id not in {None, segment.subgoal_id}:
            continue
        candidate.subgoal_id = segment.subgoal_id
        candidate.candidate_id = (
            f"C_{segment.subgoal_id}"
            if index == 1
            else f"C_{segment.subgoal_id}_{index}"
        )
        if not candidate.recommended_views:
            candidate.recommended_views = [subgoal_view.id]
        elif subgoal_view.id not in candidate.recommended_views:
            candidate.recommended_views.insert(0, subgoal_view.id)
        normalized.append(candidate)
    return normalized


def _coarse_candidate_from_subgoal_segment(segment: SubgoalSegment) -> CoarseCandidate:
    return CoarseCandidate(
        candidate_id=f"C_{segment.subgoal_id}",
        subgoal_id=segment.subgoal_id,
        start_time=segment.start_time,
        end_time=segment.end_time,
        start_frame=segment.start_frame,
        end_frame=max(segment.start_frame, segment.end_frame),
        suspected_dimensions=["task_progress"] if segment.status == "not_started" else [],
        suspicion_score=0.5 if segment.status == "not_started" else 0.2,
        reason=(
            f"Subgoal {segment.subgoal_id} has status not_started and no subgoal clip."
            if segment.status == "not_started"
            else f"No subgoal clip was available for {segment.subgoal_id}."
        ),
        recommended_views=[],
    )


def fallback_candidates(video_meta: VideoMeta) -> list[CoarseCandidate]:
    end_frame = max(0, video_meta.total_frames - 1)
    return [
        CoarseCandidate(
            candidate_id="C_fallback_progress",
            start_time=0.0,
            end_time=video_meta.duration_s,
            start_frame=0,
            end_frame=end_frame,
            suspected_dimensions=["task_progress"],
            suspicion_score=0.5,
            reason="Coarse router returned no candidates; task progress is always checked.",
            recommended_views=["global_strip", "final_frame"],
        ),
        CoarseCandidate(
            candidate_id="C_fallback_quality",
            start_time=0.0,
            end_time=video_meta.duration_s,
            start_frame=0,
            end_frame=end_frame,
            suspected_dimensions=["visual_quality"],
            suspicion_score=0.4,
            reason="Coarse router returned no candidates; visual quality is always checked.",
            recommended_views=["global_strip"],
        ),
    ]


def run_specialist_agent(
    client: ModelClient,
    agent_name: str,
    candidate: CoarseCandidate,
    selected_views: list[ViewWindow],
    task_memory: TaskMemory,
    scene_memory: SceneMemory,
    subgoal_segment: SubgoalSegment | None,
    video_meta: VideoMeta,
    *,
    next_hypothesis_index: int,
    min_confidence: float = 0.2,
) -> tuple[list[GlitchHypothesis], int]:
    system = _system_with_layout(prompts.SPECIALIST_SYSTEM_PROMPTS[agent_name], video_meta)
    payload = {
        "agent_name": agent_name,
        "allowed_glitch_types": GLITCH_TYPES[agent_name],
        "candidate": candidate.model_dump(),
        "video_meta": video_meta.model_dump(),
        "scene_memory": scene_memory.model_dump(),
        "selected_views": window_summary(selected_views),
    }
    if agent_name in prompts.TASK_AWARE_DIMENSIONS:
        payload["task_memory"] = task_memory.model_dump()
        if subgoal_segment is not None:
            payload["subgoal_segment"] = subgoal_segment.model_dump()
    media_note = "\n\nAttached video clips and reference frames follow. Use source_view_ids from selected_views."
    blocks: list[dict] = [
        text_block(
            "SPECIALIST AGENT INPUT:\n"
            + json.dumps(payload, indent=2)
            + media_note
        )
    ]
    for view in selected_views:
        _append_view(blocks, view)

    raw = client.call_json(
        system=system,
        user_blocks=blocks,
        model=client.model,
    )
    hypotheses: list[GlitchHypothesis] = []
    for item in raw.get("hypotheses") or []:
        hyp = _hypothesis_from_raw(
            item,
            agent_name,
            candidate,
            selected_views,
            video_meta,
            hypothesis_id=f"H{next_hypothesis_index}",
        )
        if hyp is None or hyp.confidence < min_confidence:
            continue
        hypotheses.append(hyp)
        next_hypothesis_index += 1
    return hypotheses, next_hypothesis_index


def run_verifier(
    client: ModelClient,
    task_memory: TaskMemory,
    scene_memory: SceneMemory,
    hypotheses: list[GlitchHypothesis],
    views_by_id: dict[str, ViewWindow],
    video_meta: VideoMeta,
    *,
    max_source_views: int | None = 8,
) -> VerifierResult:
    if not hypotheses:
        return VerifierResult()

    source_ids: list[str] = []
    for hyp in hypotheses:
        source_ids.extend(hyp.source_view_ids)
    source_ids = list(dict.fromkeys(source_ids))
    if max_source_views is not None and len(source_ids) > max_source_views:
        source_ids = source_ids[:max_source_views]

    payload = {
        "task_memory": task_memory.model_dump(),
        "scene_memory": scene_memory.model_dump(),
        "hypotheses": [h.model_dump() for h in hypotheses],
        "video_meta": video_meta.model_dump(),
        "source_views": window_summary([views_by_id[v] for v in source_ids if v in views_by_id]),
    }
    blocks: list[dict] = [text_block("VERIFIER INPUT:\n" + json.dumps(payload, indent=2))]
    for view_id in source_ids:
        if view_id in views_by_id:
            _append_view(blocks, views_by_id[view_id])

    raw = client.call_json(
        system=_system_with_layout(prompts.VERIFIER_SYSTEM, video_meta),
        user_blocks=blocks,
        model=client.model,
    )
    return _verifier_result_from_raw(raw, hypotheses)


def run_boundary_refiner(
    client: ModelClient,
    hypothesis: GlitchHypothesis,
    refined_views: list[ViewWindow],
    scene_memory: SceneMemory,
    video_meta: VideoMeta,
) -> BoundaryRefinement:
    payload = {
        "scene_memory": scene_memory.model_dump(),
        "video_meta": video_meta.model_dump(),
        "hypothesis": hypothesis.model_dump(),
        "refined_views": window_summary(refined_views),
    }
    blocks: list[dict] = [text_block("BOUNDARY REFINEMENT INPUT:\n" + json.dumps(payload, indent=2))]
    for view in refined_views:
        _append_view(blocks, view)
    raw = client.call_json(
        system=prompts.BOUNDARY_REFINER_SYSTEM,
        user_blocks=blocks,
        model=client.model,
    )
    return _boundary_refinement_from_raw(raw, hypothesis, video_meta)


def _task_memory_from_raw(raw: dict, fallback: TaskMemory) -> TaskMemory:
    data = fallback.model_dump()
    for key in data:
        if key in raw and raw[key] is not None:
            data[key] = raw[key]
    data["raw_instruction"] = fallback.raw_instruction
    data["target_objects"] = _string_list(data.get("target_objects")) or fallback.target_objects
    data["target_receptacles"] = _string_list(data.get("target_receptacles"))
    data["explicit_constraints"] = _string_list(data.get("explicit_constraints")) or fallback.explicit_constraints
    data["expected_subgoals"] = _string_list(data.get("expected_subgoals")) or fallback.expected_subgoals
    data["subgoal_checks"] = _subgoal_checks_from_raw(data.get("subgoal_checks"), fallback.subgoal_checks)
    data.pop("ambiguity_notes", None)
    data["required_effector"] = _optional_string(data.get("required_effector"))
    data["goal_condition"] = _optional_string(data.get("goal_condition"))
    return TaskMemory(**data)


def _scene_memory_from_raw(raw: dict, fallback: SceneMemory) -> SceneMemory:
    data = fallback.model_dump()
    for key in data:
        if key in raw and raw[key] is not None:
            data[key] = raw[key]
    data["initial_frame_path"] = fallback.initial_frame_path
    data["visible_objects"] = _string_list(data.get("visible_objects"))
    data["visible_hand_parts"] = _string_list(data.get("visible_hand_parts"))
    data["workspace_description"] = str(data.get("workspace_description") or "")
    data["initial_scene_description"] = str(data.get("initial_scene_description") or "")
    data["scene_entities"] = _scene_entities_from_raw(data.get("scene_entities"), fallback.scene_entities)
    data["hand_entities"] = _hand_entities_from_raw(data.get("hand_entities"))
    data["workspace_layout"] = _workspace_layout_from_raw(data.get("workspace_layout"), fallback.workspace_layout)
    data.pop("reference_uncertainties", None)
    data.pop("uncertainty_notes", None)
    if not data["visible_objects"]:
        data["visible_objects"] = [entity.name for entity in data["scene_entities"]]
    if not data["visible_hand_parts"]:
        parts: list[str] = []
        for hand in data["hand_entities"]:
            parts.extend(hand.visible_parts)
        data["visible_hand_parts"] = list(dict.fromkeys(parts))
    return SceneMemory(**data)


def _window_state_from_raw(
    raw: dict,
    fallback: WindowStateMemory,
    task_memory: TaskMemory,
    scene_memory: SceneMemory,
) -> WindowStateMemory:
    if isinstance(raw.get("window_state"), dict):
        raw = raw["window_state"]
    time_range = _float_pair(raw.get("time_range"), fallback.time_range)
    frame_range = _int_pair(raw.get("frame_range"), fallback.frame_range)
    subgoal_ids = [check.subgoal_id for check in task_memory.subgoal_checks]
    return WindowStateMemory(
        window_id=str(raw.get("window_id") or fallback.window_id),
        time_range=time_range,
        frame_range=frame_range,
        entities=_window_entities_from_raw(raw.get("entities"), scene_memory),
        hands=_window_hand_from_raw(raw.get("hands"), fallback.hands),
        subgoal_status=_subgoal_status_from_raw(raw.get("subgoal_status"), subgoal_ids),
    )


def _coarse_candidates_from_raw(
    raw: dict,
    meta: VideoMeta,
    *,
    allow_empty_dimensions: bool = False,
) -> list[CoarseCandidate]:
    out: list[CoarseCandidate] = []
    for i, item in enumerate(raw.get("candidates") or [], start=1):
        dims = [
            d
            for d in _string_list(item.get("suspected_dimensions"))
            if d in DIMENSIONS
        ]
        if not dims and not allow_empty_dimensions:
            continue
        start_time = _clamp_float(item.get("start_time"), 0.0, meta.duration_s, 0.0)
        end_time = _clamp_float(item.get("end_time"), 0.0, meta.duration_s, meta.duration_s)
        if end_time < start_time:
            start_time, end_time = end_time, start_time
        start_frame = _time_to_frame(start_time, meta)
        end_frame = _time_to_frame(end_time, meta)
        out.append(
            CoarseCandidate(
                candidate_id=str(item.get("candidate_id") or f"C{i}"),
                subgoal_id=_optional_string(item.get("subgoal_id")),
                start_time=start_time,
                end_time=end_time,
                start_frame=start_frame,
                end_frame=max(start_frame, end_frame),
                suspected_dimensions=dims,  # type: ignore[arg-type]
                suspicion_score=_clamp_float(item.get("suspicion_score"), 0.0, 1.0, 0.5),
                reason=str(item.get("reason") or ""),
                recommended_views=_string_list(item.get("recommended_views")),
            )
        )
    return out


def _hypothesis_from_raw(
    item: dict,
    agent_name: str,
    candidate: CoarseCandidate,
    selected_views: list[ViewWindow],
    meta: VideoMeta,
    *,
    hypothesis_id: str,
) -> GlitchHypothesis | None:
    glitch_type = str(item.get("glitch_type") or "").strip()
    if not valid_glitch_type_for_dimension(glitch_type, agent_name):
        inferred = dimension_for_glitch_type(glitch_type)
        if inferred != agent_name:
            return None

    rough_start_time = _clamp_float(
        item.get("rough_start_time"),
        0.0,
        meta.duration_s,
        candidate.start_time,
    )
    rough_end_time = _clamp_float(
        item.get("rough_end_time"),
        0.0,
        meta.duration_s,
        candidate.end_time,
    )
    if rough_end_time < rough_start_time:
        rough_start_time, rough_end_time = rough_end_time, rough_start_time

    selected_ids = {v.id for v in selected_views}
    source_view_ids = [v for v in _string_list(item.get("source_view_ids")) if v in selected_ids]
    if not source_view_ids:
        source_view_ids = [v.id for v in selected_views]

    severity = item.get("severity_estimate")
    try:
        severity_i = int(severity) if severity is not None else None
    except (TypeError, ValueError):
        severity_i = None
    if severity_i is not None:
        severity_i = max(1, min(5, severity_i))

    return GlitchHypothesis(
        hypothesis_id=hypothesis_id,
        source_agent=agent_name,
        glitch_type=glitch_type,
        dimension=agent_name,
        rough_start_time=rough_start_time,
        rough_end_time=rough_end_time,
        rough_start_frame=_time_to_frame(rough_start_time, meta),
        rough_end_frame=max(_time_to_frame(rough_start_time, meta), _time_to_frame(rough_end_time, meta)),
        confidence=_clamp_float(item.get("confidence"), 0.0, 1.0, 0.5),
        severity_estimate=severity_i,
        description=str(item.get("description") or f"{glitch_type} hypothesis"),
        evidence=_string_list(item.get("evidence")),
        counter_evidence=_string_list(item.get("counter_evidence")),
        source_view_ids=source_view_ids,
        needs_refinement=bool(item.get("needs_refinement", False)),
        refinement_reason=_optional_string(item.get("refinement_reason")),
    )


def _verifier_result_from_raw(raw: dict, hypotheses: list[GlitchHypothesis]) -> VerifierResult:
    known_ids = {h.hypothesis_id for h in hypotheses}
    accepted = [h for h in _string_list(raw.get("accepted")) if h in known_ids]
    rejected = []
    for item in raw.get("rejected") or []:
        hyp_id = str(item.get("hypothesis_id") or "")
        if hyp_id in known_ids:
            rejected.append(VerifierRejection(hypothesis_id=hyp_id, reason=str(item.get("reason") or "")))
    merged = []
    for i, item in enumerate(raw.get("merged") or [], start=1):
        ids = [h for h in _string_list(item.get("hypothesis_ids")) if h in known_ids]
        if len(ids) >= 2:
            primary_hypothesis_id = str(item.get("primary_hypothesis_id") or "")
            if primary_hypothesis_id not in ids:
                primary_hypothesis_id = ids[0]
            primary = next(h for h in hypotheses if h.hypothesis_id == primary_hypothesis_id)
            merged.append(
                VerifierMerge(
                    new_group_id=str(item.get("new_group_id") or f"G{i}"),
                    hypothesis_ids=ids,
                    primary_hypothesis_id=primary_hypothesis_id,
                    primary_dimension=primary.dimension,
                    primary_glitch_type=primary.glitch_type,
                    reason=str(item.get("reason") or ""),
                )
            )
    needs_refinement = []
    for item in raw.get("needs_refinement") or []:
        hyp_id = str(item.get("hypothesis_id") or "")
        if hyp_id not in known_ids:
            continue
        span = item.get("requested_span") or []
        try:
            requested_span = (float(span[0]), float(span[1]))
        except (TypeError, ValueError, IndexError):
            hyp = next(h for h in hypotheses if h.hypothesis_id == hyp_id)
            requested_span = (hyp.rough_start_time, hyp.rough_end_time)
        needs_refinement.append(
            VerifierRefinementRequest(
                hypothesis_id=hyp_id,
                requested_span=requested_span,
                reason=str(item.get("reason") or ""),
            )
        )
    conflicts = []
    for item in raw.get("conflicts") or []:
        ids = [h for h in _string_list(item.get("hypothesis_ids")) if h in known_ids]
        if ids:
            conflicts.append(
                VerifierConflict(
                    hypothesis_ids=ids,
                    description=str(item.get("description") or ""),
                )
            )
    # Guard: any hypothesis the VLM silently omitted is recovered into accepted.
    # This is a known LLM enumeration failure mode (forgetting to output a
    # decision for one item when handling several at once).  Defaulting to
    # accepted is the conservative choice — the hypothesis already passed the
    # specialist stage, so silence ≠ "obviously wrong".
    accounted: set[str] = set(accepted)
    accounted |= {r.hypothesis_id for r in rejected}
    accounted |= {hid for m in merged for hid in m.hypothesis_ids}
    accounted |= {r.hypothesis_id for r in needs_refinement}
    unaccounted = known_ids - accounted
    if unaccounted:
        logger.warning(
            "Verifier silently dropped %d hypothesis ID(s) — defaulting to accepted: %s",
            len(unaccounted),
            sorted(unaccounted),
        )
        accepted = accepted + sorted(unaccounted)

    return VerifierResult(
        accepted=accepted,
        rejected=rejected,
        merged=merged,
        needs_refinement=needs_refinement,
        conflicts=conflicts,
    )


def _boundary_refinement_from_raw(
    raw: dict,
    hypothesis: GlitchHypothesis,
    meta: VideoMeta,
) -> BoundaryRefinement:
    start_time = _clamp_float(
        raw.get("refined_start_time"),
        0.0,
        meta.duration_s,
        hypothesis.rough_start_time,
    )
    end_time = _clamp_float(
        raw.get("refined_end_time"),
        0.0,
        meta.duration_s,
        hypothesis.rough_end_time,
    )
    if end_time < start_time:
        start_time, end_time = end_time, start_time
    return BoundaryRefinement(
        hypothesis_id=hypothesis.hypothesis_id,
        refined_start_time=start_time,
        refined_end_time=end_time,
        refined_start_frame=_clamp_frame(
            _int_or(raw.get("refined_start_frame"), _time_to_frame(start_time, meta)),
            meta,
        ),
        refined_end_frame=max(
            _time_to_frame(start_time, meta),
            _clamp_frame(
                _int_or(raw.get("refined_end_frame"), _time_to_frame(end_time, meta)),
                meta,
            ),
        ),
        confidence=_clamp_float(raw.get("confidence"), 0.0, 1.0, hypothesis.confidence),
        boundary_evidence=_string_list(raw.get("boundary_evidence")),
    )


def _default_subgoal_checks(target_object: str) -> list[SubgoalCheck]:
    obj = target_object or "target object"
    return [
        SubgoalCheck(
            subgoal_id="S1",
            description="approach the target object",
            action_type="approach",
            success_evidence=[f"hand moves near {obj}"],
            failure_evidence=[f"hand never approaches {obj}"],
            allowed_variations=["approach direction may vary"],
        ),
        SubgoalCheck(
            subgoal_id="S2",
            description="interact with the target object",
            action_type="grasp",
            success_evidence=[
                f"hand contacts, grasps, pushes, pulls, opens, or otherwise manipulates {obj}"
            ],
            failure_evidence=["hand interacts only with a non-target object"],
            allowed_variations=["the exact manipulation strategy may vary"],
        ),
        SubgoalCheck(
            subgoal_id="S3",
            description="achieve the final goal condition",
            action_type="place",
            success_evidence=["the final visible state satisfies the instruction"],
            failure_evidence=["the final visible state does not satisfy the instruction"],
            allowed_variations=[],
        ),
    ]


def _fallback_scene_entities(task_memory: TaskMemory) -> list[SceneEntity]:
    entities: list[SceneEntity] = []
    seen: set[str] = set()
    for name in task_memory.target_objects:
        if name in seen:
            continue
        seen.add(name)
        entities.append(
            SceneEntity(
                entity_id=f"obj_{len(entities) + 1}",
                name=name,
                role="target_object",
                visual_description=name,
                initial_location="uncertain",
                initial_state="uncertain",
                visibility="uncertain",
            )
        )
    for name in task_memory.target_receptacles:
        if name in seen:
            continue
        seen.add(name)
        entities.append(
            SceneEntity(
                entity_id=f"obj_{len(entities) + 1}",
                name=name,
                role="target_receptacle",
                visual_description=name,
                initial_location="uncertain",
                initial_state="uncertain",
                visibility="uncertain",
            )
        )
    return entities


def _fallback_window_state(
    window: ViewWindow,
    task_memory: TaskMemory,
    scene_memory: SceneMemory,
) -> WindowStateMemory:
    return WindowStateMemory(
        window_id=window.id,
        time_range=(window.start_time, window.end_time),
        frame_range=(window.start_frame, window.end_frame),
        entities=[
            WindowEntityState(
                entity_id=entity.entity_id,
                name=entity.name,
                visibility="uncertain",
                evidence="No VLM observation was available for this window.",
            )
            for entity in scene_memory.scene_entities
        ],
        hands=WindowHandState(),
        subgoal_status={check.subgoal_id: "uncertain" for check in task_memory.subgoal_checks},
    )


_ALLOWED_ACTION_TYPES = {
    "approach", "contact", "grasp", "hold",
    "transfer", "place", "release", "retreat", "other",
}


def _action_type_from_description(description: str) -> ActionType:
    text = description.lower()
    if any(k in text for k in ("grasp", "pick up", "pick", "close gripper")):
        return "grasp"
    if any(k in text for k in ("release", "drop", "let go", "open gripper")):
        return "release"
    if "contact" in text or "touch" in text:
        return "contact"
    if "approach" in text:
        return "approach"
    if any(k in text for k in ("transfer", "move", "carry", "bring")):
        return "transfer"
    if any(k in text for k in ("place", "put", "set down")):
        return "place"
    if "hold" in text:
        return "hold"
    if "retreat" in text or "withdraw" in text:
        return "retreat"
    return "other"


def _subgoal_checks_from_raw(value: Any, fallback: list[SubgoalCheck]) -> list[SubgoalCheck]:
    if not isinstance(value, list):
        return fallback
    out: list[SubgoalCheck] = []
    for i, item in enumerate(value, start=1):
        if isinstance(item, dict):
            description = str(item.get("description") or "").strip()
            if not description:
                continue
            raw_action = str(item.get("action_type") or "").strip().lower()
            action_type: ActionType = (
                raw_action  # type: ignore[assignment]
                if raw_action in _ALLOWED_ACTION_TYPES
                else _action_type_from_description(description)
            )
            out.append(
                SubgoalCheck(
                    subgoal_id=str(item.get("subgoal_id") or f"S{i}"),
                    description=description,
                    action_type=action_type,
                    success_evidence=_string_list(item.get("success_evidence")),
                    failure_evidence=_string_list(item.get("failure_evidence")),
                    allowed_variations=_string_list(item.get("allowed_variations")),
                )
            )
        else:
            text = str(item).strip()
            if text:
                out.append(SubgoalCheck(
                    subgoal_id=f"S{i}",
                    description=text,
                    action_type=_action_type_from_description(text),
                ))
    return out or fallback


def _scene_entities_from_raw(value: Any, fallback: list[SceneEntity]) -> list[SceneEntity]:
    if not isinstance(value, list):
        return fallback
    out: list[SceneEntity] = []
    seen_ids: set[str] = set()
    for i, item in enumerate(value, start=1):
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            entity_id = str(item.get("entity_id") or f"obj_{i}").strip()
            if entity_id in seen_ids:
                entity_id = f"obj_{i}"
            seen_ids.add(entity_id)
            out.append(
                SceneEntity(
                    entity_id=entity_id,
                    name=name,
                    role=str(item.get("role") or "context_object"),
                    visual_description=str(item.get("visual_description") or ""),
                    initial_location=str(item.get("initial_location") or ""),
                    initial_state=str(item.get("initial_state") or "uncertain"),
                    visibility=str(item.get("visibility") or "visible"),
                )
            )
        else:
            name = str(item).strip()
            if name:
                entity_id = f"obj_{i}"
                out.append(SceneEntity(entity_id=entity_id, name=name))
    return out or fallback


def _hand_entities_from_raw(value: Any) -> list[HandEntity]:
    if not isinstance(value, list):
        return []
    out: list[HandEntity] = []
    for i, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        out.append(
            HandEntity(
                entity_id=str(item.get("entity_id") or f"hand_{i}"),
                name=str(item.get("name") or "hand"),
                visible_parts=_string_list(item.get("visible_parts")),
                side_identity=str(item.get("side_identity") or "uncertain"),
                uncertainty=str(item.get("uncertainty") or ""),
            )
        )
    return out


def _workspace_layout_from_raw(value: Any, fallback: WorkspaceLayout) -> WorkspaceLayout:
    if not isinstance(value, dict):
        return fallback
    return WorkspaceLayout(
        surface=str(value.get("surface") or ""),
        camera_view=str(value.get("camera_view") or ""),
        spatial_description=str(value.get("spatial_description") or ""),
    )


def _window_entities_from_raw(value: Any, scene_memory: SceneMemory) -> list[WindowEntityState]:
    reference_names = {entity.entity_id: entity.name for entity in scene_memory.scene_entities}
    if not isinstance(value, list):
        return [
            WindowEntityState(entity_id=entity.entity_id, name=entity.name)
            for entity in scene_memory.scene_entities
        ]
    out: list[WindowEntityState] = []
    for i, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        entity_id = str(item.get("entity_id") or f"obj_{i}")
        out.append(
            WindowEntityState(
                entity_id=entity_id,
                name=str(item.get("name") or reference_names.get(entity_id) or entity_id),
                visibility=_one_of(
                    item.get("visibility"),
                    {"visible", "partially_occluded", "fully_occluded", "missing", "uncertain"},
                    "uncertain",
                ),
                location_relative_to_initial=_one_of(
                    item.get("location_relative_to_initial"),
                    {"same", "moved", "large_jump", "uncertain"},
                    "uncertain",
                ),
                identity_consistency=_one_of(
                    item.get("identity_consistency"),
                    {"consistent", "changed", "duplicated", "uncertain"},
                    "uncertain",
                ),
                state_change=_one_of(
                    item.get("state_change"),
                    {"unchanged", "opening", "opened", "closed", "distorted", "uncertain"},
                    "uncertain",
                ),
                evidence=str(item.get("evidence") or ""),
            )
        )
    return out


def _window_hand_from_raw(value: Any, fallback: WindowHandState) -> WindowHandState:
    if not isinstance(value, dict):
        return fallback
    return WindowHandState(
        visible=_bool_or(value.get("visible"), fallback.visible),
        action=_one_of(
            value.get("action"),
            {"approaching", "contacting", "manipulating", "carrying", "releasing", "idle", "uncertain"},
            fallback.action,
        ),
        contact_with_target=_one_of(
            value.get("contact_with_target"),
            {"yes", "no", "possible", "occluded", "uncertain"},
            fallback.contact_with_target,
        ),
    )


def _subgoal_status_from_raw(value: Any, subgoal_ids: list[str]) -> dict[str, str]:
    allowed = {"not_started", "in_progress", "achieved", "failed", "uncertain"}
    if not isinstance(value, dict):
        return {subgoal_id: "uncertain" for subgoal_id in subgoal_ids}
    out = {
        str(k): _one_of(v, allowed, "uncertain")
        for k, v in value.items()
        if str(k).strip() and str(k) in subgoal_ids
    }
    for subgoal_id in subgoal_ids:
        out.setdefault(subgoal_id, "uncertain")
    return out


def _append_view(blocks: list[dict], view: ViewWindow) -> None:
    blocks.append(text_block("VIEW:\n" + json.dumps(view.model_dump(exclude_none=True), indent=2)))
    if view.video_path:
        _append_video(blocks, view.video_path, view.id)
        return
    if view.scale != "frame":
        raise FileNotFoundError(f"Video subclip missing for {view.id}: {view.video_path}")
    _append_image(blocks, view.image_path, view.id)


def _append_image(blocks: list[dict], image_path: str | Path, label: str) -> None:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found for {label}: {path}")
    img = Image.open(path).convert("RGB")
    blocks.append(image_block(img))


def _append_video(blocks: list[dict], video_path: str | Path, label: str) -> None:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found for {label}: {path}")
    blocks.append(video_block(path))


def _json(obj: BaseModel | dict[str, Any]) -> str:
    if isinstance(obj, BaseModel):
        return obj.model_dump_json(indent=2)
    return json.dumps(obj, indent=2, default=str)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, tuple):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value).strip()
    return [s] if s else []


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"none", "null", "unknown"}:
        return None
    return s


def _one_of(value: Any, allowed: set[str], default: str) -> str:
    s = str(value or "").strip()
    return s if s in allowed else default


def _float_pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return default
    return default


def _int_pair(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return (int(value[0]), int(value[1]))
        except (TypeError, ValueError):
            return default
    return default


def _bool_or(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    s = str(value).strip().lower()
    if s in {"true", "yes", "1"}:
        return True
    if s in {"false", "no", "0"}:
        return False
    return default


def _clean_phrase(text: str) -> str:
    text = re.sub(r"\b(the|a|an)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .,:;")
    return text


def _clamp_float(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        x = default
    if hi < lo:
        hi = lo
    return max(lo, min(hi, x))


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _time_to_frame(time_s: float, meta: VideoMeta) -> int:
    if meta.fps <= 0 or meta.total_frames <= 0:
        return 0
    return max(0, min(meta.total_frames - 1, int(round(time_s * meta.fps))))


def _clamp_frame(frame: int, meta: VideoMeta) -> int:
    if meta.total_frames <= 0:
        return max(0, frame)
    return max(0, min(meta.total_frames - 1, frame))
