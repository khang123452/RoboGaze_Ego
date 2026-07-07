"""Deterministic hypothesis selection, clustering, and fallback reporting."""

from __future__ import annotations

from .schemas import (
    BoundaryRefinement,
    GlitchEvent,
    GlitchHypothesis,
    TaskMemory,
    VerifiedEventGroup,
    VerifierResult,
    VideoMeta,
)
from .taxonomy import DEFAULT_SEVERITY_BY_TYPE
from .views import temporal_iou


def accepted_hypotheses_from_verifier(
    hypotheses: list[GlitchHypothesis],
    verifier_result: VerifierResult,
) -> list[GlitchHypothesis]:
    rejected = {r.hypothesis_id for r in verifier_result.rejected}
    accepted = set(verifier_result.accepted)

    for merge in verifier_result.merged:
        accepted.update(merge.hypothesis_ids)
    for request in verifier_result.needs_refinement:
        accepted.add(request.hypothesis_id)

    accepted.difference_update(rejected)
    return [h for h in hypotheses if h.hypothesis_id in accepted]


def verified_event_groups_from_verifier(
    accepted_hypotheses: list[GlitchHypothesis],
    verifier_result: VerifierResult,
    refinements: dict[str, BoundaryRefinement],
    *,
    iou_threshold: float = 0.3,
) -> list[VerifiedEventGroup]:
    """Convert verifier decisions into event-level groups for final reporting."""
    lookup = {h.hypothesis_id: h for h in accepted_hypotheses}
    groups: list[VerifiedEventGroup] = []
    grouped_ids: set[str] = set()
    used_group_ids: set[str] = set()

    for merge in verifier_result.merged:
        hypotheses = [
            lookup[hypothesis_id]
            for hypothesis_id in _dedupe(merge.hypothesis_ids)
            if hypothesis_id in lookup and hypothesis_id not in grouped_ids
        ]
        if not hypotheses:
            continue

        primary = (
            _primary_hypothesis(hypotheses, merge.primary_hypothesis_id)
            if merge.primary_hypothesis_id
            else hypotheses[0]
        )
        group_id = _unique_group_id(merge.new_group_id, used_group_ids)
        groups.append(
            _verified_group(
                group_id,
                hypotheses,
                primary,
                reason=merge.reason,
                primary_dimension=(
                    merge.primary_dimension
                    if merge.primary_dimension == primary.dimension
                    else None
                ),
                primary_glitch_type=(
                    merge.primary_glitch_type
                    if merge.primary_glitch_type == primary.glitch_type
                    else None
                ),
            )
        )
        grouped_ids.update(h.hypothesis_id for h in hypotheses)

    ungrouped = [
        h for h in accepted_hypotheses if h.hypothesis_id not in grouped_ids
    ]
    for hypotheses in _cluster_hypotheses(ungrouped, refinements, iou_threshold):
        primary = _primary_hypothesis(hypotheses)
        group_id = _unique_group_id(primary.hypothesis_id, used_group_ids)
        reason = "Accepted by verifier."
        if len(hypotheses) > 1:
            reason = "Accepted hypotheses clustered by matching type, dimension, and temporal overlap."
        groups.append(_verified_group(group_id, hypotheses, primary, reason=reason))

    return groups


def build_deterministic_events_from_verified_groups(
    verified_groups: list[VerifiedEventGroup],
    hypotheses: list[GlitchHypothesis],
    refinements: dict[str, BoundaryRefinement],
    video_meta: VideoMeta,
    task_memory: TaskMemory,
) -> tuple[list[GlitchEvent], str | None]:
    if not verified_groups:
        return [], "No verifier event groups were accepted."

    lookup = {h.hypothesis_id: h for h in hypotheses}
    valid_groups: list[tuple[VerifiedEventGroup, list[GlitchHypothesis]]] = []
    for group in verified_groups:
        group_hypotheses = [
            lookup[hypothesis_id]
            for hypothesis_id in group.hypothesis_ids
            if hypothesis_id in lookup
        ]
        if group_hypotheses:
            valid_groups.append((group, group_hypotheses))

    if not valid_groups:
        return [], "No verifier event groups matched known hypotheses."

    valid_groups.sort(
        key=lambda item: (
            _group_start_time(item[1], refinements),
            item[0].primary_glitch_type,
            item[0].primary_dimension,
        )
    )
    events = [
        _event_from_group(
            i + 1,
            group_hypotheses,
            refinements,
            video_meta,
            task_memory,
            primary_hypothesis_id=verified_group.primary_hypothesis_id,
            primary_dimension=verified_group.primary_dimension,
            primary_glitch_type=verified_group.primary_glitch_type,
        )
        for i, (verified_group, group_hypotheses) in enumerate(valid_groups)
    ]
    return events, None


def _cluster_hypotheses(
    hypotheses: list[GlitchHypothesis],
    refinements: dict[str, BoundaryRefinement],
    iou_threshold: float,
) -> list[list[GlitchHypothesis]]:
    groups: list[list[GlitchHypothesis]] = []
    for hypothesis in sorted(
        hypotheses,
        key=lambda h: (h.glitch_type, h.dimension, _start_time(h, refinements)),
    ):
        matched = False
        for group in groups:
            if _same_event_group(hypothesis, group, refinements, iou_threshold):
                group.append(hypothesis)
                matched = True
                break
        if not matched:
            groups.append([hypothesis])
    return groups


def _same_event_group(
    hypothesis: GlitchHypothesis,
    group: list[GlitchHypothesis],
    refinements: dict[str, BoundaryRefinement],
    iou_threshold: float,
) -> bool:
    anchor = group[0]
    if hypothesis.glitch_type != anchor.glitch_type:
        return False
    if hypothesis.dimension != anchor.dimension:
        return False
    h_start, h_end = _time_span(hypothesis, refinements)
    for other in group:
        o_start, o_end = _time_span(other, refinements)
        if temporal_iou(h_start, h_end, o_start, o_end) >= iou_threshold:
            return True
    return False


def _event_from_group(
    event_index: int,
    group: list[GlitchHypothesis],
    refinements: dict[str, BoundaryRefinement],
    video_meta: VideoMeta,
    task_memory: TaskMemory,
    *,
    primary_hypothesis_id: str | None = None,
    primary_dimension: str | None = None,
    primary_glitch_type: str | None = None,
) -> GlitchEvent:
    starts = [_time_span(h, refinements)[0] for h in group]
    ends = [_time_span(h, refinements)[1] for h in group]
    start_time = max(0.0, min(starts))
    end_time = min(video_meta.duration_s, max(ends))
    start_frame = _time_to_frame(start_time, video_meta)
    end_frame = max(start_frame, _time_to_frame(end_time, video_meta))

    source_ids = [h.hypothesis_id for h in group]
    source_view_ids: list[str] = []
    evidence: list[str] = []
    confidence_values: list[float] = []
    severity_values: list[int] = []
    for h in group:
        source_view_ids.extend(h.source_view_ids)
        evidence.extend(h.evidence)
        if h.hypothesis_id in refinements:
            ref = refinements[h.hypothesis_id]
            evidence.extend(ref.boundary_evidence)
            confidence_values.append((h.confidence + ref.confidence) / 2.0)
        else:
            confidence_values.append(h.confidence)
        severity_values.append(
            h.severity_estimate
            or DEFAULT_SEVERITY_BY_TYPE.get(h.glitch_type, 3)
        )
    representative = _primary_hypothesis(group, primary_hypothesis_id)
    return GlitchEvent(
        event_id=f"E{event_index}",
        glitch_type=primary_glitch_type or representative.glitch_type,
        dimension=primary_dimension or representative.dimension,
        start_time=round(start_time, 4),
        end_time=round(end_time, 4),
        start_frame=start_frame,
        end_frame=end_frame,
        severity=max(1, min(5, max(severity_values) if severity_values else 3)),
        confidence=round(max(confidence_values) if confidence_values else 0.5, 4),
        description=representative.description,
        evidence=_dedupe(evidence),
        affected_subgoals=_affected_subgoals(group, task_memory),
        source_hypotheses=source_ids,
        source_view_ids=_dedupe(source_view_ids),
    )


def _verified_group(
    group_id: str,
    hypotheses: list[GlitchHypothesis],
    primary: GlitchHypothesis,
    *,
    reason: str,
    primary_dimension: str | None = None,
    primary_glitch_type: str | None = None,
) -> VerifiedEventGroup:
    return VerifiedEventGroup(
        group_id=group_id,
        hypothesis_ids=[h.hypothesis_id for h in hypotheses],
        primary_hypothesis_id=primary.hypothesis_id,
        primary_dimension=primary_dimension or primary.dimension,
        primary_glitch_type=primary_glitch_type or primary.glitch_type,
        reason=reason,
    )


def _primary_hypothesis(
    group: list[GlitchHypothesis],
    preferred_id: str | None = None,
) -> GlitchHypothesis:
    if preferred_id:
        for hypothesis in group:
            if hypothesis.hypothesis_id == preferred_id:
                return hypothesis
    return max(
        group,
        key=lambda h: (
            h.confidence,
            h.severity_estimate or 0,
            len(h.description),
        ),
    )


def _unique_group_id(group_id: str, used_group_ids: set[str]) -> str:
    base = str(group_id or "G").strip() or "G"
    candidate = base
    i = 2
    while candidate in used_group_ids:
        candidate = f"{base}_{i}"
        i += 1
    used_group_ids.add(candidate)
    return candidate


def _group_start_time(
    group: list[GlitchHypothesis],
    refinements: dict[str, BoundaryRefinement],
) -> float:
    return min(_time_span(h, refinements)[0] for h in group)


def _affected_subgoals(
    group: list[GlitchHypothesis],
    task_memory: TaskMemory,
) -> list[str]:
    haystack = " ".join(
        [h.description for h in group]
        + [e for h in group for e in h.evidence]
    ).lower()
    out = []
    for subgoal in task_memory.expected_subgoals:
        words = [w for w in subgoal.lower().split() if len(w) >= 4]
        if any(w in haystack for w in words):
            out.append(subgoal)
    return out


def _time_span(
    hypothesis: GlitchHypothesis,
    refinements: dict[str, BoundaryRefinement],
) -> tuple[float, float]:
    refinement = refinements.get(hypothesis.hypothesis_id)
    if refinement is not None:
        return refinement.refined_start_time, refinement.refined_end_time
    return hypothesis.rough_start_time, hypothesis.rough_end_time


def _start_time(
    hypothesis: GlitchHypothesis,
    refinements: dict[str, BoundaryRefinement],
) -> float:
    return _time_span(hypothesis, refinements)[0]


def _time_to_frame(time_s: float, meta: VideoMeta) -> int:
    if meta.fps <= 0 or meta.total_frames <= 0:
        return 0
    return max(0, min(meta.total_frames - 1, int(round(time_s * meta.fps))))


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
