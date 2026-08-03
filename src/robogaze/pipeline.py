"""Straightforward RoboGaze pipeline orchestrator."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .client import ModelClient
from .agents import (
    build_scene_memory,
    build_task_memory,
    build_window_state_memory,
    fallback_candidates,
    run_boundary_refiner,
    run_coarse_router_for_segment,
    run_specialist_agent,
    run_verifier,
)
from .reporting import (
    accepted_hypotheses_from_verifier,
    build_deterministic_events_from_verified_groups,
    verified_event_groups_from_verifier,
)
from .schemas import (
    BoundaryRefinement,
    CoarseCandidate,
    EntitySummary,
    GlitchHypothesis,
    GlitchReport,
    HandSummary,
    SHORT_ACTION_TYPES,
    SceneMemory,
    SubgoalSegment,
    TaskMemory,
    VerifiedEventGroup,
    VerifierResult,
    VideoMeta,
    ViewWindow,
    WindowStateMemory,
)
from .taxonomy import DIMENSION_TO_AGENT
from .views import (
    apply_layout_metadata,
    build_refined_views,
    build_subgoal_views,
    build_view_bank,
    select_views_for_agent,
    view_lookup,
)


class RoboGaze:
    def __init__(
        self,
        *,
        client: ModelClient | None = None,
        cache_dir: str | Path = "cache/robogaze",
        max_views_per_agent: int = 6,
        enable_refinement: bool = True,
        min_hypothesis_confidence: float = 0.2,
        min_subgoal_duration_s: float = 1.0,
        vlm_concurrency: int | None = None,
    ) -> None:
        self.client = client or ModelClient(cache_dir=cache_dir)
        self.max_views_per_agent = max_views_per_agent
        self.enable_refinement = enable_refinement
        self.min_hypothesis_confidence = min_hypothesis_confidence
        self.min_subgoal_duration_s = min_subgoal_duration_s
        self.vlm_concurrency = _resolve_vlm_concurrency(vlm_concurrency)
        self._cache_key_log: list[dict[str, Any]] = []

    def run(
        self,
        *,
        task_instruction: str,
        initial_frame: str | Path,
        video: str | Path,
        output_dir: str | Path = "outputs/robogaze",
        video_id: str | None = None,
        video_layout_description: str | None = None,
    ) -> GlitchReport:
        t0 = time.time()
        video_path = Path(video).expanduser()
        initial_frame_path = Path(initial_frame).expanduser()
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not initial_frame_path.exists():
            raise FileNotFoundError(f"Initial frame not found: {initial_frame_path}")

        vid = video_id or video_path.stem
        run_dir = Path(output_dir).expanduser() / vid
        views_dir = run_dir / "views"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._cache_key_log = []

        video_meta, view_bank = build_view_bank(
            video_path,
            initial_frame_path,
            views_dir,
        )
        video_meta = apply_layout_metadata(
            video_meta,
            layout_description=video_layout_description,
        )
        _write_json(run_dir / "video_meta.json", video_meta)

        task_memory = build_task_memory(
            task_instruction,
            client=self.client,
            initial_frame_path=initial_frame_path,
        )
        self._record_cache_key(
            run_dir,
            step="task_memory",
            label="task_memory",
        )
        scene_memory = build_scene_memory(
            initial_frame_path,
            task_memory,
            client=self.client,
        )
        self._record_cache_key(
            run_dir,
            step="scene_memory",
            label="scene_memory",
        )
        _write_json(run_dir / "task_memory.json", task_memory)
        _write_json(run_dir / "scene_memory.json", scene_memory)

        state_views = view_bank.medium_windows
        window_states = self._build_window_states(
            state_views,
            task_memory,
            scene_memory,
            video_meta,
            run_dir,
        )
        subgoal_segments = derive_subgoal_segments(window_states, task_memory, video_meta)
        _write_json(run_dir / "subgoal_segments.json", subgoal_segments)
        view_bank.subgoal_windows = build_subgoal_views(
            video_path,
            video_meta,
            subgoal_segments,
            views_dir,
            min_duration_s=self.min_subgoal_duration_s,
        )
        _write_json(run_dir / "view_bank.json", view_bank)
        _write_json(run_dir / "window_states.json", window_states)

        candidates = self._run_coarse_router(
            task_memory,
            scene_memory,
            video_meta,
            view_bank,
            subgoal_segments=subgoal_segments,
            run_dir=run_dir,
        )
        if not candidates:
            candidates = fallback_candidates(video_meta)
        candidates = _postprocess_subgoal_candidates(candidates, subgoal_segments)
        _write_json(run_dir / "coarse_candidates.json", candidates)

        hypotheses, _ = self._run_specialists(
            candidates,
            task_memory,
            scene_memory,
            subgoal_segments,
            video_meta,
            view_bank,
            next_hypothesis_index=1,
            run_dir=run_dir,
        )
        _write_json(run_dir / "hypotheses.json", hypotheses)

        views_by_id = view_lookup(view_bank)
        verifier_result = run_verifier(
            self.client,
            task_memory,
            scene_memory,
            hypotheses,
            views_by_id,
            video_meta,
        )
        if hypotheses:
            self._record_cache_key(
                run_dir,
                step="verifier",
                label="verifier",
            )
        _write_json(run_dir / "verifier.json", verifier_result)

        accepted_hypotheses = accepted_hypotheses_from_verifier(hypotheses, verifier_result)
        _write_json(run_dir / "accepted_hypotheses.json", accepted_hypotheses)

        refinements = self._run_refinements(
            accepted_hypotheses,
            verifier_result,
            scene_memory,
            video_meta,
            video_path,
            run_dir,
        )
        _write_json(run_dir / "boundary_refinements.json", list(refinements.values()))

        verified_event_groups = verified_event_groups_from_verifier(
            accepted_hypotheses,
            verifier_result,
            refinements,
        )
        _write_json(run_dir / "verified_event_groups.json", verified_event_groups)

        events, no_glitch_reason = self._build_events(
            task_memory,
            video_meta,
            hypotheses,
            verified_event_groups,
            refinements,
        )

        report = GlitchReport(
            video_id=vid,
            task_instruction=task_instruction,
            video_meta=video_meta,
            task_memory=task_memory,
            scene_memory=scene_memory,
            window_states=window_states,
            glitches=events,
            no_glitch_reason=no_glitch_reason,
        )
        _write_json(run_dir / "report.json", report)
        _write_json(
            run_dir / "run_metadata.json",
            {
                "video_id": vid,
                "elapsed_s": round(time.time() - t0, 2),
                "candidate_count": len(candidates),
                "hypothesis_count": len(hypotheses),
                "accepted_hypothesis_count": len(accepted_hypotheses),
                "verified_event_group_count": len(verified_event_groups),
                "glitch_count": len(events),
                "cache_key_count": len(self._cache_key_log),
                "config": {
                    "max_views_per_agent": self.max_views_per_agent,
                    "enable_refinement": self.enable_refinement,
                    "min_hypothesis_confidence": self.min_hypothesis_confidence,
                    "min_subgoal_duration_s": self.min_subgoal_duration_s,
                    "vlm_concurrency": self.vlm_concurrency,
                },
            },
        )
        _write_json(run_dir / "cache_keys.json", self._cache_key_log)
        return report

    def _run_specialists(
        self,
        candidates,
        task_memory,
        scene_memory,
        subgoal_segments: list[SubgoalSegment],
        video_meta,
        view_bank,
        *,
        next_hypothesis_index: int,
        run_dir: Path,
    ) -> tuple[list[GlitchHypothesis], int]:
        segments_by_id = {segment.subgoal_id: segment for segment in subgoal_segments}
        jobs: list[dict[str, Any]] = []
        for candidate in candidates:
            if not candidate.suspected_dimensions:
                continue
            for dimension in candidate.suspected_dimensions:
                for agent_name in DIMENSION_TO_AGENT.get(dimension, []):
                    selected_views = select_views_for_agent(
                        agent_name,
                        candidate,
                        view_bank,
                        max_views=self.max_views_per_agent,
                    )
                    if not selected_views:
                        continue
                    print(
                        f"[specialist] video={video_meta.video_path} "
                        f"candidate={candidate.candidate_id} "
                        f"dimension={dimension} "
                        f"agent={agent_name} "
                        f"views={len(selected_views)} "
                        f"ids={[v.id for v in selected_views]}",
                        flush=True,
                    )
                    subgoal_segment = segments_by_id.get(candidate.subgoal_id or "")
                    jobs.append(
                        {
                            "candidate": candidate,
                            "dimension": dimension,
                            "agent_name": agent_name,
                            "selected_views": selected_views,
                            "subgoal_segment": subgoal_segment,
                        }
                    )

        results: dict[int, list[GlitchHypothesis]] = {}
        with ThreadPoolExecutor(max_workers=self._worker_count(len(jobs))) as executor:
            futures = {
                executor.submit(
                    self._run_specialist_job,
                    job,
                    task_memory,
                    scene_memory,
                    video_meta,
                    index,
                ): (index, job)
                for index, job in enumerate(jobs)
            }
            for future in as_completed(futures):
                index, job = futures[future]
                worker_client, new_hypotheses = future.result()
                self._record_cache_key(
                    run_dir,
                    step="specialist_agent",
                    label=f"{job['candidate'].candidate_id}_{job['agent_name']}",
                    client=worker_client,
                    candidate_id=job["candidate"].candidate_id,
                    dimension=job["dimension"],
                    agent=job["agent_name"],
                    selected_view_ids=[v.id for v in job["selected_views"]],
                )
                results[index] = new_hypotheses

        hypotheses: list[GlitchHypothesis] = []
        for index in sorted(results):
            for hypothesis in results[index]:
                hypotheses.append(
                    hypothesis.model_copy(
                        update={"hypothesis_id": f"H{next_hypothesis_index}"}
                    )
                )
                next_hypothesis_index += 1
        return hypotheses, next_hypothesis_index

    def _run_specialist_job(
        self,
        job: dict[str, Any],
        task_memory,
        scene_memory,
        video_meta,
        index: int,
    ) -> tuple[ModelClient, list[GlitchHypothesis]]:
        worker_client = self._worker_client()
        new_hypotheses, _ = run_specialist_agent(
            worker_client,
            job["agent_name"],
            job["candidate"],
            job["selected_views"],
            task_memory,
            scene_memory,
            job["subgoal_segment"],
            video_meta,
            next_hypothesis_index=(index + 1) * 10000,
            min_confidence=self.min_hypothesis_confidence,
        )
        return worker_client, new_hypotheses

    def _run_coarse_router(
        self,
        task_memory,
        scene_memory,
        video_meta,
        view_bank,
        *,
        subgoal_segments: list[SubgoalSegment],
        run_dir: Path,
    ) -> list[CoarseCandidate]:
        results: dict[int, list[CoarseCandidate]] = {}
        with ThreadPoolExecutor(max_workers=self._worker_count(len(subgoal_segments))) as executor:
            futures = {
                executor.submit(
                    self._run_coarse_router_job,
                    task_memory,
                    scene_memory,
                    video_meta,
                    view_bank,
                    segment,
                ): (index, segment)
                for index, segment in enumerate(subgoal_segments)
            }
            for future in as_completed(futures):
                index, segment = futures[future]
                worker_client, candidates = future.result()
                self._record_cache_key(
                    run_dir,
                    step="coarse_router",
                    label=f"coarse_{segment.subgoal_id}",
                    client=worker_client,
                    subgoal_id=segment.subgoal_id,
                )
                results[index] = candidates
        out: list[CoarseCandidate] = []
        for index in sorted(results):
            out.extend(results[index])
        return out

    def _run_coarse_router_job(
        self,
        task_memory,
        scene_memory,
        video_meta,
        view_bank,
        segment: SubgoalSegment,
    ) -> tuple[ModelClient, list[CoarseCandidate]]:
        worker_client = self._worker_client()
        candidates = run_coarse_router_for_segment(
            worker_client,
            task_memory,
            scene_memory,
            video_meta,
            view_bank,
            segment,
        )
        return worker_client, candidates

    def _build_window_states(
        self,
        state_views: list[ViewWindow],
        task_memory,
        scene_memory,
        video_meta,
        run_dir: Path,
    ) -> list[WindowStateMemory]:
        results: dict[int, WindowStateMemory] = {}
        with ThreadPoolExecutor(max_workers=self._worker_count(len(state_views))) as executor:
            futures = {}
            for index, view in enumerate(state_views):
                print(
                    f"[window-state] video={video_meta.video_path} "
                    f"window={view.id} "
                    f"range={view.start_time:.2f}-{view.end_time:.2f}s",
                    flush=True,
                )
                futures[
                    executor.submit(
                        self._build_window_state_job,
                        task_memory,
                        scene_memory,
                        video_meta,
                        view,
                    )
                ] = (index, view)
            for future in as_completed(futures):
                index, view = futures[future]
                worker_client, state = future.result()
                self._record_cache_key(
                    run_dir,
                    step="window_state_memory",
                    label=view.id,
                    client=worker_client,
                    view_id=view.id,
                )
                results[index] = state
        return [results[index] for index in sorted(results)]

    def _build_window_state_job(
        self,
        task_memory,
        scene_memory,
        video_meta,
        view: ViewWindow,
    ) -> tuple[ModelClient, WindowStateMemory]:
        worker_client = self._worker_client()
        state = build_window_state_memory(
            worker_client,
            task_memory,
            scene_memory,
            video_meta,
            view,
        )
        return worker_client, state

    def _run_refinements(
        self,
        accepted_hypotheses: list[GlitchHypothesis],
        verifier_result: VerifierResult,
        scene_memory,
        video_meta,
        video_path: Path,
        run_dir: Path,
    ) -> dict[str, BoundaryRefinement]:
        if not self.enable_refinement or not accepted_hypotheses:
            return {}

        requests = {r.hypothesis_id: r for r in verifier_result.needs_refinement}
        jobs = []
        for hypothesis in accepted_hypotheses:
            if hypothesis.hypothesis_id in requests:
                span = requests[hypothesis.hypothesis_id].requested_span
            elif hypothesis.needs_refinement:
                span = (hypothesis.rough_start_time, hypothesis.rough_end_time)
            else:
                continue
            jobs.append((hypothesis, span))

        refinements: dict[str, BoundaryRefinement] = {}
        with ThreadPoolExecutor(max_workers=self._worker_count(len(jobs))) as executor:
            futures = {
                executor.submit(
                    self._run_refinement_job,
                    hypothesis,
                    span,
                    scene_memory,
                    video_meta,
                    video_path,
                    run_dir,
                ): hypothesis
                for hypothesis, span in jobs
            }
            for future in as_completed(futures):
                hypothesis = futures[future]
                worker_client, refined_views, refinement = future.result()
                if refinement is None:
                    continue
                self._record_cache_key(
                    run_dir,
                    step="boundary_refiner",
                    label=f"boundary_{hypothesis.hypothesis_id}",
                    client=worker_client,
                    hypothesis_id=hypothesis.hypothesis_id,
                    refined_view_ids=[v.id for v in refined_views],
                )
                refinements[hypothesis.hypothesis_id] = refinement
        return refinements

    def _run_refinement_job(
        self,
        hypothesis: GlitchHypothesis,
        span: tuple[float, float],
        scene_memory,
        video_meta,
        video_path: Path,
        run_dir: Path,
    ) -> tuple[ModelClient, list[ViewWindow], BoundaryRefinement | None]:
        worker_client = self._worker_client()
        refined_dir = run_dir / "views" / "refined" / hypothesis.hypothesis_id
        refined_views = build_refined_views(
            video_path,
            video_meta,
            span[0],
            span[1],
            refined_dir,
        )
        if not refined_views:
            return worker_client, [], None
        refinement = run_boundary_refiner(
            worker_client,
            hypothesis,
            refined_views,
            scene_memory,
            video_meta,
        )
        return worker_client, refined_views, refinement

    def _build_events(
        self,
        task_memory,
        video_meta,
        hypotheses: list[GlitchHypothesis],
        verified_event_groups: list[VerifiedEventGroup],
        refinements: dict[str, BoundaryRefinement],
    ):
        events, no_glitch_reason = build_deterministic_events_from_verified_groups(
            verified_event_groups,
            hypotheses,
            refinements,
            video_meta,
            task_memory,
        )
        return events, no_glitch_reason

    def _worker_count(self, job_count: int) -> int:
        return max(1, min(self.vlm_concurrency, max(1, job_count)))

    def _worker_client(self) -> ModelClient:
        fork = getattr(self.client, "fork", None)
        if callable(fork):
            return fork()
        return self.client

    def _record_cache_key(
        self,
        run_dir: Path,
        *,
        step: str,
        label: str,
        client: ModelClient | None = None,
        **metadata: Any,
    ) -> None:
        source_client = client or self.client
        key = getattr(source_client, "last_cache_key", None)
        if not key:
            return
        entry = {
            "index": len(self._cache_key_log) + 1,
            "step": step,
            "label": label,
            "cache_key": key,
            "cache_dir": str(source_client.cache_dir),
            "cache_path": str(source_client.cache_path(key).resolve()),
            "cache_hit": getattr(source_client, "last_cache_hit", None),
            "response_meta": getattr(source_client, "last_response_meta", None),
            **metadata,
        }
        self._cache_key_log.append(entry)
        _write_json(run_dir / "cache_keys.json", self._cache_key_log)


def run_pipeline(
    *,
    task_instruction: str,
    initial_frame: str | Path,
    video: str | Path,
    output_dir: str | Path = "outputs/robogaze",
    video_id: str | None = None,
    client: ModelClient | None = None,
    video_layout_description: str | None = None,
    vlm_concurrency: int | None = None,
) -> GlitchReport:
    return RoboGaze(
        client=client,
        vlm_concurrency=vlm_concurrency,
    ).run(
        task_instruction=task_instruction,
        initial_frame=initial_frame,
        video=video,
        output_dir=output_dir,
        video_id=video_id,
        video_layout_description=video_layout_description,
    )


def _resolve_vlm_concurrency(value: int | None) -> int:
    if value is None:
        raw = os.environ.get("ROBOGAZE_VLM_CONCURRENCY", "4")
        try:
            value = int(raw)
        except ValueError:
            value = 4
    return max(1, int(value))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, BaseModel):
        path.write_text(obj.model_dump_json(indent=2))
        return
    if isinstance(obj, list):
        path.write_text(json.dumps([_to_jsonable(x) for x in obj], indent=2, default=str))
        return
    path.write_text(json.dumps(_to_jsonable(obj), indent=2, default=str))


def derive_subgoal_segments(
    window_states: list[WindowStateMemory],
    task_memory: TaskMemory,
    video_meta: VideoMeta,
) -> list[SubgoalSegment]:
    """Derive subgoal spans from medium-window subgoal_status transitions."""
    if not task_memory.subgoal_checks:
        return []
    if not window_states:
        return _fallback_subgoal_segments(task_memory, video_meta)

    ordered_states = sorted(window_states, key=lambda state: (state.time_range[0], state.time_range[1]))
    out: list[SubgoalSegment] = []
    for index, check in enumerate(task_memory.subgoal_checks):
        series = [
            (
                state.window_id,
                _clamp_time(state.time_range[0], video_meta),
                _clamp_time(state.time_range[1], video_meta),
                state.subgoal_status.get(check.subgoal_id, "uncertain"),
            )
            for state in ordered_states
        ]
        active_set = {"in_progress", "achieved", "failed"}
        resolved_set = {"achieved", "failed"}
        first_active = next((i for i, item in enumerate(series) if item[3] in active_set), None)
        first_resolved = next((i for i, item in enumerate(series) if item[3] in resolved_set), None)

        if first_active is None:
            statuses = [item[3] for item in series]
            if statuses and all(status == "uncertain" for status in statuses):
                out.append(_fallback_subgoal_segment(check, index, len(task_memory.subgoal_checks), video_meta))
            else:
                out.append(
                    _make_subgoal_segment(
                        check,
                        video_meta.duration_s,
                        video_meta.duration_s,
                        "not_started",
                        "Not observed as active in any window.",
                        1.0,
                        [],
                        video_meta,
                    )
                )
            continue

        if first_resolved is None:
            status = "in_progress"
            start_time = series[first_active][1]
            end_time = series[-1][2]
            plateau = series[first_active:]
        else:
            status = series[first_resolved][3]
            start_time = series[first_active][1]
            end_time = series[first_resolved][2]
            plateau = series[first_active : first_resolved + 1]

        confidence = _subgoal_consensus(plateau, status)
        if confidence < 0.5:
            status = "uncertain"

        evidence_windows = ", ".join(item[0] for item in plateau) or "none"
        source_windows = [item[0] for item in plateau]
        out.append(
            _make_subgoal_segment(
                check,
                start_time,
                end_time,
                status,
                (
                    f"Derived from medium-window subgoal_status transitions "
                    f"over {evidence_windows}."
                ),
                confidence,
                source_windows,
                video_meta,
            )
        )
    _fill_may_contain(out)
    _fill_state_summaries(out, ordered_states)
    return out


def _subgoal_consensus(
    plateau: list[tuple[str, float, float, str]],
    status: str,
) -> float:
    if not plateau:
        return 0.0
    if status == "in_progress":
        agree = sum(1 for item in plateau if item[3] in {"in_progress", "achieved", "failed"})
    else:
        agree = sum(1 for item in plateau if item[3] in {"in_progress", status})
    return agree / len(plateau)


def _fallback_subgoal_segments(
    task_memory: TaskMemory,
    video_meta: VideoMeta,
) -> list[SubgoalSegment]:
    total = len(task_memory.subgoal_checks)
    return [
        _fallback_subgoal_segment(check, index, total, video_meta)
        for index, check in enumerate(task_memory.subgoal_checks)
    ]


def _fallback_subgoal_segment(
    check,
    index: int,
    total: int,
    video_meta: VideoMeta,
) -> SubgoalSegment:
    duration = max(0.0, video_meta.duration_s)
    start_time = duration * index / max(1, total)
    end_time = duration * (index + 1) / max(1, total)
    return _make_subgoal_segment(
        check,
        start_time,
        end_time,
        "uncertain",
        "Fallback even split because window-state evidence was unavailable or fully uncertain.",
        0.0,
        [],
        video_meta,
    )


def _make_subgoal_segment(
    check,
    start_time: float,
    end_time: float,
    status: str,
    evidence: str,
    confidence: float,
    source_windows: list[str],
    video_meta: VideoMeta,
) -> SubgoalSegment:
    start_time = _clamp_time(start_time, video_meta)
    end_time = _clamp_time(end_time, video_meta)
    if end_time < start_time:
        start_time, end_time = end_time, start_time
    start_frame = _time_to_frame(start_time, video_meta)
    end_frame = max(start_frame, _time_to_frame(end_time, video_meta))
    return SubgoalSegment(
        subgoal_id=check.subgoal_id,
        start_time=round(start_time, 4),
        end_time=round(end_time, 4),
        start_frame=start_frame,
        end_frame=end_frame,
        status=status,  # type: ignore[arg-type]
        evidence=evidence,
        confidence=max(0.0, min(1.0, confidence)),
        action_type=check.action_type,
        is_short_action=check.action_type in SHORT_ACTION_TYPES,
        source_windows=source_windows,
        may_contain_subgoals=[],
        entity_summary={},
        hand_summary=None,
    )


def _fill_may_contain(segments: list[SubgoalSegment]) -> None:
    for segment in segments:
        if segment.status == "not_started":
            continue
        segment.may_contain_subgoals = []
        for other in segments:
            if other.subgoal_id == segment.subgoal_id or other.status == "not_started":
                continue
            if max(segment.start_time, other.start_time) < min(segment.end_time, other.end_time):
                segment.may_contain_subgoals.append(other.subgoal_id)


VIS_PRIORITY = {"missing": 4, "fully_occluded": 3, "partially_occluded": 2, "uncertain": 1, "visible": 0}
LOC_PRIORITY = {"large_jump": 3, "moved": 2, "uncertain": 1, "same": 0}
IDENTITY_PRIORITY = {"changed": 3, "duplicated": 3, "uncertain": 1, "consistent": 0}
STATE_PRIORITY = {"distorted": 4, "opening": 2, "closed": 1, "opened": 1, "uncertain": 1, "unchanged": 0}
CONTACT_PRIORITY = {"yes": 4, "occluded": 3, "possible": 2, "uncertain": 1, "no": 0}
ACTION_PRIORITY = {
    "manipulating": 4,
    "contacting": 3,
    "carrying": 3,
    "releasing": 3,
    "approaching": 2,
    "uncertain": 1,
    "idle": 0,
}


def _fill_state_summaries(
    segments: list[SubgoalSegment],
    ordered_states: list[WindowStateMemory],
) -> None:
    by_window = {state.window_id: state for state in ordered_states}
    for segment in segments:
        states = [by_window[wid] for wid in segment.source_windows if wid in by_window]
        if not states:
            continue
        per_entity = {}
        for state in states:
            for entity in state.entities:
                per_entity.setdefault(entity.entity_id, []).append(entity)
        segment.entity_summary = {
            entity_id: EntitySummary(
                name=entries[0].name,
                visibility=_max_by(entries, "visibility", VIS_PRIORITY),
                location_relative_to_initial=_max_by(entries, "location_relative_to_initial", LOC_PRIORITY),
                identity_consistency=_max_by(entries, "identity_consistency", IDENTITY_PRIORITY),
                state_change=_max_by(entries, "state_change", STATE_PRIORITY),
            )
            for entity_id, entries in per_entity.items()
        }
        segment.hand_summary = HandSummary(
            visible=any(state.hands.visible for state in states),
            action=_max_by([state.hands for state in states], "action", ACTION_PRIORITY),
            contact_with_target=_max_by([state.hands for state in states], "contact_with_target", CONTACT_PRIORITY),
        )


def _max_by(items: list[Any], attr: str, priority: dict[str, int]) -> Any:
    if not items:
        return "uncertain"
    return max((getattr(item, attr, "uncertain") for item in items), key=lambda value: priority.get(value, -1))


def _clamp_time(time_s: float, video_meta: VideoMeta) -> float:
    return max(0.0, min(max(0.0, video_meta.duration_s), float(time_s)))


def _time_to_frame(time_s: float, video_meta: VideoMeta) -> int:
    if video_meta.fps <= 0 or video_meta.total_frames <= 0:
        return 0
    return max(0, min(video_meta.total_frames - 1, int(round(time_s * video_meta.fps))))


def _postprocess_subgoal_candidates(
    candidates,
    subgoal_segments: list[SubgoalSegment],
):
    """Only add deterministic candidates for not_started subgoals that lack a router row."""
    by_subgoal = {
        candidate.subgoal_id: candidate
        for candidate in candidates
        if getattr(candidate, "subgoal_id", None)
    }
    out = list(candidates)
    for segment in subgoal_segments:
        if segment.subgoal_id in by_subgoal or segment.status != "not_started":
            continue
        candidate = _candidate_from_segment(segment)
        out.append(candidate)
        by_subgoal[segment.subgoal_id] = candidate
    return out


def _candidate_from_segment(segment: SubgoalSegment) -> CoarseCandidate:
    return CoarseCandidate(
        candidate_id=f"C_{segment.subgoal_id}",
        subgoal_id=segment.subgoal_id,
        start_time=segment.start_time,
        end_time=segment.end_time,
        start_frame=segment.start_frame,
        end_frame=max(segment.start_frame, segment.end_frame),
        suspected_dimensions=["task_progress"],
        suspicion_score=0.5,
        reason=f"Subgoal {segment.subgoal_id} has status not_started and no router clip.",
        recommended_views=[],
    )


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    return obj
