"""Pydantic schemas for the RoboGaze prototype."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DimensionName = Literal[
    "task_progress",
    "instruction_consistency",
    "object_scene_consistency",
    "hand_body_consistency",
    "visual_quality",
]

ViewScale = Literal["global", "medium", "subgoal", "refined", "frame"]
ActionType = Literal[
    "approach",
    "contact",
    "grasp",
    "hold",
    "transfer",
    "place",
    "release",
    "retreat",
    "other",
]
SHORT_ACTION_TYPES: frozenset[str] = frozenset({"grasp", "release", "contact"})
EntityVisibility = Literal[
    "visible",
    "partially_occluded",
    "fully_occluded",
    "missing",
    "uncertain",
]
EntityLocationChange = Literal["same", "moved", "large_jump", "uncertain"]
EntityIdentityConsistency = Literal["consistent", "changed", "duplicated", "uncertain"]
EntityStateChange = Literal[
    "unchanged",
    "opening",
    "opened",
    "closed",
    "distorted",
    "uncertain",
]
HandAction = Literal[
    "approaching",
    "contacting",
    "manipulating",
    "carrying",
    "releasing",
    "idle",
    "uncertain",
]
ContactStatus = Literal["yes", "no", "possible", "occluded", "uncertain"]
SubgoalStatus = Literal[
    "not_started",
    "in_progress",
    "achieved",
    "failed",
    "uncertain",
]


class VideoMeta(BaseModel):
    video_path: str
    fps: float
    total_frames: int
    duration_s: float
    width: int
    height: int
    is_multiview_grid: bool = False
    layout_type: str | None = None
    layout_description: str | None = None
    inactive_regions: list[str] = Field(default_factory=list)
    layout_source: str | None = None


class ViewWindow(BaseModel):
    id: str
    scale: ViewScale
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    frame_indices: list[int]
    rows: int | None = None
    cols: int | None = None
    image_path: str
    video_path: str | None = None
    subgoal_id: str | None = None
    segment_status: str | None = None
    unpadded_time_range: tuple[float, float] | None = None


class ViewBank(BaseModel):
    global_strip: ViewWindow
    medium_windows: list[ViewWindow]
    subgoal_windows: list[ViewWindow] = Field(default_factory=list)
    initial_frame: ViewWindow | None = None
    final_frame: ViewWindow | None = None


class SubgoalCheck(BaseModel):
    subgoal_id: str
    description: str
    action_type: ActionType = "other"
    success_evidence: list[str] = Field(default_factory=list)
    failure_evidence: list[str] = Field(default_factory=list)
    allowed_variations: list[str] = Field(default_factory=list)


class TaskMemory(BaseModel):
    raw_instruction: str
    target_objects: list[str]
    target_receptacles: list[str] = []
    required_effector: str | None = None
    goal_condition: str | None = None
    explicit_constraints: list[str]
    expected_subgoals: list[str]
    subgoal_checks: list[SubgoalCheck] = Field(default_factory=list)


class SceneEntity(BaseModel):
    entity_id: str
    name: str
    role: str = "context_object"
    visual_description: str = ""
    initial_location: str = ""
    initial_state: str = "uncertain"
    visibility: str = "visible"


class HandEntity(BaseModel):
    entity_id: str
    name: str
    visible_parts: list[str] = Field(default_factory=list)
    side_identity: str = "uncertain"
    uncertainty: str = ""


class WorkspaceLayout(BaseModel):
    surface: str = ""
    camera_view: str = ""
    spatial_description: str = ""


class SceneMemory(BaseModel):
    initial_frame_path: str
    visible_objects: list[str]
    visible_hand_parts: list[str]
    workspace_description: str
    initial_scene_description: str
    scene_entities: list[SceneEntity] = Field(default_factory=list)
    hand_entities: list[HandEntity] = Field(default_factory=list)
    workspace_layout: WorkspaceLayout = Field(default_factory=WorkspaceLayout)


class WindowEntityState(BaseModel):
    entity_id: str
    name: str
    visibility: EntityVisibility = "uncertain"
    location_relative_to_initial: EntityLocationChange = "uncertain"
    identity_consistency: EntityIdentityConsistency = "uncertain"
    state_change: EntityStateChange = "uncertain"
    evidence: str = ""


class WindowHandState(BaseModel):
    visible: bool = False
    action: HandAction = "uncertain"
    contact_with_target: ContactStatus = "uncertain"


class WindowStateMemory(BaseModel):
    window_id: str
    time_range: tuple[float, float]
    frame_range: tuple[int, int]
    entities: list[WindowEntityState] = Field(default_factory=list)
    hands: WindowHandState = Field(default_factory=WindowHandState)
    subgoal_status: dict[str, SubgoalStatus] = Field(default_factory=dict)


class EntitySummary(BaseModel):
    name: str
    visibility: EntityVisibility = "uncertain"
    location_relative_to_initial: EntityLocationChange = "uncertain"
    identity_consistency: EntityIdentityConsistency = "uncertain"
    state_change: EntityStateChange = "uncertain"


class HandSummary(BaseModel):
    visible: bool = False
    action: HandAction = "uncertain"
    contact_with_target: ContactStatus = "uncertain"


class SubgoalSegment(BaseModel):
    subgoal_id: str
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    status: SubgoalStatus = "uncertain"
    evidence: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    action_type: ActionType = "other"
    is_short_action: bool = False
    source_windows: list[str] = Field(default_factory=list)
    may_contain_subgoals: list[str] = Field(default_factory=list)
    entity_summary: dict[str, EntitySummary] = Field(default_factory=dict)
    hand_summary: HandSummary | None = None


class CoarseCandidate(BaseModel):
    candidate_id: str
    subgoal_id: str | None = None
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    suspected_dimensions: list[DimensionName]
    suspicion_score: float = Field(ge=0.0, le=1.0)
    reason: str
    recommended_views: list[str]


class GlitchHypothesis(BaseModel):
    hypothesis_id: str
    source_agent: str
    glitch_type: str
    dimension: str
    rough_start_time: float
    rough_end_time: float
    rough_start_frame: int
    rough_end_frame: int
    confidence: float = Field(ge=0.0, le=1.0)
    severity_estimate: int | None = Field(default=None, ge=1, le=5)
    description: str
    evidence: list[str]
    counter_evidence: list[str] = []
    source_view_ids: list[str]
    needs_refinement: bool
    refinement_reason: str | None = None


class VerifierRejection(BaseModel):
    hypothesis_id: str
    reason: str


class VerifierMerge(BaseModel):
    new_group_id: str
    hypothesis_ids: list[str]
    primary_hypothesis_id: str | None = None
    primary_dimension: str | None = None
    primary_glitch_type: str | None = None
    reason: str


class VerifiedEventGroup(BaseModel):
    group_id: str
    hypothesis_ids: list[str]
    primary_hypothesis_id: str
    primary_dimension: str
    primary_glitch_type: str
    reason: str


class VerifierRefinementRequest(BaseModel):
    hypothesis_id: str
    requested_span: tuple[float, float]
    reason: str


class VerifierConflict(BaseModel):
    hypothesis_ids: list[str]
    description: str


class VerifierResult(BaseModel):
    accepted: list[str] = []
    rejected: list[VerifierRejection] = []
    merged: list[VerifierMerge] = []
    needs_refinement: list[VerifierRefinementRequest] = []
    conflicts: list[VerifierConflict] = []


class BoundaryRefinement(BaseModel):
    hypothesis_id: str
    refined_start_time: float
    refined_end_time: float
    refined_start_frame: int
    refined_end_frame: int
    confidence: float = Field(ge=0.0, le=1.0)
    boundary_evidence: list[str]


class GlitchEvent(BaseModel):
    event_id: str
    glitch_type: str
    dimension: str
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    severity: int = Field(ge=1, le=5)
    confidence: float = Field(ge=0.0, le=1.0)
    description: str
    evidence: list[str]
    affected_subgoals: list[str] = []
    source_hypotheses: list[str]
    source_view_ids: list[str]


class GlitchReport(BaseModel):
    video_id: str
    task_instruction: str
    video_meta: VideoMeta
    task_memory: TaskMemory
    scene_memory: SceneMemory
    window_states: list[WindowStateMemory] = Field(default_factory=list)
    glitches: list[GlitchEvent]
    no_glitch_reason: str | None = None
