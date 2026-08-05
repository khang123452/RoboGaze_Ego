"""RoboGaze-Ego glitch taxonomy and deterministic routing maps.

Adapted from the original RoboGaze robot-generation taxonomy for real
egocentric (first-person) human hand-manipulation footage. Changes from the
upstream taxonomy:

- `robot_body_consistency` is replaced by `hand_body_consistency`, since
  there is no generated robot embodiment to hallucinate/deform -- the
  concern instead is whether the human hands are trackable, correctly
  identified, and consistent with the required-effector constraint.
- `physical_plausibility` (object teleportation/floating/penetration,
  impossible motion) and `spec_compliance` (acquisition-spec conformance
  against a specific external capture spec) are dropped. Both were built
  for QA'ing generated/synthetic video or footage bound to a named
  acquisition contract; neither transfers cleanly to arbitrary real
  captured footage. Re-add `spec_compliance` with your own dataset's
  acquisition rules if you have a spec to check against.
"""

from __future__ import annotations

GLITCH_TYPES = {
    "instruction_consistency": [
        "wrong_effector",
        "wrong_object",
        "wrong_target_location",
        "wrong_action_order",
        "ignored_instruction_constraint",
    ],
    "task_progress": [
        "task_incompletion",
        "failed_grasp",
        "failed_placement",
        "premature_termination",
        "ambiguous_task_success",
    ],
    "object_scene_consistency": [
        "object_disappearance",
        "object_identity_swap",
        "object_distortion",
        "unexpected_object_appearance",
        "object_state_mislabel",
    ],
    "hand_body_consistency": [
        "hand_occluded_during_manipulation",
        "single_hand_only_during_bimanual_task",
        "hand_object_contact_ambiguous",
        "hand_pose_tracking_implausible",
        "left_right_hand_identity_confusion",
    ],
    "visual_quality": [
        "motion_blur",
        "exposure_or_white_balance_issue",
        "encoding_or_resolution_artifact",
        "camera_instability",
        "frame_corruption",
    ],
}

DIMENSIONS = list(GLITCH_TYPES)

GLITCH_TYPE_TO_DIMENSION = {
    glitch_type: dimension
    for dimension, glitch_types in GLITCH_TYPES.items()
    for glitch_type in glitch_types
}

DIMENSION_TO_AGENT = {
    "task_progress": ["task_progress"],
    "instruction_consistency": ["instruction_consistency"],
    "object_scene_consistency": ["object_scene_consistency"],
    "hand_body_consistency": ["hand_body_consistency"],
    "visual_quality": ["visual_quality"],
}

AGENT_TO_DIMENSION = {
    agent: dimension
    for dimension, agents in DIMENSION_TO_AGENT.items()
    for agent in agents
}

DEFAULT_SEVERITY_BY_TYPE = {
    "wrong_effector": 4,
    "wrong_object": 4,
    "wrong_target_location": 4,
    "wrong_action_order": 3,
    "ignored_instruction_constraint": 4,
    "task_incompletion": 4,
    "failed_grasp": 4,
    "failed_placement": 4,
    "premature_termination": 4,
    "ambiguous_task_success": 3,
    "object_disappearance": 5,
    "object_identity_swap": 4,
    "object_distortion": 3,
    "unexpected_object_appearance": 3,
    "object_state_mislabel": 3,
    "hand_occluded_during_manipulation": 3,
    "single_hand_only_during_bimanual_task": 4,
    "hand_object_contact_ambiguous": 3,
    "hand_pose_tracking_implausible": 4,
    "left_right_hand_identity_confusion": 4,
    "motion_blur": 2,
    "exposure_or_white_balance_issue": 2,
    "encoding_or_resolution_artifact": 3,
    "camera_instability": 2,
    "frame_corruption": 3,
}


def dimension_for_glitch_type(glitch_type: str) -> str | None:
    return GLITCH_TYPE_TO_DIMENSION.get(glitch_type)


def valid_glitch_type_for_dimension(glitch_type: str, dimension: str) -> bool:
    return glitch_type in GLITCH_TYPES.get(dimension, [])
