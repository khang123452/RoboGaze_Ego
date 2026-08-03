"""RoboGaze-Ego glitch taxonomy and deterministic routing maps.

Adapted from the original RoboGaze robot-generation taxonomy for real
egocentric (first-person) human hand-manipulation footage sourced against
the VinRobotics Egocentric Manipulation Data technical specification. Two
changes from the upstream taxonomy:

- `robot_body_consistency` is replaced by `hand_body_consistency`, since
  there is no generated robot embodiment to hallucinate/deform -- the
  concern instead is whether the human hands are trackable, correctly
  identified, and consistent with the required-effector constraint.
- A new `spec_compliance` dimension checks each clip against the binding
  acquisition requirements in the tech spec (hand visibility, idle-frame
  ratio, activity density, environment/object-type constraints, background
  motion, camera framing) that have no analogue in a generation-QA tool.
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
    "physical_plausibility": [
        "object_teleportation",
        "object_floating",
        "object_penetration",
        "impossible_motion",
        "grasp_without_visible_support",
    ],
    "visual_quality": [
        "motion_blur",
        "exposure_or_white_balance_issue",
        "encoding_or_resolution_artifact",
        "camera_instability",
        "frame_corruption",
    ],
    "spec_compliance": [
        "hands_not_both_visible",
        "excessive_idle_time",
        "non_rigid_object_manipulation",
        "disallowed_environment_or_background_motion",
        "camera_framing_violation",
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
    "physical_plausibility": ["physical_plausibility"],
    "visual_quality": ["visual_quality"],
    "spec_compliance": ["spec_compliance"],
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
    "object_teleportation": 5,
    "object_floating": 4,
    "object_penetration": 4,
    "impossible_motion": 4,
    "grasp_without_visible_support": 4,
    "motion_blur": 2,
    "exposure_or_white_balance_issue": 2,
    "encoding_or_resolution_artifact": 3,
    "camera_instability": 2,
    "frame_corruption": 3,
    "hands_not_both_visible": 4,
    "excessive_idle_time": 3,
    "non_rigid_object_manipulation": 3,
    "disallowed_environment_or_background_motion": 3,
    "camera_framing_violation": 3,
}


def dimension_for_glitch_type(glitch_type: str) -> str | None:
    return GLITCH_TYPE_TO_DIMENSION.get(glitch_type)


def valid_glitch_type_for_dimension(glitch_type: str, dimension: str) -> bool:
    return glitch_type in GLITCH_TYPES.get(dimension, [])
