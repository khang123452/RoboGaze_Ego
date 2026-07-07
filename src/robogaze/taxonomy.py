"""Initial RoboGaze glitch taxonomy and deterministic routing maps."""

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
        "object_hallucination",
        "object_disappearance",
        "object_identity_swap",
        "object_distortion",
        "object_color_or_shape_drift",
    ],
    "robot_body_consistency": [
        "hallucinated_robot_part",
        "missing_robot_part",
        "duplicated_arm_or_gripper",
        "robot_body_deformation",
        "left_right_robot_identity_confusion",
    ],
    "physical_plausibility": [
        "object_teleportation",
        "object_floating",
        "object_penetration",
        "impossible_motion",
        "grasp_without_visible_support",
    ],
    "visual_quality": [
        "blur",
        "occlusion",
        "frame_corruption",
        "camera_instability",
        "low_visibility",
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
    "robot_body_consistency": ["robot_body_consistency"],
    "physical_plausibility": ["physical_plausibility"],
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
    "object_hallucination": 4,
    "object_disappearance": 5,
    "object_identity_swap": 4,
    "object_distortion": 3,
    "object_color_or_shape_drift": 3,
    "hallucinated_robot_part": 5,
    "missing_robot_part": 4,
    "duplicated_arm_or_gripper": 5,
    "robot_body_deformation": 4,
    "left_right_robot_identity_confusion": 4,
    "object_teleportation": 5,
    "object_floating": 4,
    "object_penetration": 4,
    "impossible_motion": 4,
    "grasp_without_visible_support": 4,
    "blur": 2,
    "occlusion": 2,
    "frame_corruption": 3,
    "camera_instability": 2,
    "low_visibility": 2,
}


def dimension_for_glitch_type(glitch_type: str) -> str | None:
    return GLITCH_TYPE_TO_DIMENSION.get(glitch_type)


def valid_glitch_type_for_dimension(glitch_type: str, dimension: str) -> bool:
    return glitch_type in GLITCH_TYPES.get(dimension, [])
