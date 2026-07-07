"""RoboGaze local inference pipeline."""

from .pipeline import RoboGaze, run_pipeline
from .schemas import (
    CoarseCandidate,
    GlitchEvent,
    GlitchHypothesis,
    GlitchReport,
    SceneMemory,
    SubgoalSegment,
    TaskMemory,
    VerifiedEventGroup,
    VideoMeta,
    ViewBank,
    ViewWindow,
    WindowStateMemory,
)

__all__ = [
    "RoboGaze",
    "run_pipeline",
    "VideoMeta",
    "ViewWindow",
    "ViewBank",
    "TaskMemory",
    "SceneMemory",
    "SubgoalSegment",
    "WindowStateMemory",
    "CoarseCandidate",
    "GlitchHypothesis",
    "VerifiedEventGroup",
    "GlitchEvent",
    "GlitchReport",
]
