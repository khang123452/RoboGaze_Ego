# RoboGaze-Ego — Handoff Notes

Quick orientation for anyone picking up this fork after me. Start here before touching code.

## What this repo is

[RoboGaze](https://github.com/cair-vinuni/RoboGaze) is a training-free, multi-agent VLM pipeline
originally built to QA **generated** robot-manipulation videos — it flags things like a world
model hallucinating a robot arm, teleporting an object, or ignoring the task instruction.

We forked it to instead QA **real egocentric (first-person) human hand-manipulation footage**
that VinRobotics is sourcing from data vendors, checked against the binding requirements in
`VinRobotics Egocentric Manipulation Data — Technical Specification` (24 June 2026). There's no
generative model to hallucinate here, so the taxonomy and prompts were reworked around a
different question: *is this real clip usable, spec-compliant, and correctly labeled?*

The three-stage architecture (task-scene grounding → specialist routing → critic verification)
is unchanged from upstream. What changed is the taxonomy, the schema, and every prompt.

## What changed, file by file

| File | Change |
|---|---|
| `src/robogaze/taxonomy.py` | `robot_body_consistency` → `hand_body_consistency` (hand visibility/tracking/identity instead of generation hallucinations). Added a new **`spec_compliance`** dimension with no upstream analogue — checks hand visibility, idle-time ratio, rigid-object-only, environment/background rules, camera framing. Now **7 dimensions × 5 types = 35 total**. |
| `src/robogaze/schemas.py` | `RobotEntity` → `HandEntity`, `WindowRobotState` → `WindowHandState`, `RobotSummary` → `HandSummary`, `RobotAction` → `HandAction`, `robot_entities`/`visible_robot_parts` → `hand_entities`/`visible_hand_parts`, `WindowStateMemory.robot` → `.hands`. |
| `src/robogaze/agents.py`, `pipeline.py`, `views.py` | Renames propagated through raw-JSON parsing, fallback construction, state-aggregation priority maps, and per-dimension view routing. |
| `src/robogaze/prompts.py` | Every system prompt reworded from robot-generation language to egocentric-hand language. `HAND_BODY_CONSISTENCY_SYSTEM` replaces `ROBOT_BODY_CONSISTENCY_SYSTEM`. New `SPEC_COMPLIANCE_SYSTEM` is wired directly to the tech-spec numbers (both hands visible, <20% idle, >80% activity, rigid-object-only, indoor/minimal-background-motion, eye-level/downward/landscape framing). |
| `README.md`, `scripts/` | Updated overview and example/judge prompt text. Citation to the original RoboGaze paper kept, since the architecture is theirs. |

Full diff: `git log -p -1` on commit `52f98cd` (or `git show 52f98cd --stat`).

## The 7-dimension taxonomy (current state)

| Dimension | Status | Types |
|---|---|---|
| `task_progress` | unchanged from upstream | task_incompletion, failed_grasp, failed_placement, premature_termination, ambiguous_task_success |
| `instruction_consistency` | unchanged from upstream | wrong_effector, wrong_object, wrong_target_location, wrong_action_order, ignored_instruction_constraint |
| `object_scene_consistency` | lightly revised | object_disappearance, object_identity_swap, object_distortion, unexpected_object_appearance, object_state_mislabel |
| `hand_body_consistency` | **redesigned** (was `robot_body_consistency`) | hand_occluded_during_manipulation, single_hand_only_during_bimanual_task, hand_object_contact_ambiguous, hand_pose_tracking_implausible, left_right_hand_identity_confusion |
| `physical_plausibility` | unchanged from upstream, prompt reframed | object_teleportation, object_floating, object_penetration, impossible_motion, grasp_without_visible_support |
| `visual_quality` | revised for capture (not generation) artifacts | motion_blur, exposure_or_white_balance_issue, encoding_or_resolution_artifact, camera_instability, frame_corruption |
| `spec_compliance` | **new**, no upstream analogue | hands_not_both_visible, excessive_idle_time, non_rigid_object_manipulation, disallowed_environment_or_background_motion, camera_framing_violation |

## Known gaps — read before assuming this is done

- **`spec_compliance` only checks what's visible in frame.** It cannot judge frame rate,
  bitrate, GOP length, color depth, or IMU-to-video sync tolerance (≤1ms) or head-pose odometry
  error (≤1%) from the tech spec — those need a separate metadata/file-level validation script,
  not a VLM looking at frames. This dimension is a partial spec check, not the full contract.
- **`physical_plausibility` and `object_scene_consistency` were not redesigned as heavily** as
  `hand_body_consistency`/`spec_compliance` — only the prompt framing changed, not the type list.
  Worth revisiting once we see what actually fires on real clips.
- **Never run end-to-end against real egocentric footage yet.** Everything below was verified
  with `py_compile`, a full module import, taxonomy/prompt/view coverage assertions, and the
  no-VLM heuristic task-memory fallback — not against an actual video + VLM backend. Treat the
  prompts as a strong first draft, not tuned.
- **No ground-truth annotation set for this data yet**, so `scripts/evaluate.py` (which scores
  predictions against a `ground_truth/` folder) has nothing to compare against until we have
  human-labeled egocentric clips.

## Getting set up

```bash
cd RoboGaze_Ego
uv sync                      # or: pip install -e .
ffmpeg -version              # required on PATH
bash scripts/serve_vlm.sh    # serves a local VLM at localhost:8000/v1
```

Then run one clip:

```bash
export LOCAL_BASE_URL=http://localhost:8000/v1
export LOCAL_MODEL=gemma4

uv run robogaze \
  --task-instruction "Use the right hand to pick up the red cup and place it on the plate." \
  --initial-frame /path/to/initial_frame.jpg \
  --video /path/to/clip.mp4 \
  --output-dir outputs/robogaze \
  --video-id example_001
```

Full install/run/batch/eval instructions are in `README.md`.

## Git state

- Remote: `https://github.com/khang123452/RoboGaze_Ego.git`
- Local `main` is ahead of `origin/main` by the taxonomy-adaptation commit — **not pushed yet**.
  Push from a machine with your GitHub credentials: `git push`.

## Suggested next steps

1. Run the pipeline against a handful of real VinRobotics clips end-to-end and read the raw
   `hypotheses.json` / `report.json` output in `outputs/robogaze/<video_id>/` — that's the
   fastest way to see whether the new prompts are actually catching the right things.
2. Decide whether `spec_compliance`'s non-visual checks (sync, bitrate, GOP, color depth) belong
   in this pipeline at all, or in a separate lightweight `ffprobe`-based validator that runs
   before RoboGaze-Ego even sees the clip.
3. Once a few clips are hand-reviewed, revisit `physical_plausibility` and
   `object_scene_consistency` — they're the two dimensions least adapted from the original,
   generation-focused taxonomy.
