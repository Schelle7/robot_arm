# Primitives Generator TODO

## 1. Missing Primitives

- **`close_gripper` / `open_gripper`**
  - Prompt: `"close gripper"` / `"open gripper"`
  - Target: Keep TCP position fixed, transition gripper to target closed/open limit.
  - Currently fused into `"close gripper and lift {color} box"`.

- **`pick_up(color)`**
  - Prompt: `"pick up {color} box"`
  - Target: Move TCP vertically upwards (+z) while keeping gripper closed.
  - Same fusion.

---

## 2. Implementation Tasks

- [ ] **Multiple coloured boxes**
  - `{color}` carries no information while the scene holds one box, and this is the first task
    where the image cannot be ignored.
  - Touches `scene.xml`, `_find_target_box_position`, the hardcoded "red" in the pick-and-place
    prompts, and `randomize_box` / `get_privileged_box_pose` / `disable_box_collisions` in
    `SimBackend`, all of which name the single body `target_box`.
  - Needs non-overlapping placement sampling.

- [ ] **Reachable IK boundaries and collision filtering**
  - `_is_reachable_position` checks a shoulder-distance shell and a height window, which is a proxy
    for "an IK solution exists", not a test of one.
  - Nothing rejects self-collision or ground contact, and `disable_box_collisions` is on everywhere.
  - Orientation reachability is unchecked: `generate_relative_moves` holds a world-fixed orientation
    across a translation, which a 5 DOF arm generally cannot do.

- [ ] **Dataset validation**
  - Image range is covered by `tests/test_vla_observation.py`. Action chunk dimensions, video
    alignment and tokenized task metadata are not.

---

## 3. Decisions to make

Design questions, not coding tasks.

- What is a `target_location`? A named surface, a marked spot in the scene, or the random pose the
  code currently supplies? The prompt promises a location and the code delivers a pose.
- Does `close gripper` complete on position or on measured load? A position target cannot express
  "closed on an object". See the force note in TODO.md.
- How many boxes and colours, and how are they placed without overlapping?
- Does the privileged target offset stay in `observation.state` for the vision primitives? While it
  is there, `"move to red box"` is solvable without looking at the image at all.
  