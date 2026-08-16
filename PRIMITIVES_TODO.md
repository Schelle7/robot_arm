# Primitives Generator TODO

## 1. Primitives Specification

- **`move_to(color)`**
  - Prompt: `"move to {color} box and open gripper"`
  - Target: Position TCP at object pose while driving gripper to open state.

- **`close_gripper` / `open_gripper`**
  - Prompt: `"close gripper"` / `"open gripper"`
  - Target: Keep TCP position fixed, transition gripper to target closed/open limit.

- **`pick_up(color)`**
  - Prompt: `"pick up {color} box"`
  - Target: Move TCP vertically upwards (+z) while keeping gripper closed.

- **`move_directional(direction, distance)`**
  - Prompt: `"move {distance}cm {direction} relative to camera"`
  - Directions: `left`, `right`, `forward`, `backward`, `up`, `down` (wrist/camera frame).
  - Target: Apply Cartesian offset along camera axis with unchanged gripper state.

- **`place(target_location)`**
  - Prompt: `"place on {target_location}"`
  - Target: Lower TCP to surface, then open gripper.

---

## 2. Implementation Tasks

- [ ] **Primitive Classes & State Machine**
  - Implement primitive descriptors with target pose calculation from MuJoCo scene object state.
  - Implement sequential execution that evaluates completion before progressing to next primitive.
  or just have them separately which is better probably

- [ ] **Trajectory & Waypoint Generation**
  - Replace random waypoint sampling with primitive-based interpolation.???
  - Validate reachable IK boundaries and filter collisions.???

- [ ] **Data Collection & Prompt Pairing**
  - Update data collection to sample primitive sequences.
  - Save exact active prompt string alongside camera frames in episode recordings.

- [ ] **LeRobot Conversion & Validation**
  - Run conversion to LeRobot v3.0 format.
  - Inspect action chunk dimensions, video alignment, and tokenized task metadata.
