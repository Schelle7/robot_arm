# TODO: Next control, logging, and replay-debug tasks (non-dynamics plan)

This PR is intentionally **documentation/planning only**. It does **not** modify MuJoCo simulation dynamics, actuator gains, timestep, actuator definitions, physics stepping, or control behavior.

Repository: `Schelle7/robot_arm`

Relevant files and links:
- `/home/runner/work/robot_arm/robot_arm/src/robot_arm/envs/env.py` — https://github.com/Schelle7/robot_arm/blob/main/src/robot_arm/envs/env.py
- `/home/runner/work/robot_arm/robot_arm/src/robot_arm/episode_runner.py` — https://github.com/Schelle7/robot_arm/blob/main/src/robot_arm/episode_runner.py
- `/home/runner/work/robot_arm/robot_arm/src/robot_arm/recorder.py` — https://github.com/Schelle7/robot_arm/blob/main/src/robot_arm/recorder.py
- `/home/runner/work/robot_arm/robot_arm/src/robot_arm/policies.py` — https://github.com/Schelle7/robot_arm/blob/main/src/robot_arm/policies.py
- `/home/runner/work/robot_arm/robot_arm/src/robot_arm/envs/safety.py` — https://github.com/Schelle7/robot_arm/blob/main/src/robot_arm/envs/safety.py
- `/home/runner/work/robot_arm/robot_arm/src/robot_arm/backends/sim_arm.py` — https://github.com/Schelle7/robot_arm/blob/main/src/robot_arm/backends/sim_arm.py
- `/home/runner/work/robot_arm/robot_arm/scripts/replay_sim.py` — https://github.com/Schelle7/robot_arm/blob/main/scripts/replay_sim.py
- `/home/runner/work/robot_arm/robot_arm/conf/waypoint/default.yaml` — https://github.com/Schelle7/robot_arm/blob/main/conf/waypoint/default.yaml
- `/home/runner/work/robot_arm/robot_arm/conf/control/default.yaml` — https://github.com/Schelle7/robot_arm/blob/main/conf/control/default.yaml
- `/home/runner/work/robot_arm/robot_arm/conf/experiment/one_step_cuda.yaml` — https://github.com/Schelle7/robot_arm/blob/main/conf/experiment/one_step_cuda.yaml
- `/home/runner/work/robot_arm/robot_arm/models/so101/so101.xml` — https://github.com/Schelle7/robot_arm/blob/main/models/so101/so101.xml

## Important control/debugging distinctions

Keep these concepts separate when logging and when changing behavior later:
- **High-level Cartesian waypoint speed**: the waypoint follower in `/home/runner/work/robot_arm/robot_arm/src/robot_arm/policies.py` uses `position_speed_meters_per_second` from `/home/runner/work/robot_arm/robot_arm/conf/waypoint/default.yaml` to limit Cartesian motion per low-level step.
- **Low-level policy action**: the RL low-level policy output used in `/home/runner/work/robot_arm/robot_arm/src/robot_arm/episode_runner.py` and `/home/runner/work/robot_arm/robot_arm/src/robot_arm/envs/env.py` before scaling.
- **Scaled joint delta**: in `/home/runner/work/robot_arm/robot_arm/src/robot_arm/envs/env.py`, `delta = action * self.delta_action_scale` converts the low-level policy action into a per-step joint-position delta.
- **Requested joint-position target**: also in `/home/runner/work/robot_arm/robot_arm/src/robot_arm/envs/env.py`, `target_positions = self.current_joint_angles + delta` defines the joint targets that are requested from the backend.
- **Measured joint velocity**: from `/home/runner/work/robot_arm/robot_arm/src/robot_arm/envs/env.py` observations and backend state (`Present_Velocity`), this is what the robot/simulator actually did, not what was requested.
- **Hard velocity safety limit**: a future safety rule that actively clamps or rejects commanded target motion to keep every joint at or below a configured maximum velocity. This must be distinguished from merely logging measured velocity.

## TODO items

### 1) Log TCP / end-effector velocity explicitly
- Add explicit logging of TCP/end-effector velocity, with both:
  - **computation method** (for example finite-difference of TCP position and/or full pose over time, or MuJoCo-derived body/site velocity if available), and
  - **units** (`m/s` for linear velocity, and if rotational velocity is later added, clearly label `rad/s`).
- Primary implementation candidates:
  - read current TCP pose in `/home/runner/work/robot_arm/robot_arm/src/robot_arm/envs/env.py`
  - store per-step timing and recorder output in `/home/runner/work/robot_arm/robot_arm/src/robot_arm/episode_runner.py` and `/home/runner/work/robot_arm/robot_arm/src/robot_arm/recorder.py`
  - if simulation-specific helper data is needed, isolate it in `/home/runner/work/robot_arm/robot_arm/src/robot_arm/backends/sim_arm.py`
- Include timestamps and `dt` in the same logging path so the velocity calculation is auditable.

### 2) Log complete per-joint control/debug data for every low-level step
For each joint (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`), log at every low-level step:
- raw low-level policy action
- scaled joint delta
- requested target position
- actual/present position
- position error (`requested target - actual/present position`)
- measured joint velocity
- safety-clipped target where applicable
- requested-vs-safe target difference
- per-joint configured maximum velocity once a hard limit exists
- acceleration if implemented, or otherwise a clearly labeled follow-up item to add/derive it

Notes:
- Today, low-level action is only stored inside nested `dense_trajectory` recorder data in `/home/runner/work/robot_arm/robot_arm/src/robot_arm/recorder.py`; add **explicit top-level logging/printing** for debugging instead of relying only on nested recorder internals.
- Current observations already include `joint_positions` and `joint_velocities` in `/home/runner/work/robot_arm/robot_arm/src/robot_arm/envs/env.py`, but the requested target, safety-clipped target, target difference, timestamps/`dt`, and explicit low-level action logging are missing.
- Since maximum joint velocity is safety-critical, print/log it clearly during debugging rather than burying it only in recorder structures.

### 3) Reduce high-level Cartesian waypoint speed to 5 cm/s by changing configuration only
- Update `/home/runner/work/robot_arm/robot_arm/conf/waypoint/default.yaml`:
  - change `position_speed_meters_per_second` from `0.30` to `0.05`
- This is the **Cartesian waypoint speed** used by the waypoint follower in `/home/runner/work/robot_arm/robot_arm/src/robot_arm/policies.py`.
- Do **not** confuse this with `/home/runner/work/robot_arm/robot_arm/conf/control/default.yaml` `action_scale_radians_per_second`, which affects low-level joint target increments instead.
- If experiment overrides intentionally differ, decide explicitly whether they should keep overriding the new default.

### 4) Add/enforce a hard maximum joint velocity of `0.2 rad/s`
- Requirement: enforce a real safety limit of **`0.2 rad/s` max per joint**, not just log measured velocity.
- Also log the measured velocity separately so it is obvious whether the limit is being respected in practice.

Safest enforcement location to investigate first:
- `/home/runner/work/robot_arm/robot_arm/src/robot_arm/envs/safety.py`
  - Reason: this is already the command-safety chokepoint (`SafeArmWrapper.write_goal()`), so a velocity cap can be implemented as a safety-layer transformation from requested target to safe target before the backend executes it.
  - Preferred approach: use current measured position, elapsed `dt`, and a per-joint max velocity to clamp the next commanded target increment.

Other places to inspect but treat as less safe for the primary enforcement:
- `/home/runner/work/robot_arm/robot_arm/src/robot_arm/envs/env.py`
  - could clamp `delta` before converting to target positions, but that mixes policy scaling with safety logic.
- `/home/runner/work/robot_arm/robot_arm/src/robot_arm/backends/sim_arm.py`
  - avoid putting the primary limit here if the intention is to share safety behavior across sim and real backends.

Open design questions to resolve before implementing the cap:
- Should the hard cap apply to the gripper too, or only revolute arm joints?
- Should the cap use one global value (`0.2 rad/s`) or allow per-joint overrides?
- What should be used as `dt` for enforcement in real hardware versus simulation?
- Should cap violations be logged only, clipped silently, clipped with counters, or escalate to a safety event?

### 5) Replay viewer text/debug overlay feasibility (`scripts/replay_sim.py`)
Result of current inspection:
- `/home/runner/work/robot_arm/robot_arm/scripts/replay_sim.py` already uses `mujoco.viewer.launch_passive(...)` and calls:
  - `update_tcp_debug_user_scene(...)`
  - `update_waypoint_debug_user_scene(...)`
  - `update_desired_pose_debug_user_scene(...)`
- `/home/runner/work/robot_arm/robot_arm/src/robot_arm/backends/sim_arm.py` already draws visualization-only arrows/spheres through `viewer_inst.user_scn` geoms.
- That means replay already supports **safe visualization-only overlays** in the MuJoCo viewer pipeline without touching dynamics.

Feasibility conclusion:
- A visualization-only implementation is **likely safe and isolated** if it only augments replay viewer rendering/state text and does not modify controls, targets, actuator parameters, or physics stepping.
- The existing codebase already proves the replay path can render additional debug visuals.
- Text inside the viewer window should be investigated using MuJoCo viewer overlay APIs for passive viewers; if true text overlay is awkward in the current wrapper, a practical fallback is to render concise debug geometry in-view and keep detailed numeric output in the console/recording.

Practical implementation approach:
- First inspect whether the passive viewer object used in `/home/runner/work/robot_arm/robot_arm/scripts/replay_sim.py` exposes a stable text-overlay API in this environment.
- If available, use it only in replay/debug mode to show a small HUD with frame index, time, active waypoint, TCP speed, and selected per-joint velocity/target data.
- If not available, keep the current console prints and extend `viewer_inst.user_scn` geometry overlays in `/home/runner/work/robot_arm/robot_arm/src/robot_arm/backends/sim_arm.py` for visualization-only cues.
- Any replay overlay work must remain isolated to `/home/runner/work/robot_arm/robot_arm/scripts/replay_sim.py` and replay-only rendering helpers.

### 6) Explicitly preserve non-dynamics scope in the follow-up PR(s)
Do **not** change any of the following in the logging/replay-overlay PR:
- `/home/runner/work/robot_arm/robot_arm/models/so101/so101.xml`
- MuJoCo gains (`kp`, `kv`)
- timestep
- actuator definitions
- force limits
- physics stepping behavior
- simulation control behavior outside isolated replay-only visualization

If a later PR implements the velocity cap, keep it scoped to command/safety handling and logging, not actuator model tuning.

## Missing diagnostics currently identified
These should be added or explicitly tracked as follow-ups:
- timestamps per low-level step
- `dt` per low-level step
- TCP linear velocity with method and units
- per-joint maximum velocity setting
- measured joint velocity already available, but it needs clearer explicit logging/printing
- raw low-level action explicitly logged (not only nested recorder storage)
- scaled joint delta explicitly logged
- requested target position explicitly logged
- safety-clipped target explicitly logged
- requested-vs-safe target difference explicitly logged
- position error explicitly logged
- acceleration, or a follow-up item that defines how it will be derived and validated

## PR description guidance
The PR description should say:
- this is a **non-dynamics documentation/planning PR**
- it documents the next logging, replay-debug, and velocity-safety tasks
- it intentionally does **not** change MuJoCo dynamics or actuator behavior
- enforcing the `0.2 rad/s` cap still needs a follow-up decision on exact scope (`all joints?`, `gripper?`, `global vs per-joint`, `dt source`, and logging/escalation behavior)
