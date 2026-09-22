import time

import mujoco
import mujoco.viewer
import numpy as np

from robot_arm.arms.sim_arm import (
    update_desired_pose_debug_user_scene,
    update_tcp_debug_user_scene,
    update_waypoint_debug_user_scene,
)
from robot_arm.replay.recording import calculate_pose_delta, get_desired_poses
from robot_arm.replay.display import build_replay_display
from robot_arm.robot_schema import HISTORY_CONTEXT_FEATURE_NAMES, HISTORY_FEATURE_NAMES, MOTOR_ORDER


class ReplayViewer:
    def __init__(self, recorded_cfg, data, model_path, window_title):
        self.recorded_cfg = recorded_cfg
        self.window_title = window_title
        self.model = mujoco.MjModel.from_xml_path(model_path)
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.model.geom_rgba[floor_id, 3] = 0.25
        self.mdata = mujoco.MjData(self.model)
        self.has_sim_state = recorded_cfg.arm_type == "sim"
        if self.has_sim_state:
            self.model.body_pos[:] = data["body_pos"]
            self.model.geom_matid[:] = data["geom_matid"]
            self.qpos_recording = data["qpos"]
            self.qvel_recording = data["qvel"]
        else:
            joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in MOTOR_ORDER]
            self.qpos_recording = np.tile(self.model.qpos0, (len(data["joint_positions"]), 1))
            self.qvel_recording = np.zeros((len(data["joint_velocities"]), self.model.nv))
            self.qpos_recording[:, self.model.jnt_qposadr[joint_ids]] = data["joint_positions"]
            self.qvel_recording[:, self.model.jnt_dofadr[joint_ids]] = data["joint_velocities"]
        self.desired_actions = data["cartesian_action"]
        self.joint_positions = data["joint_positions"]
        self.joint_velocities = data["joint_velocities"]
        self.dense_trajectory = data["dense_trajectory"]
        self.action_history = [[float(value) for value in sample["action"]] for trajectory in self.dense_trajectory for sample in trajectory]
        self.action_diagnostics = data["cartesian_action_diagnostics"]
        self.completes_active_primitives = data["completes_active_primitive"]
        self.primitive_prompts = data["primitive_prompt"]
        self.primitive_indices = data["primitive_index"]
        self.waypoints = data["waypoints"]
        self.recorded_poses = data["end_effector_pose"]
        self.num_states = len(self.qpos_recording)
        self.num_actions = len(self.desired_actions)
        if self.num_actions == 0:
            raise ValueError("Replay recording must contain at least one Cartesian action path.")
        self.current_frame = 0
        self.current_joint_step = 0
        self.auto_play = False

    def _clamp_joint_step(self):
        dense_step_count = len(self.dense_trajectory[self.current_frame]) if self.current_frame < self.num_actions else 0
        self.current_joint_step = min(self.current_joint_step, max(dense_step_count - 1, 0))

    def handle_key(self, keycode):
        if keycode == 262:
            self.current_frame = (self.current_frame + 1) % self.num_states
            self._clamp_joint_step()
        elif keycode == 263:
            self.current_frame = (self.current_frame - 1) % self.num_states
            self._clamp_joint_step()
        elif keycode == 32:
            self.auto_play = not self.auto_play

    def apply_commands(self, take_commands):
        frame_requested = False
        for command, value in take_commands():
            if command == "frame":
                self.current_frame = value
                self._clamp_joint_step()
                frame_requested = True
            elif command == "joint_step":
                self.current_joint_step = value
            elif command == "toggle_play":
                self.auto_play = not self.auto_play
        return frame_requested

    def update_frame(self, advance):
        if advance:
            self.current_frame = (self.current_frame + 1) % self.num_states
            self._clamp_joint_step()

        self.mdata.qpos[:] = self.qpos_recording[self.current_frame]
        self.mdata.qvel[:] = self.qvel_recording[self.current_frame]
        mujoco.mj_forward(self.model, self.mdata)

    def transition_metrics(self):
        if self.current_frame >= self.num_actions:
            return [], (None, None)

        desired_poses = get_desired_poses(
            self.recorded_poses,
            self.desired_actions,
            self.current_frame,
        )
        if self.current_frame + 1 >= self.num_states:
            return desired_poses, (None, None)
        observed_pose_delta = calculate_pose_delta(
            self.recorded_poses[self.current_frame],
            self.recorded_poses[self.current_frame + 1],
        )
        pose_tracking_error = calculate_pose_delta(
            self.recorded_poses[self.current_frame + 1],
            desired_poses[0].as_10d(),
        )
        return desired_poses, (observed_pose_delta, pose_tracking_error)

    def update_debug_scene(self, viewer_inst, desired_poses):
        current_action_diagnostics = self.action_diagnostics[self.current_frame] if self.current_frame < self.num_actions else {}
        active_primitive_index = self.primitive_indices[self.current_frame]
        update_tcp_debug_user_scene(viewer_inst.user_scn, self.model, self.mdata)
        update_waypoint_debug_user_scene(
            viewer_inst.user_scn,
            self.waypoints,
            active_primitive_index,
        )
        update_desired_pose_debug_user_scene(viewer_inst.user_scn, desired_poses)
        return current_action_diagnostics

    def render_frame(self, display, metrics, action_diagnostics):
        observed_pose_delta, pose_tracking_error = metrics
        if self.current_frame < self.num_actions:
            primitive_number = int(self.primitive_indices[self.current_frame]) + 1
            active_primitive_label = f"{primitive_number}: {self.primitive_prompts[self.current_frame]}"
            flat_policy_history = self.dense_trajectory[self.current_frame][self.current_joint_step]["obs"]["history"]
            policy_history = flat_policy_history[len(HISTORY_CONTEXT_FEATURE_NAMES) :].reshape(-1, len(HISTORY_FEATURE_NAMES))
        else:
            active_primitive_label = "N/A"
            policy_history = []
        display_lines, warnings = build_replay_display(
            self.model,
            self.mdata,
            self.joint_positions[self.current_frame],
            self.joint_velocities[self.current_frame],
            self.desired_actions[self.current_frame] if self.current_frame < self.num_actions else None,
            self.dense_trajectory[self.current_frame] if self.current_frame < self.num_actions else [],
            observed_pose_delta,
            pose_tracking_error,
            action_diagnostics,
            self.completes_active_primitives[self.current_frame] if self.current_frame < self.num_actions else False,
            self.current_frame,
            self.current_joint_step,
            self.recorded_cfg,
        )
        dense_step_count = len(self.dense_trajectory[self.current_frame]) if self.current_frame < self.num_actions else 0
        history_end_time = (
            sum(len(trajectory) for trajectory in self.dense_trajectory[: self.current_frame]) + self.current_joint_step
        ) / self.recorded_cfg.control.frequencies.joint
        display(
            display_lines,
            warnings,
            self.current_frame,
            self.current_joint_step,
            dense_step_count,
            np.asarray(policy_history).tolist(),
            history_end_time,
            active_primitive_label,
            self.auto_play,
        )

    def run(self, display, take_commands):
        cartesian_hz = self.recorded_cfg.control.frequencies.cartesian
        frame_period = 1.0 / cartesian_hz
        self.mdata.qpos[:] = self.qpos_recording[0]
        self.mdata.qvel[:] = self.qvel_recording[0]
        mujoco.mj_forward(self.model, self.mdata)

        with mujoco.viewer.launch_passive(self.model, self.mdata, key_callback=self.handle_key) as viewer_inst:
            viewer_inst.set_texts((None, None, f"{self.window_title}\nPolicy: {self.recorded_cfg.policy_name}", ""))
            viewer_inst.opt.geomgroup[5] = 0
            while viewer_inst.is_running():
                step_start = time.time()
                frame_requested = self.apply_commands(take_commands)
                self.update_frame(self.auto_play and not frame_requested)
                desired_poses, metrics = self.transition_metrics()
                action_diagnostics = self.update_debug_scene(viewer_inst, desired_poses)
                self.render_frame(display, metrics, action_diagnostics)
                viewer_inst.sync()
                time_until_next_step = frame_period - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
