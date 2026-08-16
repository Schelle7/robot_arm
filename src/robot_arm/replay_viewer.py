import time

import mujoco
import mujoco.viewer

from robot_arm.backends.sim_arm import (
    update_desired_pose_debug_user_scene,
    update_tcp_debug_user_scene,
    update_waypoint_debug_user_scene,
)
from robot_arm.replay import calculate_pose_delta, get_desired_poses
from robot_arm.replay_display import build_replay_display


class ReplayViewer:
    def __init__(self, recorded_cfg, data, model_path, window_title):
        self.recorded_cfg = recorded_cfg
        self.window_title = window_title
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.mdata = mujoco.MjData(self.model)
        self.qpos_recording = data["qpos"]
        self.qvel_recording = data["qvel"]
        self.desired_actions = data["cartesian_action_path"]
        self.joint_positions = data["joint_positions"]
        self.joint_velocities = data["joint_velocities"]
        self.dense_trajectory = data["dense_trajectory"]
        self.action_diagnostics = data["cartesian_action_diagnostics"]
        self.completes_active_primitives = data["completes_active_primitive"]
        self.primitive_indices = data["primitive_index"]
        self.waypoints = data["waypoints"]
        self.recorded_poses = data["privileged_end_effector_pose"]
        self.num_states = len(self.qpos_recording)
        self.num_actions = len(self.desired_actions)
        if self.num_actions == 0:
            raise ValueError("Replay recording must contain at least one Cartesian action path.")
        self.current_frame = 0
        self.auto_play = False

    def handle_key(self, keycode):
        if keycode == 262:
            self.current_frame = (self.current_frame + 1) % self.num_states
        elif keycode == 263:
            self.current_frame = (self.current_frame - 1) % self.num_states
        elif keycode == 32:
            self.auto_play = not self.auto_play

    def update_frame(self):
        if self.auto_play:
            self.current_frame = (self.current_frame + 1) % self.num_states

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
        current_action_diagnostics = (
            self.action_diagnostics[self.current_frame]
            if self.current_frame < self.num_actions
            else {}
        )
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
        display_lines, warnings = build_replay_display(
            self.model,
            self.mdata,
            self.joint_positions[self.current_frame],
            self.joint_velocities[self.current_frame],
            self.desired_actions[self.current_frame]
            if self.current_frame < self.num_actions
            else None,
            self.dense_trajectory[self.current_frame]
            if self.current_frame < self.num_actions
            else [],
            observed_pose_delta,
            pose_tracking_error,
            action_diagnostics,
            self.completes_active_primitives[self.current_frame]
            if self.current_frame < self.num_actions
            else False,
            self.current_frame,
            self.recorded_cfg,
        )
        display(display_lines, warnings)

    def run(self, display):
        mid_level_hz = self.recorded_cfg.control.frequencies.mid_level
        frame_period = 1.0 / mid_level_hz
        self.mdata.qpos[:] = self.qpos_recording[0]
        self.mdata.qvel[:] = self.qvel_recording[0]
        mujoco.mj_forward(self.model, self.mdata)

        with mujoco.viewer.launch_passive(self.model, self.mdata, key_callback=self.handle_key) as viewer_inst:
            viewer_inst.set_texts((None, None, f"{self.window_title}\nPolicy: {self.recorded_cfg.policy_name}", ""))
            viewer_inst.opt.geomgroup[5] = 0
            while viewer_inst.is_running():
                step_start = time.time()
                self.update_frame()
                desired_poses, metrics = self.transition_metrics()
                action_diagnostics = self.update_debug_scene(viewer_inst, desired_poses)
                self.render_frame(display, metrics, action_diagnostics)
                viewer_inst.sync()
                time_until_next_step = frame_period - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
