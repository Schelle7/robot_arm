import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from omegaconf import DictConfig

from robot_arm.pose import Pose, axis_angular_distance
from robot_arm.backends.arm import Arm
from robot_arm.robot_schema import MOTOR_ORDER


@dataclass
class EnvironmentState:
    observation: Dict[str, np.ndarray]
    sensor_state: Dict[str, Any]
    privileged_state: Dict[str, Any]


class RobotEnv:
    """
    Standard driver wrapper for the robotic arm.
    """

    def __init__(
        self,
        arm: Arm,
        cfg: DictConfig,
        output_dir: str,
    ):
        violation_penalty_factor = cfg.reward.violation_penalty_factor
        low_level_hz = cfg.control.frequencies.low_level
        mujoco_hz = cfg.control.frequencies.mujoco

        assert mujoco_hz % low_level_hz == 0, f"mujoco_hz ({mujoco_hz}) must be divisible by low_level_hz ({low_level_hz})"
        assert cfg.control.action_scale_radians_per_second >= cfg.waypoint.rotation_speed_radians_per_second, (
            "control.action_scale_radians_per_second must be at least waypoint.rotation_speed_radians_per_second"
        )
        assert cfg.control.action_scale_radians_per_second >= cfg.waypoint.gripper_speed_radians_per_second, (
            "control.action_scale_radians_per_second must be at least waypoint.gripper_speed_radians_per_second"
        )

        self.arm = arm
        self.backend = cfg.backend
        self.delta_action_scale = cfg.control.action_scale_radians_per_second / low_level_hz
        self.violation_penalty_factor = violation_penalty_factor
        self.low_level_hz = low_level_hz
        self.mid_level_hz = cfg.control.frequencies.mid_level
        self.mujoco_steps_per_low_level_step = mujoco_hz // low_level_hz
        self.tracking_deviation_enabled = cfg.reward.tracking.deviation
        self.tracking_progress_enabled = cfg.reward.tracking.progress
        self.safety_penalty_enabled = cfg.reward.safety_penalty
        self.termination_penalty_enabled = cfg.reward.termination_penalty
        self.pose_delta_diagnostics_enabled = cfg.training.pose_delta_diagnostics_enabled
        self.staging_enabled = cfg.control.staging.enabled
        self.initial_joint_range_percent = cfg.control.initial_joints.range_percent
        self.staging_speed_radians_per_second = cfg.control.staging.speed_radians_per_second
        self.staging_tolerance_radians = cfg.control.staging.tolerance_radians
        self.staging_max_steps = cfg.control.staging.max_steps
        self.staging_pause_seconds = cfg.control.staging.pause_seconds
        self.output_dir = output_dir

        self.position_distance_weight = float(cfg.reward.pose_weights.position)
        self.rotation_primary_distance_weight = float(cfg.reward.pose_weights.rotation_primary)
        self.rotation_secondary_distance_weight = float(cfg.reward.pose_weights.rotation_secondary)
        self.gripper_distance_weight = float(cfg.reward.pose_weights.gripper)
        trajectory_duration_seconds = cfg.waypoint.trajectory_length / low_level_hz
        self.position_distance_scale = cfg.waypoint.position_speed_meters_per_second * trajectory_duration_seconds
        self.rotation_distance_scale = cfg.waypoint.rotation_speed_radians_per_second * trajectory_duration_seconds
        self.gripper_distance_scale = cfg.waypoint.gripper_speed_radians_per_second * trajectory_duration_seconds

        # Hardcoding the ordered list of motors to ensure deterministic vectorization
        self.motor_order = MOTOR_ORDER

        self.reset_reward_tracking()
        self.pose_delta_diagnostics = {}

    @property
    def current_joint_angles(self) -> np.ndarray:
        state_dict = self.arm.read_state()
        return np.array(
            [state_dict["Present_Position"][m] for m in self.motor_order],
            dtype=np.float32,
        )

    def get_privileged_end_effector_pose(self) -> Pose:
        return self.arm.get_tcp_pose()

    def _get_obs(self) -> EnvironmentState:
        state_dict = self.arm.read_state()
        current_pos = np.array(
            [state_dict["Present_Position"][m] for m in self.motor_order],
            dtype=np.float32,
        )
        current_vel = np.array(
            [state_dict["Present_Velocity"][m] for m in self.motor_order],
            dtype=np.float32,
        )

        obs = {
            "joint_positions": current_pos,
            "joint_velocities": current_vel,
        }

        privileged_state = {
            "end_effector_pose": self.get_privileged_end_effector_pose(),
        }
        if self.backend == "sim":
            privileged_state["sim_state"] = state_dict["sim_state"]

        return EnvironmentState(
            observation=obs,
            sensor_state=state_dict,
            privileged_state=privileged_state,
        )

    def reset(self) -> EnvironmentState:
        self.reset_reward_tracking()

        if self.backend == "sim":
            self.arm.reset_sim()
        elif self.backend == "real" and self.staging_enabled:
            self.arm.move_to_staging_pose(
                initial_joint_range_percent=self.initial_joint_range_percent,
                speed_radians_per_second=self.staging_speed_radians_per_second,
                tolerance_radians=self.staging_tolerance_radians,
                max_steps=self.staging_max_steps,
                pause_seconds=self.staging_pause_seconds,
                output_dir=self.output_dir,
            )

        return self._get_obs()

    def reset_chunk_reward_tracking(self, chunk_start_pose: Pose, cartesian_action_path: np.ndarray) -> None:
        desired_pose = self._compute_desired_pose(chunk_start_pose, cartesian_action_path)
        (
            self.previous_position_distance,
            self.previous_primary_orientation_distance,
            self.previous_secondary_orientation_distance,
            self.previous_gripper_distance,
        ) = self._compute_pose_distances(chunk_start_pose, desired_pose)

    def reset_reward_tracking(self) -> None:
        self.previous_position_distance = 0.0
        self.previous_primary_orientation_distance = 0.0
        self.previous_secondary_orientation_distance = 0.0
        self.previous_gripper_distance = 0.0

    def read_camera(self):
        return self.arm.read_camera()

    def _compute_desired_pose(self, chunk_start_pose: Pose, cartesian_action_path: np.ndarray) -> Pose:
        assert cartesian_action_path.shape[0] == 1, "Only one desired pose is supported temporarily."
        return chunk_start_pose.apply_delta(cartesian_action_path[0])

    def _compute_pose_distances(
        self,
        current_pose: Pose,
        desired_pose: Pose,
    ) -> Tuple[float, float, float, float]:
        position_distance = current_pose.positional_distance(desired_pose)
        primary_orientation_distance = self._axis_angular_distance(
            current_pose.closing_axis,
            desired_pose.closing_axis,
        )
        secondary_orientation_distance = self._axis_angular_distance(
            current_pose.secondary_axis,
            desired_pose.secondary_axis,
        )
        gripper_distance = abs(current_pose.gripper - desired_pose.gripper)
        return (
            position_distance,
            primary_orientation_distance,
            secondary_orientation_distance,
            gripper_distance,
        )

    def _axis_angular_distance(self, first_axis: np.ndarray, second_axis: np.ndarray) -> float:
        return axis_angular_distance(first_axis, second_axis)

    def _compute_safety_penalty(self, requested_action: Dict[str, float], safe_action: Dict[str, float]) -> float:
        safety_penalty = 0.0
        for motor in requested_action:
            diff = abs(requested_action[motor] - safe_action[motor])
            if diff > 0:
                safety_penalty -= diff * self.violation_penalty_factor
        return safety_penalty

    def _compute_termination_penalty(self, chunk_terminated: bool, current_pose: Pose, desired_pose: Pose) -> float:
        termination_penalty = 0.0
        if chunk_terminated:
            (
                position_distance,
                primary_orientation_distance,
                secondary_orientation_distance,
                gripper_distance,
            ) = self._compute_pose_distances(current_pose, desired_pose)
            termination_penalty = -sum(
                (
                    self.position_distance_weight * position_distance / self.position_distance_scale,
                    self.rotation_primary_distance_weight * primary_orientation_distance / self.rotation_distance_scale,
                    self.rotation_secondary_distance_weight * secondary_orientation_distance / self.rotation_distance_scale,
                    self.gripper_distance_weight * gripper_distance / self.gripper_distance_scale,
                )
            )
        return termination_penalty

    def _compute_deviation_penalties(
        self,
        position_distance: float,
        primary_orientation_distance: float,
        secondary_orientation_distance: float,
        gripper_distance: float,
    ) -> Dict[str, float]:
        return {
            "position_deviation_penalty": -self.position_distance_weight * position_distance / self.position_distance_scale,
            "primary_orientation_deviation_penalty": -self.rotation_primary_distance_weight
            * primary_orientation_distance
            / self.rotation_distance_scale,
            "secondary_orientation_deviation_penalty": -self.rotation_secondary_distance_weight
            * secondary_orientation_distance
            / self.rotation_distance_scale,
            "gripper_deviation_penalty": -self.gripper_distance_weight * gripper_distance / self.gripper_distance_scale,
        }

    def _compute_pose_delta_diagnostics(
        self,
        current_pose: Pose,
        desired_pose: Pose,
        cartesian_action_path: np.ndarray,
    ) -> Dict[str, float]:
        (
            position_distance,
            primary_orientation_distance,
            secondary_orientation_distance,
            gripper_distance,
        ) = self._compute_pose_distances(
            current_pose,
            desired_pose,
        )
        weighted_orientation_distance = (
            self.rotation_primary_distance_weight * primary_orientation_distance + self.rotation_secondary_distance_weight * secondary_orientation_distance
        )
        desired_action = cartesian_action_path[0]
        return {
            "position_distance": position_distance,
            "primary_orientation_distance": primary_orientation_distance,
            "secondary_orientation_distance": secondary_orientation_distance,
            "orientation_distance": weighted_orientation_distance,
            "gripper_distance": gripper_distance,
            "weighted_chessboard_distance": max(
                self.position_distance_weight * position_distance,
                weighted_orientation_distance,
                self.gripper_distance_weight * gripper_distance,
            ),
            "desired_position_delta_norm_meters": float(np.linalg.norm(desired_action[:3])),
            "desired_position_speed_meters_per_second": float(np.linalg.norm(desired_action[:3]) * self.mid_level_hz),
            "desired_rotation_delta_norm_radians": float(np.linalg.norm(desired_action[3:6])),
            "desired_rotation_speed_radians_per_second": float(np.linalg.norm(desired_action[3:6]) * self.mid_level_hz),
        }

    def compute_reward(
        self,
        requested_action: Dict[str, float],
        safe_action: Dict[str, float],
        cartesian_action_path: np.ndarray,
        current_pose: Pose,
        chunk_start_pose: Pose,
        chunk_terminated: bool,
    ) -> Tuple[float, Dict[str, float]]:
        desired_pose = self._compute_desired_pose(chunk_start_pose, cartesian_action_path)
        (
            position_distance,
            primary_orientation_distance,
            secondary_orientation_distance,
            gripper_distance,
        ) = self._compute_pose_distances(
            current_pose,
            desired_pose,
        )
        position_reward = self.position_distance_weight * (self.previous_position_distance - position_distance) / self.position_distance_scale
        primary_orientation_reward = (
            self.rotation_primary_distance_weight
            * (self.previous_primary_orientation_distance - primary_orientation_distance)
            / self.rotation_distance_scale
        )
        secondary_orientation_reward = (
            self.rotation_secondary_distance_weight
            * (self.previous_secondary_orientation_distance - secondary_orientation_distance)
            / self.rotation_distance_scale
        )
        gripper_reward = self.gripper_distance_weight * (self.previous_gripper_distance - gripper_distance) / self.gripper_distance_scale
        self.previous_position_distance = position_distance
        self.previous_primary_orientation_distance = primary_orientation_distance
        self.previous_secondary_orientation_distance = secondary_orientation_distance
        self.previous_gripper_distance = gripper_distance

        safety_penalty = self._compute_safety_penalty(requested_action, safe_action)

        termination_penalty = self._compute_termination_penalty(chunk_terminated, current_pose, desired_pose)

        if self.pose_delta_diagnostics_enabled:
            self.pose_delta_diagnostics = self._compute_pose_delta_diagnostics(
                current_pose,
                desired_pose,
                cartesian_action_path,
            )

        reward_breakdown = {}
        if self.tracking_progress_enabled:
            reward_breakdown["position_reward"] = position_reward
            reward_breakdown["primary_orientation_reward"] = primary_orientation_reward
            reward_breakdown["secondary_orientation_reward"] = secondary_orientation_reward
            reward_breakdown["gripper_reward"] = gripper_reward
        if self.tracking_deviation_enabled:
            reward_breakdown.update(
                self._compute_deviation_penalties(
                    position_distance,
                    primary_orientation_distance,
                    secondary_orientation_distance,
                    gripper_distance,
                )
            )
        if self.safety_penalty_enabled:
            reward_breakdown["safety_penalty"] = safety_penalty
        if self.termination_penalty_enabled:
            reward_breakdown["termination_penalty"] = termination_penalty

        total_reward = float(sum(reward_breakdown.values()))

        return total_reward, reward_breakdown

    def step(
        self,
        action: np.ndarray,
        cartesian_action_path: np.ndarray,
        privileged_chunk_start_pose: Pose,
        chunk_terminated: bool,
    ) -> Tuple[EnvironmentState, float, Dict[str, float]]:

        # 1. Unscale delta action and add to current joint angles
        delta = action * self.delta_action_scale
        target_positions = self.current_joint_angles + delta

        # 2. Map target positions vector to dictionary and send to arm
        action_dict = {motor: float(pos) for motor, pos in zip(self.motor_order, target_positions)}

        # 3. Hold the joint target while MuJoCo advances through the control interval
        safe_action_dict = self.arm.write_goal(action_dict)
        for _ in range(1, self.mujoco_steps_per_low_level_step):
            safe_action_dict = self.arm.write_goal(action_dict)

        # 4. Get new observation
        state = self._get_obs()

        reward, reward_breakdown = self.compute_reward(
            requested_action=action_dict,
            safe_action=safe_action_dict,
            cartesian_action_path=cartesian_action_path,
            current_pose=state.privileged_state["end_effector_pose"],
            chunk_start_pose=privileged_chunk_start_pose,
            chunk_terminated=chunk_terminated,
        )

        return state, reward, reward_breakdown
