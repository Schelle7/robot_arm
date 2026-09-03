import json
import math
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
from omegaconf import DictConfig

from robot_arm.pose import Pose, axis_angular_distance
from robot_arm.backends.arm import Arm
from robot_arm.robot_schema import MOTOR_ORDER


@dataclass
class EnvironmentState:
    observation: Dict[str, np.ndarray]
    sensor_state: Dict[str, Any]
    end_effector_pose: Pose
    sim_state: Dict[str, np.ndarray] | None


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
        joint_limit_penalty_factor = cfg.reward.joint_limit_penalty_factor
        joint_hz = cfg.control.frequencies.joint
        mujoco_hz = cfg.control.frequencies.mujoco

        assert mujoco_hz % joint_hz == 0, f"mujoco_hz ({mujoco_hz}) must be divisible by joint_hz ({joint_hz})"

        self.arm = arm
        self.backend = cfg.backend
        self.joint_limit_penalty_factor = joint_limit_penalty_factor
        self.joint_hz = joint_hz
        self.cartesian_hz = cfg.control.frequencies.cartesian
        self.tracking_progress_enabled = cfg.reward.tracking.progress
        self.joint_limit_penalty_enabled = cfg.reward.joint_limit_penalty
        self.sustained_duty_penalty_enabled = cfg.reward.sustained_duty_penalty
        self.termination_penalty_enabled = cfg.reward.termination_penalty
        self.pose_delta_diagnostics_enabled = cfg.training.pose_delta_diagnostics_enabled
        self.staging_enabled = cfg.control.staging.enabled
        self.initial_joint_range_percent = cfg.control.initial_joints.range_percent
        self.staging_speed_radians_per_second = cfg.control.staging.speed_radians_per_second
        self.staging_tolerance_radians = cfg.control.staging.tolerance_radians
        self.staging_max_seconds = cfg.control.staging.max_seconds
        self.output_dir = output_dir
        if self.backend == "real":
            self._save_servo_configuration()

        self.position_distance_weight = float(cfg.reward.pose_weights.position)
        self.rotation_primary_distance_weight = float(cfg.reward.pose_weights.rotation_primary)
        self.rotation_secondary_distance_weight = float(cfg.reward.pose_weights.rotation_secondary)
        self.gripper_distance_weight = float(cfg.reward.pose_weights.gripper)
        self.gripper_duty_weight = float(cfg.reward.pose_weights.gripper_duty)
        self.sustained_duty_allowance = float(cfg.reward.sustained_duty_allowance)
        self.sustained_duty_penalty_factor = float(cfg.reward.sustained_duty_penalty_factor)
        command_duration_seconds = 1.0 / self.cartesian_hz
        self.position_distance_scale = cfg.waypoint.position_speed_meters_per_second * command_duration_seconds
        self.rotation_distance_scale = cfg.waypoint.rotation_speed_radians_per_second * command_duration_seconds
        self.gripper_distance_scale = cfg.waypoint.gripper_speed_radians_per_second * command_duration_seconds

        # Hardcoding the ordered list of motors to ensure deterministic vectorization
        self.motor_order = MOTOR_ORDER

        self.duty_history_alpha = 1.0 - math.exp(-1.0 / (joint_hz * cfg.control.duty_history_seconds))
        self.duty_history = np.zeros(len(MOTOR_ORDER), dtype=np.float32)

        self.reset_reward_tracking()
        self.pose_delta_diagnostics = {}

    def get_end_effector_pose(self, state_dict: Dict[str, Any]) -> Pose:
        return self.arm.get_tcp_pose(state_dict)

    def _tcp_velocity(self, previous_pose: Pose, current_pose: Pose) -> np.ndarray:
        """
        Differenced from the previous observation rather than taken from a Jacobian, so simulation
        and hardware measure the same thing the same way and neither needs its own derivation.
        """
        return (previous_pose.delta_to(current_pose)[:6] * self.joint_hz).astype(np.float32)

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

        current_duty = np.array(
            [state_dict["Present_Load"][m] for m in self.motor_order],
            dtype=np.float32,
        )
        self.duty_history += self.duty_history_alpha * (np.abs(current_duty) - self.duty_history)

        end_effector_pose = self.get_end_effector_pose(state_dict)
        tcp_velocity = self._tcp_velocity(self.previous_end_effector_pose, end_effector_pose)
        self.previous_end_effector_pose = end_effector_pose

        obs = {
            "joint_positions": current_pos,
            "joint_velocities": current_vel,
            "gripper_duty": np.array([state_dict["Present_Load"]["gripper"]], dtype=np.float32),
            "duty_history": self.duty_history.copy(),
            "tcp_velocity": tcp_velocity,
        }

        return EnvironmentState(
            observation=obs,
            sensor_state=state_dict,
            end_effector_pose=end_effector_pose,
            sim_state=self.arm.sim_state() if self.backend == "sim" else None,
        )

    def reset(self) -> EnvironmentState:
        self.reset_reward_tracking()
        self.duty_history[:] = 0.0

        if self.backend == "sim":
            self.arm.reset_sim()
        elif self.backend == "real" and self.staging_enabled:
            self.arm.move_to_staging_pose(
                initial_joint_range_percent=self.initial_joint_range_percent,
                speed_radians_per_second=self.staging_speed_radians_per_second,
                tolerance_radians=self.staging_tolerance_radians,
                max_seconds=self.staging_max_seconds,
                output_dir=self.output_dir,
            )

        self.previous_end_effector_pose = self.arm.get_tcp_pose(self.arm.read_state())

        return self._get_obs()

    def reset_from_sim_state(self, qpos: np.ndarray, qvel: np.ndarray) -> EnvironmentState:
        if self.backend != "sim":
            raise ValueError("A recorded MuJoCo state can only initialize the simulation backend.")

        self.reset_reward_tracking()
        self.duty_history[:] = 0.0
        self.arm.restore_sim_state(qpos, qvel)
        self.previous_end_effector_pose = self.arm.get_tcp_pose(self.arm.read_state())
        return self._get_obs()

    def reset_cartesian_action_reward_tracking(self, cartesian_action_start_pose: Pose, cartesian_action: np.ndarray) -> None:
        desired_pose = self._compute_desired_pose(cartesian_action_start_pose, cartesian_action)
        (
            self.previous_position_distance,
            self.previous_primary_orientation_distance,
            self.previous_secondary_orientation_distance,
            self.previous_gripper_distance,
        ) = self._compute_pose_distances(cartesian_action_start_pose, desired_pose)

    def _save_servo_configuration(self) -> None:
        """
        Snapshots the servo firmware settings beside the MuJoCo model, for the same reason: a run is
        only interpretable if you know what the hardware was configured to do while it happened.
        """
        configuration_path = Path(self.output_dir) / "servo_configuration.json"
        configuration_path.write_text(json.dumps(self.arm.configuration, indent=2, sort_keys=True) + "\n")

    def reset_reward_tracking(self) -> None:
        self.previous_position_distance = 0.0
        self.previous_primary_orientation_distance = 0.0
        self.previous_secondary_orientation_distance = 0.0
        self.previous_gripper_distance = 0.0

    def read_camera(self):
        return self.arm.read_camera()

    def _compute_desired_pose(self, cartesian_action_start_pose: Pose, cartesian_action: np.ndarray) -> Pose:
        return cartesian_action_start_pose.apply_delta(cartesian_action)

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

    def _compute_joint_limit_penalty(self, requested_action: Dict[str, float], safe_action: Dict[str, float]) -> float:
        joint_limit_penalty = 0.0
        for motor in requested_action:
            diff = abs(requested_action[motor] - safe_action[motor])
            if diff > 0:
                joint_limit_penalty -= diff * self.joint_limit_penalty_factor
        return joint_limit_penalty

    def _compute_sustained_duty_penalty(self) -> float:
        excess = np.maximum(self.duty_history - self.sustained_duty_allowance, 0.0)
        return -self.sustained_duty_penalty_factor * float(excess.sum())

    def _compute_gripper_duty_distance(self, gripper_duty: float, desired_gripper_duty: float) -> float:
        return abs(float(desired_gripper_duty) - float(gripper_duty))

    def _compute_gripper_term(self, gripper_distance: float, gripper_duty_distance: float, desired_gripper_duty_active: bool) -> float:
        if desired_gripper_duty_active:
            return self.gripper_duty_weight * gripper_duty_distance
        return self.gripper_distance_weight * gripper_distance / self.gripper_distance_scale

    def _compute_termination_penalty(
        self,
        cartesian_action_terminated: bool,
        current_pose: Pose,
        desired_pose: Pose,
        gripper_duty_distance: float,
        desired_gripper_duty_active: bool,
    ) -> float:
        termination_penalty = 0.0
        if cartesian_action_terminated:
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
                    self._compute_gripper_term(gripper_distance, gripper_duty_distance, desired_gripper_duty_active),
                )
            )
        return termination_penalty

    def _compute_pose_delta_diagnostics(
        self,
        current_pose: Pose,
        desired_pose: Pose,
        cartesian_action: np.ndarray,
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
        desired_action = cartesian_action
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
            "desired_position_speed_meters_per_second": float(np.linalg.norm(desired_action[:3]) * self.cartesian_hz),
            "desired_rotation_delta_norm_radians": float(np.linalg.norm(desired_action[3:6])),
            "desired_rotation_speed_radians_per_second": float(np.linalg.norm(desired_action[3:6]) * self.cartesian_hz),
        }

    def compute_reward(
        self,
        requested_action: Dict[str, float],
        safe_action: Dict[str, float],
        cartesian_action: np.ndarray,
        current_pose: Pose,
        cartesian_action_start_pose: Pose,
        cartesian_action_terminated: bool,
        gripper_duty: float,
        desired_gripper_duty: float,
        desired_gripper_duty_active: bool,
    ) -> Tuple[float, Dict[str, float]]:
        desired_pose = self._compute_desired_pose(cartesian_action_start_pose, cartesian_action)
        gripper_duty_distance = self._compute_gripper_duty_distance(gripper_duty, desired_gripper_duty)
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
            self.rotation_primary_distance_weight * (self.previous_primary_orientation_distance - primary_orientation_distance) / self.rotation_distance_scale
        )
        secondary_orientation_reward = (
            self.rotation_secondary_distance_weight * (self.previous_secondary_orientation_distance - secondary_orientation_distance) / self.rotation_distance_scale
        )
        gripper_reward = self.gripper_distance_weight * (self.previous_gripper_distance - gripper_distance) / self.gripper_distance_scale
        self.previous_position_distance = position_distance
        self.previous_primary_orientation_distance = primary_orientation_distance
        self.previous_secondary_orientation_distance = secondary_orientation_distance
        self.previous_gripper_distance = gripper_distance

        joint_limit_penalty = self._compute_joint_limit_penalty(requested_action, safe_action)

        termination_penalty = self._compute_termination_penalty(
            cartesian_action_terminated,
            current_pose,
            desired_pose,
            gripper_duty_distance,
            desired_gripper_duty_active,
        )

        if self.pose_delta_diagnostics_enabled:
            self.pose_delta_diagnostics = self._compute_pose_delta_diagnostics(
                current_pose,
                desired_pose,
                cartesian_action,
            )

        reward_breakdown = {}
        if self.tracking_progress_enabled:
            reward_breakdown["position_reward"] = position_reward
            reward_breakdown["primary_orientation_reward"] = primary_orientation_reward
            reward_breakdown["secondary_orientation_reward"] = secondary_orientation_reward
            if not desired_gripper_duty_active:
                reward_breakdown["gripper_reward"] = gripper_reward
        if desired_gripper_duty_active:
            reward_breakdown["gripper_duty_penalty"] = -self.gripper_duty_weight * gripper_duty_distance
        if self.joint_limit_penalty_enabled:
            reward_breakdown["joint_limit_penalty"] = joint_limit_penalty
        if self.sustained_duty_penalty_enabled:
            reward_breakdown["sustained_duty_penalty"] = self._compute_sustained_duty_penalty()
        if self.termination_penalty_enabled:
            reward_breakdown["termination_penalty"] = termination_penalty

        total_reward = float(sum(reward_breakdown.values()))

        return total_reward, reward_breakdown

    def step(
        self,
        action: np.ndarray,
        joint_positions: np.ndarray,
        cartesian_action: np.ndarray,
        cartesian_action_start_pose: Pose,
        cartesian_action_terminated: bool,
        desired_gripper_duty: float,
        desired_gripper_duty_active: bool,
    ) -> Tuple[EnvironmentState, float, Dict[str, float]]:
        
        action_dict = {motor: float(duty) for motor, duty in zip(self.motor_order, action)}
        position_dict = {motor: float(pos) for motor, pos in zip(self.motor_order, joint_positions)}

        # 2. Command the duty, then let one control period elapse against it
        safe_action_dict = self.arm.write_duty(action_dict, position_dict)
        self.arm.advance_control_step()

        # 3. Get new observation
        state = self._get_obs()

        reward, reward_breakdown = self.compute_reward(
            requested_action=action_dict,
            safe_action=safe_action_dict,
            cartesian_action=cartesian_action,
            current_pose=state.end_effector_pose,
            cartesian_action_start_pose=cartesian_action_start_pose,
            cartesian_action_terminated=cartesian_action_terminated,
            gripper_duty=float(state.observation["gripper_duty"][0]),
            desired_gripper_duty=desired_gripper_duty,
            desired_gripper_duty_active=desired_gripper_duty_active,
        )

        return state, reward, reward_breakdown
