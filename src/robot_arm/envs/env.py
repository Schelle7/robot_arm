import json
import numpy as np
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
from omegaconf import DictConfig

from robot_arm.geometry.pose import Pose, axis_angular_distance
from robot_arm.backends.arm import Arm
from robot_arm.robot_schema import BOX_BODY_NAMES, HISTORY_APPLIED_DUTY_SLICE, HISTORY_FEATURE_NAMES, MOTOR_ORDER
from robot_arm.grasp_estimator import GraspEstimator


@dataclass
class EnvironmentState:
    observation: Dict[str, np.ndarray]
    sensor_state: Dict[str, Any]
    end_effector_pose: Pose
    sim_state: Dict[str, np.ndarray] | None
    grasp_confirmed: bool


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
        policy_history_steps = joint_hz * cfg.control.policy_history_seconds

        assert mujoco_hz % joint_hz == 0, f"mujoco_hz ({mujoco_hz}) must be divisible by joint_hz ({joint_hz})"
        assert policy_history_steps >= 2, "policy_history_seconds must span at least two low-level control intervals"
        assert float(policy_history_steps).is_integer(), "policy_history_seconds must contain an integer number of low-level control intervals"

        self.arm = arm
        self.grasp_estimator = GraspEstimator(cfg.control.grasp_estimation)
        self.max_box_distance_meters = float(cfg.control.grasp_estimation.max_box_distance_meters)
        self.backend = cfg.backend
        self.joint_limit_penalty_factor = joint_limit_penalty_factor
        self.joint_hz = joint_hz
        self.cartesian_hz = cfg.control.frequencies.cartesian
        self.policy_history_steps = int(policy_history_steps)
        self.policy_history = deque(maxlen=self.policy_history_steps)
        self.tracking_progress_enabled = cfg.reward.tracking.progress
        self.joint_limit_penalty_enabled = cfg.reward.joint_limit_penalty
        self.sustained_duty_penalty_enabled = cfg.reward.sustained_duty_penalty
        self.termination_penalty_enabled = cfg.reward.termination_penalty
        self.idle_action_penalty_enabled = cfg.reward.idle_action_penalty
        self.action_change_penalty_enabled = cfg.reward.action_change_penalty
        self.pose_delta_diagnostics_enabled = cfg.training.pose_delta_diagnostics_enabled
        self.initial_joint_range_percent = cfg.control.initial_joints.range_percent
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
        self.idle_action_threshold = float(cfg.reward.idle_action_threshold)
        self.idle_action_penalty_factor = float(cfg.reward.idle_action_penalty_factor)
        self.action_change_penalty_factor = float(cfg.reward.action_change_penalty_factor)
        command_duration_seconds = 1.0 / self.cartesian_hz
        self.position_distance_scale = cfg.waypoint.position_speed_meters_per_second * command_duration_seconds
        self.rotation_distance_scale = cfg.waypoint.rotation_speed_radians_per_second * command_duration_seconds
        self.gripper_distance_scale = cfg.waypoint.gripper_speed_radians_per_second * command_duration_seconds

        # Hardcoding the ordered list of motors to ensure deterministic vectorization
        self.motor_order = MOTOR_ORDER

        self.previous_action = np.zeros(len(MOTOR_ORDER), dtype=np.float32)

        self.reset_reward_tracking()
        self.pose_delta_diagnostics = {}

    def get_end_effector_pose(self, state_dict: Dict[str, Any]) -> Pose:
        return self.arm.get_tcp_pose(state_dict)

    def _reset_policy_history(self) -> None:
        self.policy_history.clear()
        empty_entry = np.zeros(len(HISTORY_FEATURE_NAMES), dtype=np.float32)
        for _ in range(self.policy_history_steps):
            self.policy_history.append(empty_entry.copy())

    def _set_motion_reference(self, joint_positions: np.ndarray, tcp_pose: Pose, sample_time_ns: int) -> None:
        self.previous_joint_positions = joint_positions.copy()
        self.previous_tcp_pose = tcp_pose
        self.previous_sample_time_ns = sample_time_ns

    def _consecutive_velocities(self, joint_positions: np.ndarray, tcp_pose: Pose, sample_time_ns: int) -> tuple[np.ndarray, np.ndarray]:
        elapsed_seconds = (sample_time_ns - self.previous_sample_time_ns) / 1_000_000_000
        assert elapsed_seconds > 0.0, "Sensor sample timestamps must increase"
        joint_velocity = (joint_positions - self.previous_joint_positions) / elapsed_seconds
        tcp_velocity = self.previous_tcp_pose.delta_to(tcp_pose)[:6] / elapsed_seconds
        self._set_motion_reference(joint_positions, tcp_pose, sample_time_ns)
        return joint_velocity.astype(np.float32), tcp_velocity.astype(np.float32)

    def _read_arm_state(self):
        state_dict = self.arm.read_state()
        joint_positions = np.array(
            [state_dict["Present_Position"][m] for m in self.motor_order],
            dtype=np.float32,
        )
        end_effector_pose = self.get_end_effector_pose(state_dict)
        return state_dict, joint_positions, end_effector_pose

    def _environment_state(
        self,
        state_dict: Dict[str, Any],
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        tcp_velocity: np.ndarray,
        end_effector_pose: Pose,
    ) -> EnvironmentState:
        obs = {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "tcp_position": end_effector_pose.position.copy(),
            "tcp_velocity": tcp_velocity,
            "gripper_duty": np.array([state_dict["Present_Load"]["gripper"]], dtype=np.float32),
            "policy_history": np.stack(self.policy_history),
        }

        grasp_confirmed = self._update_grasp_estimate(state_dict, joint_velocities, end_effector_pose)

        return EnvironmentState(
            observation=obs,
            sensor_state=state_dict,
            end_effector_pose=end_effector_pose,
            sim_state=self.arm.sim_state() if self.backend == "sim" else None,
            grasp_confirmed=grasp_confirmed,
        )

    def _update_grasp_estimate(
        self,
        state_dict: Dict[str, Any],
        joint_velocities: np.ndarray,
        end_effector_pose: Pose,
    ) -> bool:
        grasp_confirmed = self.grasp_estimator.update(
            state_dict["Present_Position"]["gripper"],
            joint_velocities[self.motor_order.index("gripper")],
            state_dict["Present_Load"]["gripper"],
            state_dict["sample_time_ns"],
        )
        if self.box_too_far(end_effector_pose):
            self.grasp_estimator.reset()
            grasp_confirmed = False

        return grasp_confirmed

    def box_too_far(self, end_effector_pose: Pose) -> bool:
        return bool(
            self.backend == "sim"
            and np.linalg.norm(
                self.arm.get_privileged_box_pose(BOX_BODY_NAMES[0]).position - end_effector_pose.position
            ) > self.max_box_distance_meters
        )

    def _initial_environment_state(self) -> EnvironmentState:
        state_dict, joint_positions, end_effector_pose = self._read_arm_state()
        self._set_motion_reference(joint_positions, end_effector_pose, state_dict["sample_time_ns"])
        zero_joint_velocity = np.zeros(len(MOTOR_ORDER), dtype=np.float32)
        zero_tcp_velocity = np.zeros(6, dtype=np.float32)
        return self._environment_state(
            state_dict,
            joint_positions,
            zero_joint_velocity,
            zero_tcp_velocity,
            end_effector_pose,
        )

    def reset(self, enable_added_weight: bool) -> EnvironmentState:
        self.grasp_estimator.reset()
        self.reset_reward_tracking()
        self.previous_action[:] = 0.0
        self._reset_policy_history()

        if self.backend == "sim":
            self.arm.reset_sim(enable_added_weight)

        return self._initial_environment_state()

    def reset_from_sim_state(self, qpos: np.ndarray, qvel: np.ndarray) -> EnvironmentState:
        self.grasp_estimator.reset()
        if self.backend != "sim":
            raise ValueError("A recorded MuJoCo state can only initialize the simulation backend.")

        self.reset_reward_tracking()
        self.previous_action[:] = 0.0
        self._reset_policy_history()
        self.arm.restore_sim_state(qpos, qvel)
        return self._initial_environment_state()

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

    def read_cameras(self):
        return self.arm.read_cameras()

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
        history = np.stack(self.policy_history)
        mean_absolute_duty = np.mean(np.abs(history[:, HISTORY_APPLIED_DUTY_SLICE]), axis=0)
        excess = np.maximum(mean_absolute_duty - self.sustained_duty_allowance, 0.0)
        return -self.sustained_duty_penalty_factor * float(excess.sum())

    def _compute_idle_action_penalty(self, policy_action: np.ndarray, time_left: float) -> float:
        # The shortfall is charged rather than a flat penalty below the threshold, so that a policy
        # sitting at zero action still sees a gradient out of it.
        shortfall = max(self.idle_action_threshold - float(np.abs(policy_action).sum()), 0.0)
        return -self.idle_action_penalty_factor * shortfall * time_left

    def _compute_gripper_duty_distance(self, gripper_duty: float, desired_gripper_duty: float) -> float:
        return abs(float(desired_gripper_duty) - float(gripper_duty))

    def _compute_gripper_term(self, gripper_distance: float, gripper_duty_distance: float, desired_gripper_duty_active: bool) -> float:
        if desired_gripper_duty_active:
            return self.gripper_duty_weight * gripper_duty_distance
        return self.gripper_distance_weight * gripper_distance / self.gripper_distance_scale

    def _compute_termination_penalty(
        self,
        cartesian_action_ends: bool,
        current_pose: Pose,
        desired_pose: Pose,
        gripper_duty_distance: float,
        desired_gripper_duty_active: bool,
    ) -> float:
        termination_penalty = 0.0
        if cartesian_action_ends:
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
        policy_action: np.ndarray,
        time_left: float,
        cartesian_action: np.ndarray,
        current_pose: Pose,
        cartesian_action_start_pose: Pose,
        cartesian_action_ends: bool,
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
            cartesian_action_ends,
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
        if self.idle_action_penalty_enabled:
            reward_breakdown["idle_action_penalty"] = self._compute_idle_action_penalty(policy_action, time_left)
        if self.action_change_penalty_enabled:
            # This fights chattering and may reduce long-term wear, but is primarily an attempted
            # workaround for the current oscillation. It can reduce responsiveness.
            action_change = policy_action - self.previous_action
            reward_breakdown["action_change_penalty"] = -self.action_change_penalty_factor * float(np.mean(np.square(action_change)))
        if self.termination_penalty_enabled:
            reward_breakdown["termination_penalty"] = termination_penalty

        total_reward = float(sum(reward_breakdown.values()))

        return total_reward, reward_breakdown

    def step(
        self,
        action: np.ndarray,
        joint_positions: np.ndarray,
        policy_action: np.ndarray,
        time_left: float,
        cartesian_action: np.ndarray,
        cartesian_action_start_pose: Pose,
        cartesian_action_ends: bool,
        desired_gripper_duty: float,
        desired_gripper_duty_active: bool,
    ) -> Tuple[EnvironmentState, float, Dict[str, float]]:
        
        action_dict = {motor: float(duty) for motor, duty in zip(self.motor_order, action)}
        position_dict = {motor: float(pos) for motor, pos in zip(self.motor_order, joint_positions)}

        # 2. Command the duty, then let one control period elapse against it
        safe_action_dict = self.arm.write_duty(action_dict, position_dict)
        self.arm.advance_control_step()

        state_dict, current_positions, end_effector_pose = self._read_arm_state()
        joint_velocities, tcp_velocity = self._consecutive_velocities(
            current_positions,
            end_effector_pose,
            state_dict["sample_time_ns"],
        )
        applied_duty = np.array([safe_action_dict[motor] for motor in self.motor_order], dtype=np.float32)
        history_entry = np.concatenate((policy_action, applied_duty, joint_velocities, tcp_velocity)).astype(np.float32)
        assert history_entry.shape == (len(HISTORY_FEATURE_NAMES),)
        self.policy_history.append(history_entry)
        state = self._environment_state(
            state_dict,
            current_positions,
            joint_velocities,
            tcp_velocity,
            end_effector_pose,
        )

        reward, reward_breakdown = self.compute_reward(
            requested_action=action_dict,
            safe_action=safe_action_dict,
            policy_action=policy_action,
            time_left=time_left,
            cartesian_action=cartesian_action,
            current_pose=state.end_effector_pose,
            cartesian_action_start_pose=cartesian_action_start_pose,
            cartesian_action_ends=cartesian_action_ends,
            gripper_duty=float(state.observation["gripper_duty"][0]),
            desired_gripper_duty=desired_gripper_duty,
            desired_gripper_duty_active=desired_gripper_duty_active,
        )
        self.previous_action[:] = policy_action

        return state, reward, reward_breakdown
