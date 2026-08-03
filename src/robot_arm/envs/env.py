import numpy as np
from typing import Dict, Tuple
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation

from robot_arm.pose import Pose
from robot_arm.backends.arm import Arm


class RobotEnv:
    """
    Standard driver wrapper for the robotic arm.
    """

    def __init__(
        self,
        arm: Arm,
        cfg: DictConfig,
    ):
        delta_action_scale = cfg.control.action_scale_radians
        violation_penalty_factor = cfg.reward.violation_penalty_factor
        low_level_hz = cfg.control.frequencies.low_level
        mujoco_hz = cfg.control.frequencies.mujoco

        assert mujoco_hz % low_level_hz == 0, (
            f"mujoco_hz ({mujoco_hz}) must be divisible by low_level_hz ({low_level_hz})"
        )

        self.arm = arm
        self.delta_action_scale = delta_action_scale
        self.violation_penalty_factor = violation_penalty_factor
        self.low_level_hz = low_level_hz
        self.mujoco_steps_per_low_level_step = mujoco_hz // low_level_hz
        self.tracking_deviation_enabled = cfg.reward.tracking.deviation
        self.tracking_progress_enabled = cfg.reward.tracking.progress
        self.safety_penalty_enabled = cfg.reward.safety_penalty
        self.termination_penalty_enabled = cfg.reward.termination_penalty
        self.pose_delta_diagnostics_enabled = (
            cfg.training.pose_delta_diagnostics_enabled
        )

        self.position_distance_weight = float(cfg.reward.pose_weights.position)
        self.rotation_primary_distance_weight = float(
            cfg.reward.pose_weights.rotation_primary
        )
        self.rotation_secondary_distance_weight = float(
            cfg.reward.pose_weights.rotation_secondary
        )
        self.gripper_distance_weight = float(cfg.reward.pose_weights.gripper)

        # Hardcoding the ordered list of motors to ensure deterministic vectorization
        self.motor_order = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]

        self.previous_deviation = 0.0
        self.previous_progress = 0.0
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

    def _get_obs(self) -> Dict[str, np.ndarray]:
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

        return obs

    def reset(self) -> Dict[str, np.ndarray]:
        self.previous_deviation = 0.0
        self.previous_progress = 0.0

        # We don't magically reset the physical arm to zero, we just start observing from where it is
        # However, for simulation, the SimBackend handles advancing time and scene resets
        if hasattr(self.arm, "reset_sim"):
            self.arm.reset_sim()

        return self._get_obs()

    def read_camera(self):
        return self.arm.read_camera()

    def _get_closest_path_point(
        self, current_point: np.ndarray, path: np.ndarray
    ) -> Tuple[np.ndarray, int, float]:
        """
        Projects current_point onto the polyline path.
        Returns:
          - closest_point: The actual coordinate on the path.
          - segment_idx: The index of the path segment (A, B) it matched on.
          - t: The normalized scalar [0, 1] along the specific segment.
        """
        if path.shape[0] < 2:
            raise ValueError(
                "Delta path must contain at least one predicted delta."
            )

        # Segment start points (A) and end points (B)
        A = path[:-1]
        B = path[1:]

        # Vectors
        AB = B - A
        AP = current_point - A

        ab_sq = np.einsum("ij,ij->i", AB, AB)
        ab_sq_safe = ab_sq.copy()
        ab_sq_safe[ab_sq_safe == 0] = 1.0

        t = np.einsum("ij,ij->i", AP, AB) / ab_sq_safe

        t_clipped = np.clip(t, 0.0, 1.0)

        closest_points = A + t_clipped[:, np.newaxis] * AB
        distances = np.linalg.norm(current_point - closest_points, axis=1)

        # Note: Global argmin can cause "shortcut hacks" if the trajectory loops over itself.
        # For self-intersecting paths, this needs to be constrained to sequential active-segment tracking.
        min_idx = int(np.argmin(distances))
        return closest_points[min_idx], min_idx, float(t_clipped[min_idx])

    def _compute_path_deviation(
        self, current_point: np.ndarray, closest_point: np.ndarray
    ) -> float:
        return float(np.linalg.norm(current_point - closest_point))

    def _compute_path_progress(
        self, path: np.ndarray, segment_idx: int, t: float
    ) -> float:
        """
        Computes progress as distance traveled *along the line* from the very first waypoint.
        """
        if path.shape[0] < 2:
            raise ValueError(
                "Path must contain at least 2 waypoints to define progressing segments."
            )

        A = path[:-1]
        B = path[1:]
        AB = B - A
        ab_sq = np.einsum("ij,ij->i", AB, AB)
        segment_lengths = np.sqrt(ab_sq)

        past_progress = float(np.sum(segment_lengths[:segment_idx]))
        current_progress = t * segment_lengths[segment_idx]

        return float(past_progress + current_progress)

    def _axis_angular_distance(
        self, first_axis: np.ndarray, second_axis: np.ndarray
    ) -> float:
        dot_product = np.clip(np.dot(first_axis, second_axis), -1.0, 1.0)
        return float(np.arccos(dot_product))

    def _compute_weighted_pose_delta_and_path(
        self,
        current_pose: Pose,
        chunk_start_pose: Pose,
        high_level_delta_action: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pos_diff = current_pose.position - chunk_start_pose.position
        chunk_start_primary = chunk_start_pose.closing_axis
        chunk_start_secondary = chunk_start_pose.secondary_axis
        current_primary = current_pose.closing_axis
        current_secondary = current_pose.secondary_axis
        primary_delta = self._axis_angular_distance(
            chunk_start_primary, current_primary
        )
        secondary_delta = self._axis_angular_distance(
            chunk_start_secondary, current_secondary
        )
        gripper_diff = np.array(
            [current_pose.gripper - chunk_start_pose.gripper], dtype=np.float32
        )

        pose_delta = np.concatenate(
            [pos_diff, [primary_delta, secondary_delta], gripper_diff]
        )
        pose_weights = np.array(
            [
                self.position_distance_weight,
                self.position_distance_weight,
                self.position_distance_weight,
                self.rotation_primary_distance_weight,
                self.rotation_secondary_distance_weight,
                self.gripper_distance_weight,
            ],
            dtype=np.float32,
        )
        weighted_pose_delta = pose_delta * pose_weights
        delta_rotations = Rotation.from_rotvec(high_level_delta_action[:, 3:6])
        target_rotations = chunk_start_pose.rotation * delta_rotations
        target_matrices = target_rotations.as_matrix()
        target_primary = target_matrices[:, :, 0]
        target_secondary = target_matrices[:, :, 1]
        rotation_primary_path = np.array(
            [
                self._axis_angular_distance(chunk_start_primary, target)
                for target in target_primary
            ],
            dtype=np.float32,
        )
        rotation_secondary_path = np.array(
            [
                self._axis_angular_distance(chunk_start_secondary, target)
                for target in target_secondary
            ],
            dtype=np.float32,
        )
        delta_path = np.concatenate(
            [
                high_level_delta_action[:, :3],
                rotation_primary_path[:, np.newaxis],
                rotation_secondary_path[:, np.newaxis],
                high_level_delta_action[:, 6:7],
            ],
            axis=1,
        )
        weighted_delta_path = delta_path * pose_weights
        zero_delta = np.zeros((1, weighted_delta_path.shape[1]), dtype=np.float32)
        weighted_delta_path = np.concatenate(
            [zero_delta, weighted_delta_path], axis=0
        )

        return weighted_pose_delta, weighted_delta_path

    def _compute_tracking_rewards(
        self,
        weighted_pose_delta: np.ndarray,
        weighted_delta_path: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        closest_pt, seg_idx, t = self._get_closest_path_point(
            weighted_pose_delta, weighted_delta_path
        )

        current_deviation = self._compute_path_deviation(
            weighted_pose_delta, closest_pt
        )
        current_progress = self._compute_path_progress(weighted_delta_path, seg_idx, t)

        dev_reward = self.previous_deviation - current_deviation
        prog_reward = current_progress - self.previous_progress

        return dev_reward, prog_reward, current_deviation, current_progress

    def _compute_safety_penalty(
        self, requested_action: Dict[str, float], safe_action: Dict[str, float]
    ) -> float:
        safety_penalty = 0.0
        for motor in requested_action:
            diff = abs(requested_action[motor] - safe_action[motor])
            if diff > 0:
                safety_penalty -= diff * self.violation_penalty_factor
        return safety_penalty

    def _compute_termination_penalty(
        self,
        chunk_terminated: bool,
        weighted_pose_delta: np.ndarray,
        weighted_delta_path: np.ndarray,
    ) -> float:
        termination_penalty = 0.0
        if chunk_terminated:
            final_target_delta = weighted_delta_path[-1]
            end_distance = np.linalg.norm(
                final_target_delta - weighted_pose_delta
            )
            termination_penalty = -float(end_distance)
        return termination_penalty

    def _compute_pose_delta_diagnostics(
        self,
        weighted_pose_delta: np.ndarray,
        weighted_delta_path: np.ndarray,
    ) -> Dict[str, float]:
        desired_delta = weighted_delta_path[-1]
        return {
            "moved_delta_norm": float(np.linalg.norm(weighted_pose_delta)),
            "desired_delta_norm": float(np.linalg.norm(desired_delta)),
            "delta_error_norm": float(
                np.linalg.norm(desired_delta - weighted_pose_delta)
            ),
        }

    def compute_reward(
        self,
        requested_action: Dict[str, float],
        safe_action: Dict[str, float],
        high_level_delta_action: np.ndarray,
        current_pose: Pose,
        chunk_start_pose: Pose,
        chunk_terminated: bool,
    ) -> Tuple[float, Dict[str, float]]:
        weighted_pose_delta, weighted_delta_path = (
            self._compute_weighted_pose_delta_and_path(
                current_pose,
                chunk_start_pose,
                high_level_delta_action,
            )
        )

        (
            dev_reward,
            prog_reward,
            current_deviation,
            current_progress,
        ) = self._compute_tracking_rewards(
            weighted_pose_delta, weighted_delta_path
        )
        self.previous_deviation = current_deviation
        self.previous_progress = current_progress

        safety_penalty = self._compute_safety_penalty(requested_action, safe_action)

        termination_penalty = self._compute_termination_penalty(
            chunk_terminated, weighted_pose_delta, weighted_delta_path
        )

        if self.pose_delta_diagnostics_enabled:
            self.pose_delta_diagnostics = self._compute_pose_delta_diagnostics(
                weighted_pose_delta, weighted_delta_path
            )

        reward_breakdown = {}
        if self.tracking_deviation_enabled:
            reward_breakdown["dev_reward"] = dev_reward
        if self.tracking_progress_enabled:
            reward_breakdown["prog_reward"] = prog_reward
        if self.safety_penalty_enabled:
            reward_breakdown["safety_penalty"] = safety_penalty
        if self.termination_penalty_enabled:
            reward_breakdown["termination_penalty"] = termination_penalty

        total_reward = float(sum(reward_breakdown.values()))

        return total_reward, reward_breakdown

    def step(
        self,
        action: np.ndarray,
        high_level_delta_action: np.ndarray,
        privileged_chunk_start_pose: Pose,
        chunk_terminated: bool,
    ) -> Tuple[Dict[str, np.ndarray], float, Dict[str, float]]:

        # 1. Unscale delta action and add to current joint angles
        delta = action * self.delta_action_scale
        target_positions = self.current_joint_angles + delta

        # 2. Map target positions vector to dictionary and send to arm
        action_dict = {
            motor: float(pos) for motor, pos in zip(self.motor_order, target_positions)
        }

        # 3. Hold the joint target while MuJoCo advances through the control interval
        safe_action_dict = self.arm.write_goal(action_dict)
        for _ in range(1, self.mujoco_steps_per_low_level_step):
            safe_action_dict = self.arm.write_goal(action_dict)

        # 4. Get new observation
        obs = self._get_obs()

        reward, reward_breakdown = self.compute_reward(
            requested_action=action_dict,
            safe_action=safe_action_dict,
            high_level_delta_action=high_level_delta_action,
            current_pose=self.get_privileged_end_effector_pose(),
            chunk_start_pose=privileged_chunk_start_pose,
            chunk_terminated=chunk_terminated,
        )

        return obs, reward, reward_breakdown
