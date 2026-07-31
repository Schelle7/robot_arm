import numpy as np
from typing import Dict, Tuple

from robot_arm.pose import Pose
from robot_arm.backends.arm import Arm


class RobotEnv:
    """
    Standard driver wrapper for the robotic arm.
    """

    def __init__(
        self,
        arm: Arm,
        pose_distance_weights: np.ndarray,
        delta_action_scale: float,
        violation_penalty_factor: float,
        low_level_hz: int,
    ):
        self.arm = arm
        self.delta_action_scale = delta_action_scale
        self.violation_penalty_factor = violation_penalty_factor
        self.low_level_hz = low_level_hz

        self.pose_distance_weights = np.array(pose_distance_weights, dtype=np.float32)

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

    @property
    def current_joint_angles(self) -> np.ndarray:
        state_dict = self.arm.read_state()
        return np.array(
            [state_dict["Present_Position"][m] for m in self.motor_order],
            dtype=np.float32,
        )

    def get_privileged_end_effector_pose(self) -> Pose:
        if (
            hasattr(self.arm, "get_end_effector_pose")
            or hasattr(self.arm, "backend_arm")
            and hasattr(
                self.arm.backend_arm, "get_end_effector_pose_forward_kinematics"
            )
        ):
            if (
                hasattr(self.arm, "backend_arm")
                and type(self.arm.backend_arm).__name__ == "RealArm"
            ):
                state_dict = self.arm.read_state()
                return self.arm.backend_arm.get_end_effector_pose_forward_kinematics(
                    state_dict["Present_Position"]
                )
            else:
                return self.arm.get_end_effector_pose()
        raise NotImplementedError("Arm backend does not support pose retrieval.")

    def _get_obs(self) -> Dict[str, np.ndarray]:
        state_dict = self.arm.read_state()
        current_pos = self.current_joint_angles
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
                "Path must contain at least 2 waypoints to define progressing segments."
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
        t_clipped[0] = min(t[0], 1.0)  # allow negatives on the first segment
        # the cliiping might prove to be problematic
        # the whole path reward might be problematic.
        # maybe a more complicated progress tracking is required, like deleting reached segments.

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

    def _compute_tracking_rewards(
        self, weighted_relative_pose: np.ndarray, weighted_trajectory: np.ndarray
    ) -> Tuple[float, float]:
        closest_pt, seg_idx, t = self._get_closest_path_point(
            weighted_relative_pose, weighted_trajectory
        )

        current_deviation = self._compute_path_deviation(
            weighted_relative_pose, closest_pt
        )
        current_progress = self._compute_path_progress(weighted_trajectory, seg_idx, t)

        # Improvement math
        dev_reward = self.previous_deviation - current_deviation
        prog_reward = current_progress - self.previous_progress

        self.previous_deviation = current_deviation
        self.previous_progress = current_progress

        return dev_reward, prog_reward

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
        weighted_relative_pose: np.ndarray,
        weighted_trajectory: np.ndarray,
    ) -> float:
        termination_penalty = 0.0
        if chunk_terminated:
            final_target_relative = weighted_trajectory[-1]
            end_distance = np.linalg.norm(
                final_target_relative - weighted_relative_pose
            )
            termination_penalty = -float(end_distance)
        return termination_penalty

    def compute_reward(
        self,
        requested_action: Dict[str, float],
        safe_action: Dict[str, float],
        high_level_action: np.ndarray,
        current_pose: Pose,
        chunk_start_pose: Pose,
        chunk_terminated: bool,
    ) -> Tuple[float, Dict[str, float]]:
        # We map our current position relative to where the chunk started.
        pos_diff = (
            current_pose.position - chunk_start_pose.position
        ) * self.pose_distance_weights[:3]

        # Use 6D rotation for trajectory math
        rot_diff_6d = (
            current_pose.as_rot_6d() - chunk_start_pose.as_rot_6d()
        ) * self.pose_distance_weights[3:9]

        gripper_diff = (
            np.array([current_pose.gripper - chunk_start_pose.gripper])
            * self.pose_distance_weights[9:]
        )

        # Combine back into a weighted 10D (pos + 6D rot + 1 Grip) array for path matching
        weighted_relative_pose = np.concatenate([pos_diff, rot_diff_6d, gripper_diff])

        weighted_trajectory = high_level_action * self.pose_distance_weights

        dev_reward, prog_reward = self._compute_tracking_rewards(
            weighted_relative_pose, weighted_trajectory
        )

        safety_penalty = self._compute_safety_penalty(requested_action, safe_action)

        termination_penalty = self._compute_termination_penalty(
            chunk_terminated, weighted_relative_pose, weighted_trajectory
        )

        total_reward = float(
            dev_reward + prog_reward + safety_penalty + termination_penalty
        )
        reward_breakdown = {
            "dev_reward": dev_reward,
            "prog_reward": prog_reward,
            "safety_penalty": safety_penalty,
            "termination_penalty": termination_penalty,
        }

        return total_reward, reward_breakdown

    def step(
        self,
        action: np.ndarray,
        high_level_action: np.ndarray,
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

        # 3. Apply safe action through the wrapper
        safe_action_dict = self.arm.write_goal(action_dict)

        # 4. Get new observation
        obs = self._get_obs()

        reward, reward_breakdown = self.compute_reward(
            requested_action=action_dict,
            safe_action=safe_action_dict,
            high_level_action=high_level_action,
            current_pose=self.get_privileged_end_effector_pose(),
            chunk_start_pose=privileged_chunk_start_pose,
            chunk_terminated=chunk_terminated,
        )

        return obs, reward, reward_breakdown
