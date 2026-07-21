import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Any

from robot_arm.arm import Arm


class RobotEnv(gym.Env):
    """
    Standard MDP wrapper for the robotic arm.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        arm: Arm,
        max_seconds: float,
        trajectory_length: int,
        trajectory_dim: int,
        pose_distance_weights: np.ndarray,
        high_level_hz: int,
        low_level_hz: int,
        delta_action_scale: float,
        violation_penalty_factor: float,
    ):
        super().__init__()
        self.arm = arm
        self.max_seconds = max_seconds
        self.delta_action_scale = delta_action_scale
        self.violation_penalty_factor = violation_penalty_factor

        if low_level_hz % high_level_hz != 0:
            raise ValueError(
                f"low_level_hz ({low_level_hz}) must be cleanly divisible by high_level_hz ({high_level_hz})"
            )

        self.chunk_size = low_level_hz // high_level_hz
        self.step_in_chunk = 0

        self.trajectory_length = trajectory_length
        self.trajectory_dim = trajectory_dim
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

        # Normalized action space for delta control: [-1.0, 1.0].
        # The agent outputs a proportion of the maximum delta_action_scale per step.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        self.observation_space = spaces.Dict(
            {
                "joint_positions": spaces.Box(
                    low=-math.pi, high=math.pi, shape=(6,), dtype=np.float32
                ),
                "joint_velocities": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
                ),
                # The joint positions at the exact moment the VLA generated the trajectory
                "start_joint_positions": spaces.Box(
                    low=-math.pi, high=math.pi, shape=(6,), dtype=np.float32
                ),
                "high_level_action": spaces.Box(
                    # array of future relative waypoints from the VLA.
                    low=-1.0,
                    high=1.0,
                    shape=(trajectory_length, trajectory_dim),
                    dtype=np.float32,
                ),
                "time_left": spaces.Box(
                    low=0.0, high=np.inf, shape=(1,), dtype=np.float32
                ),
            }
        )
        # We store the current goals so they can be accessed in step() and _get_obs()
        self.current_trajectory_goal = np.zeros(
            (trajectory_length, trajectory_dim), dtype=np.float32
        )
        self.start_joint_positions = np.zeros(6, dtype=np.float32)

        self.previous_deviation = 0.0
        self.previous_progress = 0.0

    @property
    def current_joint_angles(self) -> np.ndarray:
        state_dict = self.arm.read_state()
        return np.array(
            [state_dict["Present_Position"][m] for m in self.motor_order],
            dtype=np.float32,
        )

    def _get_obs(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        state_dict = self.arm.read_state()
        current_pos = self.current_joint_angles
        current_vel = np.array(
            [state_dict["Present_Velocity"][m] for m in self.motor_order],
            dtype=np.float32,
        )

        obs = {
            "joint_positions": current_pos,
            "joint_velocities": current_vel,
            "start_joint_positions": self.start_joint_positions.copy(),
            "high_level_action": self.current_trajectory_goal.copy(),
            "time_left": np.array(
                [self.chunk_size - self.step_in_chunk - 1], dtype=np.float32
            ),
        }

        info = {}
        if hasattr(self.arm, "get_end_effector_pose_7d"):
            info["privileged_end_effector_pose_7d"] = (
                self.arm.get_end_effector_pose_7d()
            )
        if hasattr(self.arm, "get_privileged_box_pose_6d"):
            info["privileged_box_pose_6d"] = self.arm.get_privileged_box_pose_6d()

        return obs, info

    def reset(
        self, seed=None, options=None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0

        state_dict = self.arm.read_state()
        self.start_joint_positions = np.array(
            [state_dict["Present_Position"][m] for m in self.motor_order],
            dtype=np.float32,
        )

        # At reset, there is no trajectory plan yet, so the goal is simply zero relative deltas
        self.current_trajectory_goal = np.zeros(
            (self.trajectory_length, self.trajectory_dim), dtype=np.float32
        )

        self.previous_deviation = 0.0
        self.previous_progress = 0.0
        self.step_in_chunk = 0

        # We don't magically reset the physical arm to zero, we just start observing from where it is
        # However, for simulation, the SimBackend handles advancing time and scene resets
        if hasattr(self.arm, "reset_sim"):
            self.arm.reset_sim()

        obs, info = self._get_obs()
        return obs, info

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

    def update_path(self, new_trajectory: np.ndarray):
        """Called by the high-level controller whenever a new plan is generated."""
        
        self.chunk_start_pose = self.arm.get_end_effector_pose_7d()
        self.current_trajectory_goal = new_trajectory

        # We start a new high level instruction chunk, reset chunk time
        self.step_in_chunk = 0
        self.previous_deviation = 0.0
        self.previous_progress = 0.0

    def get_privileged_box_pose_6d(self) -> np.ndarray:
        """Helper method so SubprocVecEnv can unpack this without pickling the Arm."""
        return self.arm.get_privileged_box_pose_6d()

    def compute_reward(self, requested_action: Dict[str, float], safe_action: Dict[str, float]) -> float:
        try:
            # The trajectory goal is local deltas.
            # We must map our current position relative to where the chunk started.
            weighted_relative_pose = (
                self.arm.get_end_effector_pose_7d() - self.chunk_start_pose
            ) * self.pose_distance_weights
            weighted_trajectory = (
                self.current_trajectory_goal * self.pose_distance_weights
            )

            closest_pt, seg_idx, t = self._get_closest_path_point(
                weighted_relative_pose, weighted_trajectory
            )

            current_deviation = self._compute_path_deviation(
                weighted_relative_pose, closest_pt
            )
            current_progress = self._compute_path_progress(
                weighted_trajectory, seg_idx, t
            )

            # Improvement math
            dev_reward = self.previous_deviation - current_deviation
            prog_reward = current_progress - self.previous_progress

            self.previous_deviation = current_deviation
            self.previous_progress = current_progress

            # Safety penalty
            safety_penalty = 0.0
            for motor in requested_action:
                diff = abs(requested_action[motor] - safe_action[motor])
                if diff > 0:
                    safety_penalty -= diff * self.violation_penalty_factor

            return float(dev_reward + prog_reward + safety_penalty)

        except NotImplementedError:
            return np.nan

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        self.step_in_chunk += 1

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
        obs, info = self._get_obs()

        reward = self.compute_reward(requested_action=action_dict, safe_action=safe_action_dict)

        terminated = False
        truncated = False

        return obs, reward, terminated, truncated, info
