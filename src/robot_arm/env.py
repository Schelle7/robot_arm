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
        height: int,
        width: int,
        trajectory_length: int,
        trajectory_dim: int,
        pose_distance_weights: np.ndarray,
    ):
        super().__init__()
        self.arm = arm
        self.max_seconds = max_seconds
        self.current_step = 0
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

        # Rough joint limits in radians (from the MJCF / hardware limits)
        # We use symmetric pi for simplicity in normalized action spaces,
        # but could clamp exactly to the specific mechanical limits if needed.
        self.action_space = spaces.Box(
            low=-math.pi, high=math.pi, shape=(6,), dtype=np.float32
        )

        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Box(
                    low=0, high=255, shape=(height, width, 3), dtype=np.uint8
                ),
                "agent_pos": spaces.Box(
                    low=-math.pi, high=math.pi, shape=(6,), dtype=np.float32
                ),
                # The joint positions at the exact moment the VLA generated the trajectory
                "original_agent_pos": spaces.Box(
                    low=-math.pi, high=math.pi, shape=(6,), dtype=np.float32
                ),
                "low_level_trajectory_goal": spaces.Box(
                    # array of future waypoints from the VLA.
                    low=-1.0,
                    high=1.0,
                    shape=(trajectory_length, trajectory_dim),
                    dtype=np.float32,
                ),
            }
        )
        # We store the current goals so they can be accessed in step() and _get_obs()
        self.current_trajectory_goal = np.zeros(
            (trajectory_length, trajectory_dim), dtype=np.float32
        )
        self.original_agent_pos = np.zeros(6, dtype=np.float32)

        self.previous_deviation = 0.0
        self.previous_progress = 0.0

    def _get_obs(self) -> Dict[str, np.ndarray]:
        state_dict = self.arm.read_state()
        current_pos = np.array(
            [state_dict["Present_Position"][m] for m in self.motor_order],
            dtype=np.float32,
        )

        pixels = self.arm.read_image()

        return {
            "pixels": pixels,
            "agent_pos": current_pos,
            "original_agent_pos": self.original_agent_pos.copy(),
            "low_level_trajectory_goal": self.current_trajectory_goal.copy(),
        }

    def reset(
        self, seed=None, options=None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0

        if (
            options is None
            or "low_level_trajectory_goal" not in options
            or "original_agent_pos" not in options
        ):
            raise ValueError(
                "Must provide 'low_level_trajectory_goal' and 'original_agent_pos' in reset options."
            )

        self.current_trajectory_goal = np.array(
            options["low_level_trajectory_goal"], dtype=np.float32
        )
        self.original_agent_pos = np.array(
            options["original_agent_pos"], dtype=np.float32
        )

        self.previous_deviation = 0.0
        self.previous_progress = 0.0

        # We don't magically reset the physical arm to zero, we just start observing from where it is
        # However, for simulation, the SimArm backend handles advancing time

        obs = self._get_obs()
        return obs, {}

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

    def update_trajectory(self, new_trajectory: np.ndarray):
        """Called by the high-level controller whenever a new plan is generated."""
        self.current_trajectory_goal = new_trajectory
        self.original_agent_pos = self.arm.get_pinch_point()

        self.previous_deviation = 0.0
        self.previous_progress = 0.0

    def compute_reward(self) -> float:
        try:
            current_pinch = self.arm.get_pinch_point() * self.pose_distance_weights
            trajectory = self.current_trajectory_goal * self.pose_distance_weights

            closest_pt, seg_idx, t = self._get_closest_path_point(
                current_pinch, trajectory
            )

            current_deviation = self._compute_path_deviation(current_pinch, closest_pt)
            current_progress = self._compute_path_progress(trajectory, seg_idx, t)

            # Improvement math
            dev_reward = self.previous_deviation - current_deviation
            prog_reward = current_progress - self.previous_progress

            self.previous_deviation = current_deviation
            self.previous_progress = current_progress

            return float(dev_reward + prog_reward)

        except NotImplementedError:
            return np.nan

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.current_step += 1

        # 1. Map action vector to dictionary and send to arm
        action_dict = {
            motor: float(action[i]) for i, motor in enumerate(self.motor_order)
        }
        self.arm.write_goal(action_dict)

        # 2. Get new observation
        obs = self._get_obs()
        current_pos = obs["agent_pos"][:6]

        # 3. Environment logic for VLA / behavior cloning
        reward = self.compute_reward()
        terminated = False  # VLA episodes run until max_seconds is hit, handled by caller loop boundary
        truncated = False

        info = {"current_pos": current_pos, "action": action}

        return obs, reward, terminated, truncated, info
