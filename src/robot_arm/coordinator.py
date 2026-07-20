import numpy as np
from typing import Dict, Tuple, Any

from robot_arm.policies import Policy
from robot_arm.env import RobotEnv


class Coordinator:
    """
    Coordinates the 10Hz VLA inference with the 200Hz RobotEnv physics.
    Bridging the gap between the high-level policy and the lower-level environment.
    """

    def __init__(
        self,
        env: RobotEnv,
        low_level_policy,
        high_level_policy: Policy,
        high_level_hz: int,
        low_level_hz: int,
    ):
        if low_level_hz % high_level_hz != 0:
            raise ValueError(
                f"low_level_hz ({low_level_hz}) must be divisible by high_level_hz ({high_level_hz})"
            )

        self.env = env
        self.skip_frames = low_level_hz // high_level_hz
        self.low_level_policy = low_level_policy
        self.high_level_policy = high_level_policy

    def step(
        self, obs: Dict[str, np.ndarray], info: Dict[str, Any], instruction: str
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        total_reward = 0.0
        dense_trajectory = []

        # 1. High Level Inference at 10Hz
        high_level_action = self.high_level_policy.get_action(
            obs, info, instruction=instruction
        )

        # 2. Update MDP configuration for the 200Hz steps
        current_pos_dict = self.env.arm.read_state()
        start_pos = np.array(
            [current_pos_dict["Present_Position"][m] for m in self.env.motor_order],
            dtype=np.float32,
        )

        # later on I will have to make this parallel.
        # for now I'll just start getting it to run at all.

        target_pos = high_level_action

        self.env.update_path(target_pos)

        # 3. Step physics using the reactive RL agent
        for step_idx in range(self.skip_frames):
            time_left = self.skip_frames - step_idx - 1
            
            # Format the observation for the SAC policy
            rl_obs = {
                "agent_pos": obs["agent_pos"],
                "agent_vel": obs["agent_vel"],
                "low_level_trajectory_goal": obs["low_level_trajectory_goal"].flatten(),
                "time_left": np.array([time_left], dtype=np.float32),
            }
            
            # TODO: Uncomment when SAC policy is hooked up
            # low_level_action, _ = self.low_level_policy.predict(rl_obs, deterministic=True)
            low_level_action = np.zeros(6, dtype=np.float32) # Temporary fallback
            
            obs, reward, terminated, truncated, info = self.env.step(low_level_action)
            total_reward += reward

            dense_trajectory.append(
                {
                    "agent_pos": obs["agent_pos"].copy(),
                    "action": low_level_action.copy(),
                }
            )

            if terminated or truncated:
                break

        info["dense_trajectory"] = dense_trajectory
        info["high_level_action"] = high_level_action.copy()

        return obs, total_reward, terminated, truncated, info
