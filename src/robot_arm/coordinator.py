import numpy as np
from typing import Dict, Tuple, Any

from robot_arm.policies import Policy
from robot_arm.envs.env import RobotEnv


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
        training: bool,
    ):
        if low_level_hz % high_level_hz != 0:
            raise ValueError(
                f"low_level_hz ({low_level_hz}) must be divisible by high_level_hz ({high_level_hz})"
            )

        self.env = env
        self.chunk_size = low_level_hz // high_level_hz
        self.low_level_policy = low_level_policy
        self.high_level_policy = high_level_policy
        self.training = training

        self.global_step = 0

    def step(
        self, obs: Dict[str, np.ndarray], info: Dict[str, Any], instruction: str
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        total_reward = 0.0
        dense_trajectory = []
        low_level_transitions = [] # Collect transitions for the training buffer

        # 1. High Level Inference at 10Hz
        high_level_action = self.high_level_policy.get_action(
            obs, info, instruction=instruction
        )

        self.env.update_path(high_level_action)

        # 3. Step physics using the reactive RL agent
        for step_idx in range(self.chunk_size):

            low_level_action, _ = self.low_level_policy.predict(obs, deterministic=not self.training)

            next_obs, reward, terminated, truncated, info = self.env.step(
                low_level_action
            )
            total_reward += reward
            
            chunk_terminated = terminated or (obs["time_left"] == 0)

            # Record transition decoupled from the SB3 replay buffer directly
            low_level_transitions.append((
                obs,
                next_obs,
                low_level_action,
                reward,
                chunk_terminated,
                info
            ))

            self.global_step += 1
            obs = next_obs
            dense_trajectory.append(
                {
                    "joint_positions": obs["joint_positions"].copy(),
                    "action": low_level_action.copy(),
                }
            )

            if terminated or truncated:
                break

        info["dense_trajectory"] = dense_trajectory
        info["high_level_action"] = high_level_action.copy()
        
        # Add transitions array to info for extraction by external learner loop
        info["low_level_transitions"] = low_level_transitions

        return obs, total_reward, terminated, truncated, info
