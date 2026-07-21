import numpy as np
from typing import Dict, Tuple, Any
from stable_baselines3.common.vec_env import VecEnv

from robot_arm.policies import Policy


class Coordinator:
    """
    Coordinates the 10Hz VLA inference with the 200Hz RobotEnv physics.
    Bridging the gap between the high-level policy and the batched lower-level environment.
    """

    def __init__(
        self,
        env: VecEnv,
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
        self, obs: Dict[str, np.ndarray], info: tuple, instruction: str
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, tuple]:
        batch_size = self.env.num_envs
        total_reward = np.zeros(batch_size, dtype=np.float32)
        dense_trajectory = [[] for _ in range(batch_size)]

        # 1. High Level Inference at 10Hz
        # High level policy handles waypoints internally based on the unbatched logic right now.
        # But VecEnv converts the obs dictionary into batched arrays.
        high_level_actions = []
        for i in range(batch_size):
            single_obs = {k: v[i] for k, v in obs.items()}
            # VecEnv returns info as a tuple of dicts
            single_action = self.high_level_policy.get_action(
                single_obs, info[i], instruction=instruction
            )
            high_level_actions.append(single_action)
            
        high_level_action_batch = np.array(high_level_actions, dtype=np.float32)

        # 2. Push the newly generated trajectories into the parallel wrappers
        for i in range(batch_size):
            self.env.env_method("update_path", high_level_action_batch[i], indices=[i])

        # 3. Step physics using the reactive RL agent
        for step_idx in range(self.chunk_size):

            low_level_action, _ = self.low_level_policy.predict(obs)

            next_obs, reward, done, next_info = self.env.step(
                low_level_action
            )
            
            terminated = np.array([i.get("TimeLimit.truncated", False) is False and d for d, i in zip(done, next_info)])
            truncated = np.array([i.get("TimeLimit.truncated", False) for i in next_info])
            total_reward += reward

            if self.training:
                # time_left is natively part of the obs now.
                actual_terminated = terminated | (obs["time_left"].squeeze(-1) == 0)
                
                # Manual insertion into Buffer per env
                self.low_level_policy.replay_buffer.add(
                    obs,
                    next_obs,
                    low_level_action,
                    reward,
                    actual_terminated,
                    next_info,
                )

                self.global_step += batch_size

                # Check bounds and train exactly as stable-baselines intends
                if (
                    self.global_step > self.low_level_policy.learning_starts
                    and self.global_step % self.low_level_policy.train_freq.frequency
                    == 0
                ):
                    self.low_level_policy.train(
                        gradient_steps=self.low_level_policy.gradient_steps,
                        batch_size=self.low_level_policy.batch_size,
                    )
                    # Tell SB3 logger to dump logs at standard intervals
                    if self.global_step % 1000 == 0:
                        self.low_level_policy.logger.dump(step=self.global_step)

            obs = next_obs
            
            for env_idx in range(batch_size):
                dense_trajectory[env_idx].append(
                    {
                        "joint_positions": obs["joint_positions"][env_idx].copy(),
                        "action": low_level_action[env_idx].copy(),
                    }
                )

            # Do not break out early; VecEnv resets individual environments dynamically under the hood!

        for env_idx in range(batch_size):
            next_info[env_idx]["dense_trajectory"] = dense_trajectory[env_idx]
            next_info[env_idx]["high_level_action"] = high_level_action_batch[env_idx].copy()

        return obs, total_reward, terminated, truncated, next_info
