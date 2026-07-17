import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Any

class HighLevelControlWrapper(gym.Wrapper):
    """
    Wraps the underlying 200Hz RobotEnv to present a 10Hz interface to the VLA.
    Repeats the high-level action over the intermediate steps and buffers the 
    dense physical state to info dict for recording.
    """
    def __init__(self, env: gym.Env, high_level_hz: int, low_level_hz: int):
        super().__init__(env)
        if low_level_hz % high_level_hz != 0:
            raise ValueError(f"low_level_hz ({low_level_hz}) must be divisible by high_level_hz ({high_level_hz})")
            
        self.skip_frames = low_level_hz // high_level_hz
        
    def step(self, high_level_action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        total_reward = 0.0
        dense_trajectory = []
        
        # High level action is an absolute position target.
        current_pos = self.env.unwrapped.arm.read_state()
        start_pos = np.array([
            current_pos["Present_Position"][m] for m in self.env.unwrapped.motor_order
        ], dtype=np.float32)
        
        target_pos = high_level_action
        
        # Interpolate the low-level actions smoothly over the skip frames
        # have to repalce it with something more sensible later on / that means a controller that can learn PWM
        low_level_actions = np.linspace(start_pos, target_pos, self.skip_frames)
        
        for i in range(self.skip_frames):
            low_level_action = low_level_actions[i]
            obs, reward, terminated, truncated, info = self.env.step(low_level_action)
            total_reward += reward
            
            # Buffer the dense state and the low level action inside info
            dense_trajectory.append({
                "agent_pos": obs["agent_pos"].copy(),
                "action": low_level_action.copy()
            })
            
            if terminated or truncated:
                break
                
        info["dense_trajectory"] = dense_trajectory
        info["high_level_action"] = high_level_action.copy()
        
        return obs, total_reward, terminated, truncated, info
