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

    def __init__(self, arm: Arm, max_steps: int, height: int, width: int):
        super().__init__()
        self.arm = arm
        self.max_steps = max_steps
        self.current_step = 0
        
        # Hardcoding the ordered list of motors to ensure deterministic vectorization
        self.motor_order = [
            "shoulder_pan", 
            "shoulder_lift", 
            "elbow_flex", 
            "wrist_flex", 
            "wrist_roll", 
            "gripper"
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
                    low=-math.pi, high=math.pi, shape=(12,), dtype=np.float32
                ),
            }
        )
        
        self.target_positions = np.zeros(6, dtype=np.float32)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        state_dict = self.arm.read_state()
        current_pos = np.array([
            state_dict["Present_Position"][m] for m in self.motor_order
        ], dtype=np.float32)
        
        agent_pos = np.concatenate([current_pos, self.target_positions])
        pixels = self.arm.read_image()
        
        return {
            "pixels": pixels,
            "agent_pos": agent_pos
        }

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        
        # Sample a random target within safe limits (-1.5 to 1.5 rad to avoid extreme crashes)
        self.target_positions = self.np_random.uniform(
            low=-1.5, high=1.5, size=(6,)
        ).astype(np.float32)
        
        # We don't magically reset the physical arm to zero, we just start observing from where it is
        # However, for simulation, the SimArm backend handles advancing time
        
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        
        # 1. Map action vector to dictionary and send to arm
        action_dict = {
            motor: float(action[i]) for i, motor in enumerate(self.motor_order)
        }
        self.arm.write_goal(action_dict)
        
        # 2. Get new observation
        obs = self._get_obs()
        current_pos = obs["agent_pos"][:6]
        
        # 3. Compute reward (negative L2 distance -> max reward is 0)
        distance = float(np.linalg.norm(current_pos - self.target_positions))
        reward = -distance
        
        # 4. Check termination criteria
        terminated = distance < 0.1  # Reached goal within 0.1 rad roughly
        truncated = self.current_step >= self.max_steps
        
        info = {
            "current_pos": current_pos,
            "action": action
        }
        
        return obs, reward, terminated, truncated, info
