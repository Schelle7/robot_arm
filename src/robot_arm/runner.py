import numpy as np
from typing import Optional
from omegaconf import DictConfig

from robot_arm.sim_arm import SimArm
from robot_arm.env import RobotEnv
from robot_arm.high_level_control_wrapper import HighLevelControlWrapper
from robot_arm.policies import Policy
from robot_arm.recorder import EpisodeRecorder


def execute_episode(cfg: DictConfig, policy: Policy, recorder: EpisodeRecorder, instruction: str):
    """
    Core execution loop for an episode.
    Handles environment initialization (Steps 1 & 2) and the recording loop (Step 5).
    """
    # 1. Initialize backend
    print("Initializing SimArm...")
    arm = SimArm(
        model_path=cfg.model_path, 
        height=cfg.camera.height, 
        width=cfg.camera.width
    )

    # 2. Track steps for this trajectory
    env = RobotEnv(
        arm=arm, 
        max_steps=cfg.max_steps, 
        height=cfg.camera.height, 
        width=cfg.camera.width
    )

    env = HighLevelControlWrapper(
        env=env,
        high_level_hz=cfg.frequencies.high_level,
        low_level_hz=cfg.frequencies.low_level
    )

    # 5. Execute and record loop
    obs, info = env.reset()
    print(f"Executing '{instruction}' and recording to {recorder.episode_dir}...")
    
    for step_idx in range(cfg.max_steps):
        action = policy.get_action(obs, instruction=instruction)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        recorder.step(step_idx, obs, reward=reward, info=info, instruction=instruction)
        
        if terminated or truncated:
            break
            
    recorder.save()
    print("Recording complete.")
    