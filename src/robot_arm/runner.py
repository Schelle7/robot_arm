from omegaconf import DictConfig
from typing import Any

from robot_arm.backends.sim_arm import SimBackend
from robot_arm.envs.env import RobotEnv
from robot_arm.coordinator import Coordinator
from robot_arm.policies import Policy
from robot_arm.recorder import EpisodeRecorder


def execute_episode(
    cfg: DictConfig,
    policy: Policy,
    low_level_policy: Any,
    recorder: EpisodeRecorder,
    instruction: str,
):
    """
    Core execution loop for an episode.
    Handles environment initialization (Steps 1 & 2) and the recording loop (Step 5).
    """
    # 1. Initialize backend
    print("Initializing SimBackend...")
    arm = SimBackend(
        model_path=cfg.model_path, height=cfg.camera.height, width=cfg.camera.width
    )

    # 2. Track steps for this trajectory
    env = RobotEnv(
        arm=arm,
        max_seconds=cfg.max_seconds,
        trajectory_length=cfg.trajectory_length,
        trajectory_dim=cfg.trajectory_dim,
        pose_distance_weights=cfg.pose_distance_weights,
    )

    coordinator = Coordinator(
        env=env,
        low_level_policy=low_level_policy,
        high_level_policy=policy,
        high_level_hz=cfg.frequencies.high_level,
        low_level_hz=cfg.frequencies.low_level,
    )

    max_steps = int(cfg.max_seconds * cfg.frequencies.high_level)

    # 5. Execute and record loop
    obs, info = env.reset()
    print(f"Executing '{instruction}' and recording to {recorder.episode_dir}...")

    for step_idx in range(max_steps):
        obs, reward, terminated, truncated, info = coordinator.step(
            obs, info, instruction
        )

        recorder.step(step_idx, obs, reward=reward, info=info, instruction=instruction)

        if terminated or truncated:
            break

    recorder.save()
    print("Recording complete.")
