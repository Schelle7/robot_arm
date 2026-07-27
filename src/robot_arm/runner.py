from omegaconf import DictConfig
from typing import Any

from robot_arm.envs.factory import make_env
from robot_arm.coordinator import Coordinator
from robot_arm.policies import Policy
from robot_arm.recorder import EpisodeRecorder


def execute_episode(
    cfg: DictConfig,
    policy: Policy,
    low_level_policy: Any,
    recorder: EpisodeRecorder,
    instruction: str,
    generate_waypoints: bool,
):
    """
    Core execution loop for an episode.
    Handles environment initialization (Steps 1 & 2) and the recording loop (Step 5).
    """
    # 1. & 2. Initialize exactly using the shared factory
    print("Initializing Environment...")
    env = make_env(cfg)

    coordinator = Coordinator(
        env=env,
        low_level_policy=low_level_policy,
        high_level_policy=policy,
        high_level_hz=cfg.frequencies.high_level,
        low_level_hz=cfg.frequencies.low_level,
        training=False,
    )

    max_steps = int(cfg.max_seconds * cfg.frequencies.high_level)

    # 5. Execute and record loop
    obs, info = env.reset()
    recorder.record_reset(obs, info, instruction)
    
    if generate_waypoints:
        policy.generate_grab_waypoints(
            box_pose_6d=info["privileged_box_pose_6d"],
            lift_height=cfg.training.lift_height,
            gripper_open=cfg.training.gripper_open,
            gripper_closed=cfg.training.gripper_closed,
        )

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
