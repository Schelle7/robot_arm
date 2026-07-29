import os
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import numpy as np

from robot_arm.recorder import EpisodeRecorder
from robot_arm.policies import WaypointPolicy, load_latest_low_level_policy
from robot_arm.runner import execute_episode


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Setup Policy
    instruction = "Sanity check: Initial pose, move front, move back."

    # Initialize a WaypointPolicy
    policy = WaypointPolicy(
        trajectory_length=cfg.trajectory_length,
        speed=cfg.training.waypoint_speed,
    )

    # Load the latest trained low-level RL model
    low_level_policy = load_latest_low_level_policy()

    # Initialize recorder inside Hydra's output directory
    hydra_cfg = HydraConfig.get()
    output_dir = os.path.join(hydra_cfg.runtime.output_dir, "waypoint_recording")

    recorder = EpisodeRecorder(
        output_dir=output_dir,
        jpeg_quality=cfg.camera.jpeg_quality,
        episode_name="waypoint_sanity_check",
    )

    # Define our manual waypoints for the sanity check
    # 7D target: [x, y, z, roll, pitch, yaw, gripper]

    # Standard neutral looking forward, gripper slightly open
    wp_mid = np.array([0.25, 0.0, 0.20, 0.0, 0.0, 0.0, 0.3], dtype=np.float32)
    # Reach forward 15cm
    wp_front = np.array([0.40, 0.0, 0.20, 0.0, 0.0, 0.0, 0.3], dtype=np.float32)

    # Sequence: Mid -> Front -> Mid -> Close gripper
    policy.waypoints = [wp_mid.copy(), wp_front.copy(), wp_mid.copy()]
    # Close gripper on the last waypoint
    wp_close = wp_mid.copy()
    wp_close[6] = 0.05  # closed
    policy.waypoints.append(wp_close)

    policy.current_wp_idx = 0

    # Execute
    execute_episode(
        cfg=cfg,
        policy=policy,
        low_level_policy=low_level_policy,
        recorder=recorder,
        instruction=instruction,
        generate_waypoints=False,
    )


if __name__ == "__main__":
    main()
