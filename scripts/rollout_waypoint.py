import os
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import numpy as np

from robot_arm.pose import Pose
from robot_arm.recorder import EpisodeRecorder
from robot_arm.policies import WaypointPolicy, load_latest_low_level_policy
from robot_arm.episode_runner import EpisodeRunner
from robot_arm.envs.factory import make_env


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
        chunk_size=cfg.frequencies.low_level // cfg.frequencies.high_level,
        episode_name="waypoint_sanity_check",
        record_sim_state=cfg["record_sim_state"],
    )

    # Define our manual waypoints for the sanity check
    # 10D target: [x, y, z, r1, r2, r3, r4, r5, r6, gripper]
    # Neutral looking forward, gripper slightly open
    wp_mid_pose = Pose.from_euler([0.25, 0.0, 0.20], [0.0, 0.0, 0.0], 0.3, "XYZ", False)
    wp_mid = wp_mid_pose.as_10d()
    
    # Reach left, rotate wrist +45 deg (roll around X)
    wp_left_pose = Pose.from_euler([0.25, 0.15, 0.20], [150, 0, 0], 0.3, "XYZ", False)
    wp_left = wp_left_pose.as_10d()

    # Reach right, rotate wrist -45 deg (roll around X)
    wp_right_pose = Pose.from_euler([0.25, -0.15, 0.20], [-np.pi/4, 0, 0], 0.3, "XYZ", False)
    wp_right = wp_right_pose.as_10d()

    # Sequence: Mid -> Left (+45) -> Mid -> Right (-45)
    policy.waypoints = [wp_mid.copy(), wp_left.copy(), wp_mid.copy(), wp_right.copy()]

    policy.current_wp_idx = 0

    env = make_env(cfg)

    runner = EpisodeRunner(
        cfg=cfg,
        env=env,
        low_level_policy=low_level_policy,
        high_level_policy=policy,
        training=False,
        recorder=recorder,
    )

    # Execute
    # No generate_waypoints=True for the sanity check, we supply our own
    runner.run_episode(
        instruction=instruction,
        generate_waypoints=False,
    )


if __name__ == "__main__":
    main()
