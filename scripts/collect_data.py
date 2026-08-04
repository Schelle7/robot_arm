import os
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from robot_arm.recorder import EpisodeRecorder
from robot_arm.policies import WaypointPolicy, load_latest_low_level_policy
from robot_arm.runner import execute_episode


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    task = "Grip the red box."

    # Initialize the Waypoint Policy (Data generator)
    policy = WaypointPolicy(
        trajectory_length=cfg.waypoint.trajectory_length,
        low_level_hz=cfg.control.frequencies.low_level,
        position_speed_meters_per_second=cfg.waypoint.position_speed_meters_per_second,
        rotation_speed_radians_per_second=cfg.waypoint.rotation_speed_radians_per_second,
        gripper_speed_units_per_second=cfg.waypoint.gripper_speed_units_per_second,
    )

    # Load the latest trained low-level RL model
    low_level_policy = load_latest_low_level_policy()

    # Initialize recorder inside Hydra's output directory
    hydra_cfg = HydraConfig.get()
    output_dir = os.path.join(hydra_cfg.runtime.output_dir, "recordings")

    recorder = EpisodeRecorder(
        output_dir=output_dir,
        cfg=cfg,
        episode_name="waypoint_dataset_01",
    )

    # Execute
    execute_episode(
        cfg=cfg,
        policy=policy,
        low_level_policy=low_level_policy,
        recorder=recorder,
        task=task,
        generate_waypoints=True,
    )


if __name__ == "__main__":
    main()
