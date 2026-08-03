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
from robot_arm.model_snapshot import snapshot_model_files


def get_waypoint_list():
    # Define our manual waypoints for a grabbing motion
    # 10D target: [x, y, z, r1, r2, r3, r4, r5, r6, gripper]
    # Neutral looking forward, gripper slightly open
    wp_hover_pose = Pose.from_euler([0.25, 0.0, 0.20], [0, 0.0, 0.0], 0.3, "XYZ", False)
    wp_hover = wp_hover_pose.as_10d()

    # Reach forward low, rotate wrist +45 deg (roll around X)
    wp_reach_box_pose = Pose.from_euler(
        [0.35, 0.0, 0.05], [np.pi / 2, 0, 0], 1, "XYZ", False
    )
    wp_reach_forward = wp_reach_box_pose.as_10d()

    # Move right and up, rotate wrist -45 deg (roll around X)
    wp_move_right_pose = Pose.from_euler(
        [0.35, 0, 0.05], [np.pi / 2, 0, 0], 0.3, "XYZ", False
    )
    wp_move_right = wp_move_right_pose.as_10d()

    # 4. Lift higher and closer to base: 15cm X (forward), 40cm Z (height)
    wp_lift_pose = Pose.from_euler(
        [0.45, 0.0, 0.20], [0.0, 0.0, 0.0], 0.3, "XYZ", False
    )
    wp_lift = wp_lift_pose.as_10d()

    return [
        wp_hover.copy(),
        wp_reach_forward.copy(),
        wp_move_right.copy(),
        wp_lift.copy(),
    ]


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Setup Policy
    task = "Sanity check: Initial pose, move front, move back."

    # Initialize a WaypointPolicy
    policy = WaypointPolicy(
        trajectory_length=cfg.waypoint.trajectory_length,
        speed=cfg.waypoint.speed,
    )

    # Load the latest trained low-level RL model
    low_level_policy = load_latest_low_level_policy()

    # Initialize recorder inside Hydra's output directory
    hydra_cfg = HydraConfig.get()
    snapshot_model_files(cfg.model_path, hydra_cfg.runtime.output_dir)
    output_dir = os.path.join(hydra_cfg.runtime.output_dir, "waypoint_recording")

    recorder = EpisodeRecorder(
        output_dir=output_dir,
        cfg=cfg,
        episode_name="waypoint_sanity_check",
    )

    # Sequence: Hover -> Reach Forward -> Move Right -> Lift
    policy.waypoints = get_waypoint_list()

    policy.current_wp_idx = 0

    env = make_env(cfg)

    runner = EpisodeRunner(
        cfg=cfg,
        env=env,
        low_level_policy=low_level_policy,
        high_level_policy=policy,
        training=False,
        recorder=recorder,
        replay_buffer=None,
        metrics_queue=None,
        weights_queue=None,
    )

    # Execute
    # No generate_waypoints=True for the sanity check, we supply our own
    runner.run_episode(
        task=task,
        generate_waypoints=False,
    )


if __name__ == "__main__":
    main()
