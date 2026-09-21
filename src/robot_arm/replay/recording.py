import glob
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from robot_arm.arms.sim_arm import build_desired_poses
from robot_arm.geometry.pose import Pose, axis_angular_distance
from robot_arm.run_paths import run_timestamp


def calculate_pose_delta(start_pose_10d, end_pose_10d):
    start_pose = Pose.from_10d(start_pose_10d)
    end_pose = Pose.from_10d(end_pose_10d)
    position_delta = end_pose.position - start_pose.position
    primary_orientation_distance = axis_angular_distance(
        start_pose.closing_axis,
        end_pose.closing_axis,
    )
    secondary_orientation_distance = axis_angular_distance(
        start_pose.secondary_axis,
        end_pose.secondary_axis,
    )
    gripper_delta = end_pose.gripper - start_pose.gripper
    return np.concatenate(
        [
            position_delta,
            [primary_orientation_distance, secondary_orientation_distance, gripper_delta],
        ]
    )


def rollout_timestamp(episode_path: str) -> datetime:
    """Every rollout script writes outputs/rollout/<script>/<date>/<time>/<recordings>/<episode>/episode.npz."""
    run_directory = Path(episode_path).resolve().parents[2]
    return run_timestamp(run_directory)


def find_latest_episode():
    outputs_dir = Path(__file__).resolve().parents[3] / "outputs"
    search_pattern = os.path.join(str(outputs_dir), "rollout", "*", "*", "*", "**", "episode.npz")
    files = glob.glob(search_pattern, recursive=True)

    if not files:
        return None

    return max(files, key=rollout_timestamp)


def load_recorded_config(episode_path: str) -> DictConfig:
    rollout_directory = Path(episode_path).resolve().parents[2]
    config_path = rollout_directory / ".hydra" / "config.yaml"
    return OmegaConf.load(config_path)


def recorded_model_path(episode_path: str, recorded_cfg: DictConfig) -> str:
    rollout_directory = Path(episode_path).resolve().parents[2]
    model_filename = Path(recorded_cfg.model_path).name
    return str(rollout_directory / "model" / model_filename)


def load_replay_recording(cfg: DictConfig):
    episode_path = cfg.episode_path
    if episode_path is None:
        episode_path = find_latest_episode()
        if episode_path is None:
            raise FileNotFoundError("Could not find any episode.npz files under outputs/rollout.")

    recorded_cfg = load_recorded_config(episode_path)
    data = np.load(episode_path, allow_pickle=True)
    num_actions = len(data["cartesian_action"])
    if recorded_cfg.backend == "sim":
        state_key = "qpos"
        num_states = len(data[state_key])
        if num_states != num_actions + 1:
            raise ValueError(f"Invalid sim replay: {num_states} states for {num_actions} Cartesian actions.")
    elif recorded_cfg.backend == "real":
        state_key = "joint_positions"
        num_states = len(data[state_key])
        if num_states not in (num_actions, num_actions + 1):
            raise ValueError(f"Invalid real replay: {num_states} states for {num_actions} Cartesian actions.")
    else:
        raise ValueError(f"Unsupported replay backend: {recorded_cfg.backend!r}.")
    return episode_path, recorded_cfg, data


def get_desired_poses(recorded_poses, desired_actions, frame_index):
    action_index = min(frame_index, len(desired_actions) - 1)
    desired_start_pose = Pose.from_10d(recorded_poses[action_index])
    return build_desired_poses(
        desired_start_pose,
        desired_actions[action_index],
    )
