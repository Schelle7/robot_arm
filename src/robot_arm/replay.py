import glob
import os
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from robot_arm.backends.sim_arm import build_desired_poses
from robot_arm.pose import Pose, axis_angular_distance


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


def find_latest_episode():
    outputs_dir = Path(__file__).resolve().parents[2] / "outputs"
    search_pattern = os.path.join(str(outputs_dir), "rollout_waypoint", "*", "*", "**", "episode.npz")
    files = glob.glob(search_pattern, recursive=True)

    if not files:
        return None

    def extract_datetime_key(filepath):
        parts = filepath.split(os.sep)
        idx = parts.index("rollout_waypoint")
        return (parts[idx + 1], parts[idx + 2])

    return max(files, key=extract_datetime_key)


def load_recorded_config(episode_path: str) -> DictConfig:
    rollout_directory = Path(episode_path).resolve().parents[2]
    config_path = rollout_directory / ".hydra" / "config.yaml"
    return OmegaConf.load(config_path)


def recorded_model_path(episode_path: str, recorded_cfg: DictConfig) -> str:
    rollout_directory = Path(episode_path).resolve().parents[2]
    model_filename = Path(recorded_cfg.model_path).name
    return str(rollout_directory / "model" / model_filename)


def load_recorded_timing(episode_path: str):
    recorded_cfg = load_recorded_config(episode_path)
    if recorded_cfg.backend != "sim":
        raise ValueError(f"replay_sim.py requires a simulation recording, got {recorded_cfg.backend!r}.")
    return recorded_cfg


def load_replay_recording(cfg: DictConfig):
    episode_path = cfg.episode_path
    if episode_path is None:
        episode_path = find_latest_episode()
        if episode_path is None:
            raise FileNotFoundError("Could not find any episode.npz files in outputs/rollout_waypoint.")

    recorded_cfg = load_recorded_timing(episode_path)
    data = np.load(episode_path, allow_pickle=True)
    num_states = len(data["qpos"])
    num_actions = len(data["cartesian_action_path"])
    if num_states != num_actions + 1:
        raise ValueError("Replay recording must contain one more state than Cartesian action paths.")
    return episode_path, recorded_cfg, data


def get_desired_poses(recorded_poses, desired_actions, frame_index):
    action_index = min(frame_index, len(desired_actions) - 1)
    desired_start_pose = Pose.from_10d(recorded_poses[action_index])
    return build_desired_poses(
        desired_start_pose,
        desired_actions[action_index],
    )
