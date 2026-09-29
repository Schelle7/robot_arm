import pickle
from pathlib import Path

import numpy as np

from robot_arm.robot_schema import POLICY_OBSERVATION_NAMES


def save_numpy_policy(path: Path, actor_params: dict, observation_sizes: dict[str, int], random_seed: int) -> None:
    arrays = {}
    for branch in ("history_encoder", "state_encoder", "goal_encoder", "fusion", "forward_head"):
        for index, layer in enumerate(actor_params[branch]):
            for name, value in layer.items():
                arrays[f"{branch}_{index}_{name}"] = np.asarray(value)
    for head in ("mean", "log_std", "forward_output"):
        for name, value in actor_params[head].items():
            arrays[f"{head}_{name}"] = np.asarray(value)
    for name in POLICY_OBSERVATION_NAMES:
        arrays[f"observation_size_{name}"] = np.array(observation_sizes[name], dtype=np.int64)
    arrays["random_seed"] = np.array(random_seed, dtype=np.int64)
    with path.open("wb") as file:
        np.savez(file, **arrays)


def convert_joint_checkpoint(source: Path, destination: Path) -> None:
    """Export action and motion inference weights from a trusted full training checkpoint."""
    checkpoint = pickle.loads(source.read_bytes())
    save_numpy_policy(destination, checkpoint["state"].actor_params, checkpoint["observation_sizes"], checkpoint["random_seed"])
