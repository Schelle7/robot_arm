import glob
import os
from pathlib import Path

import numpy as np

from robot_arm.policies.numpy_policy import load_numpy_policy


class FixedDutyPolicy:
    def __init__(self, duties: np.ndarray):
        self.duties = np.asarray(duties, dtype=np.float32)

    def predict(self, observation, deterministic):
        return self.duties.copy(), None


def find_latest_low_level_checkpoint() -> str:
    """
    Loads the most recent low-level SAC policy from the outputs/ directory.
    Searches the directory structure for the newest final checkpoint.
    """
    outputs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "outputs"))
    if not os.path.exists(outputs_dir):
        raise FileNotFoundError(f"Outputs directory not found at {outputs_dir}.")

    search_pattern = os.path.join(
        outputs_dir,
        "train_low_level",
        "*",
        "*",
        "checkpoints",
        "jax_sac_final_*.pkl",
    )
    checkpoints = glob.glob(search_pattern)

    if not checkpoints:
        raise FileNotFoundError("No final low-level policy checkpoints found in any outputs directory.")

    # Sort by the YYYY-MM-DD and HH-MM-SS folder names implicitly found in the path
    # Path structure: .../outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/jax_sac...pkl
    def extract_datetime_key(filepath):
        parts = filepath.split(os.sep)
        return (parts[-4], parts[-3])

    latest_checkpoint = max(checkpoints, key=extract_datetime_key)

    return latest_checkpoint


def resolve_low_level_checkpoint(policy_name: str) -> str:
    if policy_name == "latest":
        return find_latest_low_level_checkpoint()

    policy_path = Path(policy_name)
    if not policy_path.is_absolute():
        policy_path = Path(__file__).resolve().parents[3] / policy_path
    return str(policy_path.resolve())


def load_low_level_policy(checkpoint_path: str):
    print(f"Loading low level policy from: {checkpoint_path}")
    return load_numpy_policy(checkpoint_path)


def load_latest_low_level_policy():
    return load_low_level_policy(find_latest_low_level_checkpoint())
