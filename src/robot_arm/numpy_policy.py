import numpy as np
from pathlib import Path

from robot_arm.robot_schema import POLICY_OBSERVATION_NAMES


def flatten_policy_observation(observation: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(observation[name], dtype=np.float32).reshape(-1) for name in POLICY_OBSERVATION_NAMES])


def apply_linear(params: dict[str, np.ndarray], inputs: np.ndarray) -> np.ndarray:
    return inputs @ params["weight"] + params["bias"]


def actor_distribution(params: dict, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    activations = observation
    for layer_params in params["trunk"]:
        activations = np.maximum(apply_linear(layer_params, activations), 0.0)
    mean = apply_linear(params["mean"], activations)
    log_std = np.clip(apply_linear(params["log_std"], activations), -20.0, 2.0)
    return mean, log_std


class NumpySACPolicy:
    def __init__(self, actor_params: dict, random_seed: int):
        self.actor_params = actor_params
        self.random = np.random.default_rng(random_seed)

    def set_actor_params(self, actor_params: dict) -> None:
        self.actor_params = actor_params

    def predict(self, observation: dict[str, np.ndarray], deterministic: bool) -> tuple[np.ndarray, None]:
        flat_observation = flatten_policy_observation(observation)
        mean, log_std = actor_distribution(self.actor_params, flat_observation)
        if deterministic:
            unsquashed_action = mean
        else:
            noise = self.random.standard_normal(mean.shape).astype(np.float32)
            unsquashed_action = mean + np.exp(log_std) * noise
        return np.tanh(unsquashed_action).astype(np.float32), None


def load_numpy_policy(checkpoint_path: str) -> NumpySACPolicy:
    actor_path = Path(checkpoint_path).with_suffix(".actor.npz")
    arrays = np.load(actor_path)
    trunk_indices = sorted({int(name.split("_")[1]) for name in arrays.files if name.startswith("trunk_")})
    actor_params = {
        "trunk": tuple({"weight": arrays[f"trunk_{index}_weight"], "bias": arrays[f"trunk_{index}_bias"]} for index in trunk_indices),
        "mean": {"weight": arrays["mean_weight"], "bias": arrays["mean_bias"]},
        "log_std": {"weight": arrays["log_std_weight"], "bias": arrays["log_std_bias"]},
    }
    return NumpySACPolicy(actor_params, int(arrays["random_seed"]))