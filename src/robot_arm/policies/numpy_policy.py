import numpy as np

from robot_arm.robot_schema import POLICY_OBSERVATION_NAMES


def apply_linear(params: dict[str, np.ndarray], inputs: np.ndarray) -> np.ndarray:
    return inputs @ params["weight"] + params["bias"]


def apply_hidden_layers(params: tuple, inputs: np.ndarray) -> np.ndarray:
    activations = inputs
    for layer_params in params:
        activations = np.maximum(apply_linear(layer_params, activations), 0.0)
    return activations


def actor_distribution(
    params: dict,
    observation: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    encoded_observation = np.concatenate(
        (
            apply_hidden_layers(params["history_encoder"], observation["history"]),
            apply_hidden_layers(params["state_encoder"], observation["state"]),
            apply_hidden_layers(params["goal_encoder"], observation["goal"]),
        ),
        axis=-1,
    )
    latent = apply_hidden_layers(params["fusion"], encoded_observation)
    mean = apply_linear(params["mean"], latent)
    log_std = np.clip(apply_linear(params["log_std"], latent), -20.0, 2.0)
    return mean, log_std


class NumpySACPolicy:
    def __init__(self, actor_params: dict, observation_sizes: dict[str, int], random_seed: int):
        self.actor_params = actor_params
        self.observation_sizes = observation_sizes
        self.random = np.random.default_rng(random_seed)

    def set_actor_params(self, actor_params: dict) -> None:
        self.actor_params = actor_params

    def predict(self, observation: dict[str, np.ndarray], deterministic: bool) -> tuple[np.ndarray, None]:
        policy_observation = {name: np.asarray(observation[name], dtype=np.float32) for name in POLICY_OBSERVATION_NAMES}
        mean, log_std = actor_distribution(self.actor_params, policy_observation)
        if deterministic:
            unsquashed_action = mean
        else:
            noise = self.random.standard_normal(mean.shape).astype(np.float32)
            unsquashed_action = mean + np.exp(log_std) * noise
        return np.tanh(unsquashed_action).astype(np.float32), None


def load_actor_layers(arrays, branch: str) -> tuple[dict[str, np.ndarray], ...]:
    prefix = f"{branch}_"
    indices = sorted({int(name.removeprefix(prefix).split("_")[0]) for name in arrays.files if name.startswith(prefix)})
    return tuple(
        {
            "weight": arrays[f"{branch}_{index}_weight"],
            "bias": arrays[f"{branch}_{index}_bias"],
        }
        for index in indices
    )


def load_numpy_policy(checkpoint_path: str) -> NumpySACPolicy:
    print(f"Loading joint policy from: {checkpoint_path}")
    arrays = np.load(checkpoint_path)
    observation_sizes = {name: int(arrays[f"observation_size_{name}"]) for name in POLICY_OBSERVATION_NAMES}
    actor_params = {
        "history_encoder": load_actor_layers(arrays, "history_encoder"),
        "state_encoder": load_actor_layers(arrays, "state_encoder"),
        "goal_encoder": load_actor_layers(arrays, "goal_encoder"),
        "fusion": load_actor_layers(arrays, "fusion"),
        "mean": {"weight": arrays["mean_weight"], "bias": arrays["mean_bias"]},
        "log_std": {"weight": arrays["log_std_weight"], "bias": arrays["log_std_bias"]},
    }
    return NumpySACPolicy(actor_params, observation_sizes, int(arrays["random_seed"]))
