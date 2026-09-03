import math
import pickle
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import DictConfig

from robot_arm.numpy_policy import flatten_policy_observation
from robot_arm.robot_schema import policy_observation_dim

# jax.config.update("jax_default_matmul_precision", "highest")


class TrainState(NamedTuple):
    actor_params: dict
    critic_params: tuple
    target_critic_params: tuple
    log_ent_coef: jax.Array
    actor_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    ent_coef_optimizer_state: optax.OptState
    random_key: jax.Array
    update_count: jax.Array


class Batch(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    next_observations: jax.Array
    rewards: jax.Array
    dones: jax.Array


class UpdateMetrics(NamedTuple):
    actor_loss: jax.Array
    critic_loss: jax.Array
    ent_coef_loss: jax.Array
    ent_coef: jax.Array
    mean_q: jax.Array
    finite: jax.Array


class NumpyReplayBuffer:
    def __init__(self, capacity: int, observation_dim: int, action_dim: int, random_seed: int, device):
        self.capacity = capacity
        self.observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.next_observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0
        self.random = np.random.default_rng(random_seed)
        self.device = device

    def add(self, observation: dict[str, np.ndarray], next_observation: dict[str, np.ndarray], action: np.ndarray, reward: float, done: bool) -> None:
        self.observations[self.position] = flatten_policy_observation(observation)
        self.next_observations[self.position] = flatten_policy_observation(next_observation)
        self.actions[self.position] = action
        self.rewards[self.position, 0] = reward
        self.dones[self.position, 0] = done
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        indices = self.random.integers(self.size, size=batch_size)
        return Batch(
            observations=jax.device_put(self.observations[indices], self.device),
            actions=jax.device_put(self.actions[indices], self.device),
            next_observations=jax.device_put(self.next_observations[indices], self.device),
            rewards=jax.device_put(self.rewards[indices], self.device),
            dones=jax.device_put(self.dones[indices], self.device),
        )


def init_linear(random_key: jax.Array, input_dim: int, output_dim: int) -> dict[str, jax.Array]:
    weight_key, bias_key = jax.random.split(random_key)
    bound = 1.0 / math.sqrt(input_dim)
    return {
        "weight": jax.random.uniform(weight_key, (input_dim, output_dim), minval=-bound, maxval=bound),
        "bias": jax.random.uniform(bias_key, (output_dim,), minval=-bound, maxval=bound),
    }


def apply_linear(params: dict[str, jax.Array], inputs: jax.Array) -> jax.Array:
    return inputs @ params["weight"] + params["bias"]


def init_hidden_layers(random_key: jax.Array, input_dim: int, hidden_dims: tuple[int, ...]) -> tuple:
    layer_keys = jax.random.split(random_key, len(hidden_dims))
    layer_dims = (input_dim, *hidden_dims)
    return tuple(init_linear(layer_key, layer_dims[index], layer_dims[index + 1]) for index, layer_key in enumerate(layer_keys))


def apply_hidden_layers(params: tuple, inputs: jax.Array) -> jax.Array:
    activations = inputs
    for layer_params in params:
        activations = jax.nn.relu(apply_linear(layer_params, activations))
    return activations


def init_actor(random_key: jax.Array, observation_dim: int, action_dim: int, hidden_dims: tuple[int, ...]) -> dict:
    trunk_key, mean_key, log_std_key = jax.random.split(random_key, 3)
    return {
        "trunk": init_hidden_layers(trunk_key, observation_dim, hidden_dims),
        "mean": init_linear(mean_key, hidden_dims[-1], action_dim),
        "log_std": init_linear(log_std_key, hidden_dims[-1], action_dim),
    }


def actor_distribution(actor_params: dict, observations: jax.Array) -> tuple[jax.Array, jax.Array]:
    latent = apply_hidden_layers(actor_params["trunk"], observations)
    mean = apply_linear(actor_params["mean"], latent)
    log_std = jnp.clip(apply_linear(actor_params["log_std"], latent), -20.0, 2.0)
    return mean, log_std


def sample_actor(actor_params: dict, observations: jax.Array, random_key: jax.Array) -> tuple[jax.Array, jax.Array]:
    mean, log_std = actor_distribution(actor_params, observations)
    noise = jax.random.normal(random_key, mean.shape)
    unsquashed_actions = mean + jnp.exp(log_std) * noise
    actions = jnp.tanh(unsquashed_actions)
    gaussian_log_prob = -0.5 * (jnp.square(noise) + 2.0 * log_std + math.log(2.0 * math.pi))
    log_prob = jnp.sum(gaussian_log_prob - jnp.log(1.0 - jnp.square(actions) + 1e-6), axis=1, keepdims=True)
    return actions, log_prob


def init_q_network(random_key: jax.Array, observation_dim: int, action_dim: int, hidden_dims: tuple[int, ...]) -> dict:
    hidden_key, output_key = jax.random.split(random_key)
    return {
        "hidden": init_hidden_layers(hidden_key, observation_dim + action_dim, hidden_dims),
        "output": init_linear(output_key, hidden_dims[-1], 1),
    }


def apply_q_network(params: dict, observations: jax.Array, actions: jax.Array) -> jax.Array:
    latent = apply_hidden_layers(params["hidden"], jnp.concatenate((observations, actions), axis=1))
    return apply_linear(params["output"], latent)


def apply_twin_critics(params: tuple, observations: jax.Array, actions: jax.Array) -> tuple[jax.Array, jax.Array]:
    return tuple(apply_q_network(network_params, observations, actions) for network_params in params)


def polyak_update(target_params, source_params, tau: float):
    return jax.tree.map(lambda target, source: (1.0 - tau) * target + tau * source, target_params, source_params)


def build_update_step(actor_optimizer, critic_optimizer, ent_coef_optimizer, gamma: float, tau: float, target_entropy: float):
    def update_step(state: TrainState, batch: Batch) -> tuple[TrainState, UpdateMetrics]:
        random_key, actor_key, next_actor_key = jax.random.split(state.random_key, 3)
        entropy_coefficient = jnp.exp(state.log_ent_coef)
        next_actions, next_log_prob = sample_actor(state.actor_params, batch.next_observations, next_actor_key)
        next_q_values = apply_twin_critics(state.target_critic_params, batch.next_observations, next_actions)
        next_q_value = jnp.minimum(*next_q_values) - entropy_coefficient * next_log_prob
        target_q_value = jax.lax.stop_gradient(batch.rewards + (1.0 - batch.dones) * gamma * next_q_value)

        def critic_loss_fn(critic_params):
            current_q_values = apply_twin_critics(critic_params, batch.observations, batch.actions)
            loss = 0.5 * sum(jnp.mean(jnp.square(current_q_value - target_q_value)) for current_q_value in current_q_values)
            return loss, jnp.mean(jnp.minimum(*current_q_values))

        (critic_loss, mean_q), critic_gradients = jax.value_and_grad(critic_loss_fn, has_aux=True)(state.critic_params)
        critic_updates, critic_optimizer_state = critic_optimizer.update(critic_gradients, state.critic_optimizer_state, state.critic_params)
        critic_params = optax.apply_updates(state.critic_params, critic_updates)

        def actor_loss_fn(actor_params):
            policy_actions, log_prob = sample_actor(actor_params, batch.observations, actor_key)
            q_values = apply_twin_critics(critic_params, batch.observations, policy_actions)
            return jnp.mean(entropy_coefficient * log_prob - jnp.minimum(*q_values)), log_prob

        (actor_loss, policy_log_prob), actor_gradients = jax.value_and_grad(actor_loss_fn, has_aux=True)(state.actor_params)
        actor_updates, actor_optimizer_state = actor_optimizer.update(actor_gradients, state.actor_optimizer_state, state.actor_params)
        actor_params = optax.apply_updates(state.actor_params, actor_updates)

        def ent_coef_loss_fn(log_ent_coef):
            return -jnp.mean(log_ent_coef * jax.lax.stop_gradient(policy_log_prob + target_entropy))

        ent_coef_loss, ent_coef_gradients = jax.value_and_grad(ent_coef_loss_fn)(state.log_ent_coef)
        ent_coef_updates, ent_coef_optimizer_state = ent_coef_optimizer.update(ent_coef_gradients, state.ent_coef_optimizer_state, state.log_ent_coef)
        log_ent_coef = optax.apply_updates(state.log_ent_coef, ent_coef_updates)
        target_critic_params = polyak_update(state.target_critic_params, critic_params, tau)
        finite = jnp.logical_and(
            jnp.all(jnp.isfinite(jnp.array((actor_loss, critic_loss, ent_coef_loss, entropy_coefficient, mean_q)))),
            jax.tree.reduce(
                lambda result, value: jnp.logical_and(result, jnp.all(jnp.isfinite(value))),
                (actor_params, critic_params, target_critic_params, log_ent_coef),
                initializer=True,
            ),
        )
        new_state = TrainState(
            actor_params=actor_params,
            critic_params=critic_params,
            target_critic_params=target_critic_params,
            log_ent_coef=log_ent_coef,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
            ent_coef_optimizer_state=ent_coef_optimizer_state,
            random_key=random_key,
            update_count=state.update_count + 1,
        )
        metrics = UpdateMetrics(actor_loss, critic_loss, ent_coef_loss, entropy_coefficient, mean_q, finite)
        return new_state, metrics

    return jax.jit(update_step, donate_argnums=(0,))


def initialize_state(random_key: jax.Array, observation_dim: int, action_dim: int, hidden_dims: tuple[int, ...], learning_rate: float):
    actor_key, first_critic_key, second_critic_key, state_key = jax.random.split(random_key, 4)
    actor_params = init_actor(actor_key, observation_dim, action_dim, hidden_dims)
    critic_params = (
        init_q_network(first_critic_key, observation_dim, action_dim, hidden_dims),
        init_q_network(second_critic_key, observation_dim, action_dim, hidden_dims),
    )
    target_critic_params = jax.tree.map(jnp.copy, critic_params)
    actor_optimizer = optax.adam(learning_rate, eps=1e-8)
    critic_optimizer = optax.adam(learning_rate, eps=1e-8)
    ent_coef_optimizer = optax.adam(learning_rate, eps=1e-8)
    log_ent_coef = jnp.zeros((), dtype=jnp.float32)
    state = TrainState(
        actor_params=actor_params,
        critic_params=critic_params,
        target_critic_params=target_critic_params,
        log_ent_coef=log_ent_coef,
        actor_optimizer_state=actor_optimizer.init(actor_params),
        critic_optimizer_state=critic_optimizer.init(critic_params),
        ent_coef_optimizer_state=ent_coef_optimizer.init(log_ent_coef),
        random_key=state_key,
        update_count=jnp.zeros((), dtype=jnp.int32),
    )
    return state, actor_optimizer, critic_optimizer, ent_coef_optimizer


class JaxSAC:
    def __init__(self, cfg: DictConfig):
        self.observation_dim = policy_observation_dim(int(cfg.waypoint.cartesian_action_dim))
        self.action_dim = 6
        self.hidden_dims = tuple(int(hidden_dim) for hidden_dim in cfg.policy.net_arch)
        self.batch_size = int(cfg.training.batch_size)
        self.gradient_steps = int(cfg.training.gradient_steps)
        self.train_frequency = int(cfg.training.train_freq)
        self.broadcast_weights_every_n_steps = int(cfg.training.broadcast_weights_every_n_steps)
        assert self.broadcast_weights_every_n_steps % self.train_frequency == 0, "train_freq must divide broadcast_weights_every_n_steps"
        self.learning_starts = int(cfg.training.learning_starts)
        self.random_seed = int(cfg.seed)
        if cfg.device != "cuda":
            raise ValueError(f"JAX SAC requires device='cuda', got {cfg.device!r}")
        self.device = jax.devices("gpu")[0]
        self.replay_buffer = NumpyReplayBuffer(int(cfg.training.buffer_size), self.observation_dim, self.action_dim, self.random_seed, self.device)
        with jax.default_device(self.device):
            state, actor_optimizer, critic_optimizer, ent_coef_optimizer = initialize_state(
                jax.random.PRNGKey(self.random_seed),
                self.observation_dim,
                self.action_dim,
                self.hidden_dims,
                float(cfg.training.learning_rate),
            )
        self.state = state
        self.update_step = build_update_step(
            actor_optimizer,
            critic_optimizer,
            ent_coef_optimizer,
            float(cfg.training.gamma),
            float(cfg.training.tau),
            -float(self.action_dim),
        )

    def train(self, gradient_steps: int) -> dict[str, float]:
        assert gradient_steps > 0, "gradient_steps must be positive"
        metrics = None
        for _ in range(gradient_steps):
            self.state, metrics = self.update_step(self.state, self.replay_buffer.sample(self.batch_size))
        metrics.finite.block_until_ready()
        if not bool(metrics.finite):
            raise FloatingPointError("JAX SAC update produced non-finite values")
        return {
            "actor_loss": float(metrics.actor_loss),
            "critic_loss": float(metrics.critic_loss),
            "ent_coef_loss": float(metrics.ent_coef_loss),
            "ent_coef": float(metrics.ent_coef),
            "mean_q": float(metrics.mean_q),
        }

    def actor_params_numpy(self) -> dict:
        return jax.tree.map(lambda value: np.asarray(value), self.state.actor_params)

    def save(self, path: str) -> Path:
        checkpoint_path = Path(path).with_suffix(".pkl")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state": jax.tree.map(lambda value: np.asarray(value), self.state),
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "hidden_dims": self.hidden_dims,
            "random_seed": self.random_seed,
        }
        checkpoint_path.write_bytes(pickle.dumps(checkpoint))
        actor_params = self.actor_params_numpy()
        actor_arrays = {
            **{f"trunk_{index}_{name}": value for index, layer in enumerate(actor_params["trunk"]) for name, value in layer.items()},
            **{f"mean_{name}": value for name, value in actor_params["mean"].items()},
            **{f"log_std_{name}": value for name, value in actor_params["log_std"].items()},
            "random_seed": np.array(self.random_seed, dtype=np.int64),
        }
        np.savez(checkpoint_path.with_suffix(".actor.npz"), **actor_arrays)
        return checkpoint_path

    def load(self, path: str) -> None:
        checkpoint = pickle.loads(Path(path).read_bytes())
        if checkpoint["observation_dim"] != self.observation_dim or checkpoint["action_dim"] != self.action_dim or tuple(checkpoint["hidden_dims"]) != self.hidden_dims:
            raise ValueError("Checkpoint architecture does not match the configured JAX SAC architecture")
        self.state = jax.tree.map(lambda value: jax.device_put(value, self.device), checkpoint["state"])
