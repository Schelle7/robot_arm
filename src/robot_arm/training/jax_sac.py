import math
import pickle
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import DictConfig

from robot_arm.robot_schema import POLICY_OBSERVATION_NAMES, STATE_JOINT_VELOCITY_SLICE, STATE_TCP_VELOCITY_SLICE, policy_observation_sizes
from robot_arm.training.replay_buffer import Batch, MixedReplayBuffer

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


class UpdateMetrics(NamedTuple):
    actor_loss: jax.Array
    actor_total_loss: jax.Array
    forward_loss: jax.Array
    forward_joint_loss: jax.Array
    forward_tcp_loss: jax.Array
    critic_loss: jax.Array
    critic_total_loss: jax.Array
    critic_forward_loss: jax.Array
    critic_forward_joint_loss: jax.Array
    critic_forward_tcp_loss: jax.Array
    ent_coef_loss: jax.Array
    ent_coef: jax.Array
    mean_q: jax.Array
    finite: jax.Array


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


def init_actor(
    random_key: jax.Array,
    observation_sizes: dict[str, int],
    action_dim: int,
    history_encoder_dims: tuple[int, ...],
    state_encoder_dims: tuple[int, ...],
    goal_encoder_dims: tuple[int, ...],
    fusion_dims: tuple[int, ...],
    forward_head_dims: tuple[int, ...],
) -> dict:
    history_key, state_key, goal_key, fusion_key, mean_key, log_std_key = jax.random.split(random_key, 6)
    forward_key, forward_output_key = jax.random.split(jax.random.fold_in(random_key, 1))
    fusion_input_dim = history_encoder_dims[-1] + state_encoder_dims[-1] + goal_encoder_dims[-1]
    return {
        "history_encoder": init_hidden_layers(history_key, observation_sizes["history"], history_encoder_dims),
        "state_encoder": init_hidden_layers(state_key, observation_sizes["state"], state_encoder_dims),
        "goal_encoder": init_hidden_layers(goal_key, observation_sizes["goal"], goal_encoder_dims),
        "fusion": init_hidden_layers(fusion_key, fusion_input_dim, fusion_dims),
        "mean": init_linear(mean_key, fusion_dims[-1], action_dim),
        "log_std": init_linear(log_std_key, fusion_dims[-1], action_dim),
        "forward_head": init_hidden_layers(forward_key, history_encoder_dims[-1] + state_encoder_dims[-1] + action_dim, forward_head_dims),
        "forward_output": init_linear(forward_output_key, forward_head_dims[-1], 12),
    }


def predict_forward_motion(params: dict, observations: dict[str, jax.Array], actions: jax.Array) -> jax.Array:
    history = apply_hidden_layers(params["history_encoder"], observations["history"])
    state = apply_hidden_layers(params["state_encoder"], observations["state"])
    latent = apply_hidden_layers(params["forward_head"], jnp.concatenate((history, state, actions), axis=1))
    return apply_linear(params["forward_output"], latent)


def forward_motion_loss(params: dict, batch: Batch) -> tuple[jax.Array, jax.Array, jax.Array]:
    predicted = predict_forward_motion(params, batch.observations, batch.actions)
    # Next-state velocities describe the recorded interval's pose deltas, already normalized by the runner.
    target = jnp.concatenate(
        (batch.next_observations["state"][:, STATE_JOINT_VELOCITY_SLICE], batch.next_observations["state"][:, STATE_TCP_VELOCITY_SLICE]),
        axis=1,
    )
    squared_error = jnp.square(predicted - target)
    return jnp.mean(squared_error), jnp.mean(squared_error[:, :6]), jnp.mean(squared_error[:, 6:])


def encode_observation(params: dict, observations: dict[str, jax.Array]) -> jax.Array:
    return jnp.concatenate(
        (
            apply_hidden_layers(params["history_encoder"], observations["history"]),
            apply_hidden_layers(params["state_encoder"], observations["state"]),
            apply_hidden_layers(params["goal_encoder"], observations["goal"]),
        ),
        axis=1,
    )


def actor_distribution(actor_params: dict, observations: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
    latent = apply_hidden_layers(actor_params["fusion"], encode_observation(actor_params, observations))
    mean = apply_linear(actor_params["mean"], latent)
    log_std = jnp.clip(apply_linear(actor_params["log_std"], latent), -20.0, 2.0)
    return mean, log_std


def sample_actor(
    actor_params: dict,
    observations: dict[str, jax.Array],
    random_key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    mean, log_std = actor_distribution(actor_params, observations)
    noise = jax.random.normal(random_key, mean.shape)
    unsquashed_actions = mean + jnp.exp(log_std) * noise
    actions = jnp.tanh(unsquashed_actions)
    gaussian_log_prob = -0.5 * (jnp.square(noise) + 2.0 * log_std + math.log(2.0 * math.pi))
    log_prob = jnp.sum(gaussian_log_prob - jnp.log(1.0 - jnp.square(actions) + 1e-6), axis=1, keepdims=True)
    return actions, log_prob


def init_q_network(
    random_key: jax.Array,
    observation_sizes: dict[str, int],
    action_dim: int,
    history_encoder_dims: tuple[int, ...],
    state_encoder_dims: tuple[int, ...],
    goal_encoder_dims: tuple[int, ...],
    fusion_dims: tuple[int, ...],
    forward_head_dims: tuple[int, ...],
) -> dict:
    history_key, state_key, goal_key, fusion_key, output_key = jax.random.split(random_key, 5)
    forward_key, forward_output_key = jax.random.split(jax.random.fold_in(random_key, 1))
    fusion_input_dim = history_encoder_dims[-1] + state_encoder_dims[-1] + goal_encoder_dims[-1] + action_dim
    return {
        "history_encoder": init_hidden_layers(history_key, observation_sizes["history"], history_encoder_dims),
        "state_encoder": init_hidden_layers(state_key, observation_sizes["state"], state_encoder_dims),
        "goal_encoder": init_hidden_layers(goal_key, observation_sizes["goal"], goal_encoder_dims),
        "fusion": init_hidden_layers(fusion_key, fusion_input_dim, fusion_dims),
        "output": init_linear(output_key, fusion_dims[-1], 1),
        "forward_head": init_hidden_layers(forward_key, history_encoder_dims[-1] + state_encoder_dims[-1] + action_dim, forward_head_dims),
        "forward_output": init_linear(forward_output_key, forward_head_dims[-1], 12),
    }


def apply_q_network(
    params: dict,
    observations: dict[str, jax.Array],
    actions: jax.Array,
) -> jax.Array:
    encoded_observation = encode_observation(params, observations)
    latent = apply_hidden_layers(params["fusion"], jnp.concatenate((encoded_observation, actions), axis=1))
    return apply_linear(params["output"], latent)


def apply_twin_critics(
    params: tuple,
    observations: dict[str, jax.Array],
    actions: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return tuple(apply_q_network(network_params, observations, actions) for network_params in params)


def polyak_update(target_params, source_params, tau: float):
    return jax.tree.map(lambda target, source: (1.0 - tau) * target + tau * source, target_params, source_params)


def build_update_step(
    actor_optimizer,
    critic_optimizer,
    ent_coef_optimizer,
    gamma: float,
    tau: float,
    target_entropy: float,
    actor_forward_loss_weight: float,
    critic_forward_loss_weight: float,
):
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
            forward_metrics = jnp.mean(jnp.stack([jnp.stack(forward_motion_loss(params, batch)) for params in critic_params]), axis=0)
            total_loss = loss + critic_forward_loss_weight * forward_metrics[0]
            return total_loss, (loss, jnp.mean(jnp.minimum(*current_q_values)), forward_metrics)

        (critic_total_loss, critic_metrics), critic_gradients = jax.value_and_grad(critic_loss_fn, has_aux=True)(state.critic_params)
        critic_loss, mean_q, critic_forward_metrics = critic_metrics
        critic_forward_loss, critic_forward_joint_loss, critic_forward_tcp_loss = critic_forward_metrics
        critic_updates, critic_optimizer_state = critic_optimizer.update(critic_gradients, state.critic_optimizer_state, state.critic_params)
        critic_params = optax.apply_updates(state.critic_params, critic_updates)

        def actor_loss_fn(actor_params):
            policy_actions, log_prob = sample_actor(actor_params, batch.observations, actor_key)
            q_values = apply_twin_critics(critic_params, batch.observations, policy_actions)
            actor_loss = jnp.mean(entropy_coefficient * log_prob - jnp.minimum(*q_values))
            forward_loss, forward_joint_loss, forward_tcp_loss = forward_motion_loss(actor_params, batch)
            total_loss = actor_loss + actor_forward_loss_weight * forward_loss
            return total_loss, (log_prob, actor_loss, forward_loss, forward_joint_loss, forward_tcp_loss)

        (actor_total_loss, actor_metrics), actor_gradients = jax.value_and_grad(actor_loss_fn, has_aux=True)(state.actor_params)
        policy_log_prob, actor_loss, forward_loss, forward_joint_loss, forward_tcp_loss = actor_metrics
        actor_updates, actor_optimizer_state = actor_optimizer.update(actor_gradients, state.actor_optimizer_state, state.actor_params)
        actor_params = optax.apply_updates(state.actor_params, actor_updates)

        def ent_coef_loss_fn(log_ent_coef):
            return -jnp.mean(log_ent_coef * jax.lax.stop_gradient(policy_log_prob + target_entropy))

        ent_coef_loss, ent_coef_gradients = jax.value_and_grad(ent_coef_loss_fn)(state.log_ent_coef)
        ent_coef_updates, ent_coef_optimizer_state = ent_coef_optimizer.update(ent_coef_gradients, state.ent_coef_optimizer_state, state.log_ent_coef)
        log_ent_coef = optax.apply_updates(state.log_ent_coef, ent_coef_updates)
        target_critic_params = polyak_update(state.target_critic_params, critic_params, tau)
        finite = jnp.logical_and(
            jnp.all(jnp.isfinite(jnp.array((actor_loss, actor_total_loss, forward_loss, critic_loss, critic_total_loss, critic_forward_loss, ent_coef_loss, entropy_coefficient, mean_q)))),
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
        metrics = UpdateMetrics(
            actor_loss=actor_loss,
            actor_total_loss=actor_total_loss,
            forward_loss=forward_loss,
            forward_joint_loss=forward_joint_loss,
            forward_tcp_loss=forward_tcp_loss,
            critic_loss=critic_loss,
            critic_total_loss=critic_total_loss,
            critic_forward_loss=critic_forward_loss,
            critic_forward_joint_loss=critic_forward_joint_loss,
            critic_forward_tcp_loss=critic_forward_tcp_loss,
            ent_coef_loss=ent_coef_loss,
            ent_coef=entropy_coefficient,
            mean_q=mean_q,
            finite=finite,
        )
        return new_state, metrics

    return jax.jit(update_step, donate_argnums=(0,))


def initialize_state(
    random_key: jax.Array,
    observation_sizes: dict[str, int],
    action_dim: int,
    architecture: dict[str, tuple[int, ...]],
    learning_rate: float,
):
    actor_key, first_critic_key, second_critic_key, state_key = jax.random.split(random_key, 4)
    actor_params = init_actor(
        actor_key,
        observation_sizes,
        action_dim,
        architecture["history_encoder"],
        architecture["state_encoder"],
        architecture["goal_encoder"],
        architecture["fusion"],
        architecture["forward_head"],
    )
    critic_params = (
        init_q_network(
            first_critic_key,
            observation_sizes,
            action_dim,
            architecture["history_encoder"],
            architecture["state_encoder"],
            architecture["goal_encoder"],
            architecture["fusion"],
            architecture["forward_head"],
        ),
        init_q_network(
            second_critic_key,
            observation_sizes,
            action_dim,
            architecture["history_encoder"],
            architecture["state_encoder"],
            architecture["goal_encoder"],
            architecture["fusion"],
            architecture["forward_head"],
        ),
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
        configured_history_steps = cfg.control.frequencies.joint * cfg.control.policy_history_seconds
        assert configured_history_steps >= 2, "policy_history_seconds must span at least two joint control intervals"
        assert float(configured_history_steps).is_integer(), "policy_history_seconds must contain an integer number of joint control intervals"
        history_steps = int(configured_history_steps)
        self.observation_sizes = policy_observation_sizes(int(cfg.waypoint.cartesian_action_dim), history_steps)
        self.action_dim = 6
        self.architecture = {
            "history_encoder": tuple(int(hidden_dim) for hidden_dim in cfg.policy.history_encoder),
            "state_encoder": tuple(int(hidden_dim) for hidden_dim in cfg.policy.state_encoder),
            "goal_encoder": tuple(int(hidden_dim) for hidden_dim in cfg.policy.goal_encoder),
            "fusion": tuple(int(hidden_dim) for hidden_dim in cfg.policy.fusion),
            "forward_head": tuple(int(hidden_dim) for hidden_dim in cfg.policy.forward_head),
        }
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
        self.replay_buffer = MixedReplayBuffer(
            int(cfg.training.buffer_size),
            int(cfg.training.real_data.buffer_size),
            self.observation_sizes,
            self.action_dim,
            self.random_seed,
            float(cfg.training.real_data.fraction),
            self.device,
        )
        with jax.default_device(self.device):
            state, actor_optimizer, critic_optimizer, ent_coef_optimizer = initialize_state(
                jax.random.PRNGKey(self.random_seed),
                self.observation_sizes,
                self.action_dim,
                self.architecture,
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
            float(cfg.training.actor_forward_loss_weight),
            float(cfg.training.critic_forward_loss_weight),
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
            "actor_total_loss": float(metrics.actor_total_loss),
            "forward_loss": float(metrics.forward_loss),
            "forward_joint_loss": float(metrics.forward_joint_loss),
            "forward_tcp_loss": float(metrics.forward_tcp_loss),
            "critic_loss": float(metrics.critic_loss),
            "critic_total_loss": float(metrics.critic_total_loss),
            "critic_forward_loss": float(metrics.critic_forward_loss),
            "critic_forward_joint_loss": float(metrics.critic_forward_joint_loss),
            "critic_forward_tcp_loss": float(metrics.critic_forward_tcp_loss),
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
            "observation_sizes": self.observation_sizes,
            "action_dim": self.action_dim,
            "architecture": self.architecture,
            "random_seed": self.random_seed,
        }
        checkpoint_path.write_bytes(pickle.dumps(checkpoint))
        actor_params = self.actor_params_numpy()
        actor_arrays = {}
        for branch in ("history_encoder", "state_encoder", "goal_encoder", "fusion"):
            for index, layer in enumerate(actor_params[branch]):
                for name, value in layer.items():
                    actor_arrays[f"{branch}_{index}_{name}"] = value
        for head in ("mean", "log_std"):
            for name, value in actor_params[head].items():
                actor_arrays[f"{head}_{name}"] = value
        for name in POLICY_OBSERVATION_NAMES:
            actor_arrays[f"observation_size_{name}"] = np.array(self.observation_sizes[name], dtype=np.int64)
        actor_arrays["random_seed"] = np.array(self.random_seed, dtype=np.int64)
        np.savez(checkpoint_path.with_suffix(".actor.npz"), **actor_arrays)
        return checkpoint_path

    def load(self, path: str) -> None:
        checkpoint = pickle.loads(Path(path).read_bytes())
        if (
            checkpoint["observation_sizes"] != self.observation_sizes
            or checkpoint["action_dim"] != self.action_dim
            or checkpoint["architecture"] != self.architecture
        ):
            raise ValueError("Checkpoint architecture does not match the configured JAX SAC architecture")
        saved_leaves, saved_structure = jax.tree.flatten(checkpoint["state"])
        current_leaves, current_structure = jax.tree.flatten(self.state)
        if saved_structure != current_structure or [value.shape for value in saved_leaves] != [value.shape for value in current_leaves]:
            raise ValueError("Checkpoint parameter and optimizer structure does not match the configured JAX SAC architecture")
        self.state = jax.tree.map(lambda value: jax.device_put(value, self.device), checkpoint["state"])
