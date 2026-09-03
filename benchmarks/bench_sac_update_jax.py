import math
import time
from typing import NamedTuple

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import DictConfig


class TrainState(NamedTuple):
    actor_params: dict
    critic_params: tuple
    target_critic_params: tuple
    log_ent_coef: jax.Array
    actor_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    ent_coef_optimizer_state: optax.OptState
    random_key: jax.Array


class Batch(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    next_observations: jax.Array
    rewards: jax.Array
    dones: jax.Array


def init_linear(random_key, input_dim, output_dim):
    bound = 1.0 / math.sqrt(input_dim)
    return {
        "weight": jax.random.uniform(random_key, (input_dim, output_dim), minval=-bound, maxval=bound),
        "bias": jnp.zeros((output_dim,), dtype=jnp.float32),
    }


def apply_linear(params, inputs):
    return inputs @ params["weight"] + params["bias"]


def init_hidden_layers(random_key, input_dim, hidden_dims):
    layer_keys = jax.random.split(random_key, len(hidden_dims))
    layer_dims = (input_dim, *hidden_dims)
    return tuple(init_linear(layer_key, layer_dims[index], layer_dims[index + 1]) for index, layer_key in enumerate(layer_keys))


def apply_hidden_layers(params, inputs):
    activations = inputs
    for layer_params in params:
        activations = jax.nn.relu(apply_linear(layer_params, activations))
    return activations


def init_actor(random_key, observation_dim, action_dim, hidden_dims):
    trunk_key, mean_key, log_std_key = jax.random.split(random_key, 3)
    trunk = init_hidden_layers(trunk_key, observation_dim, hidden_dims)
    latent_dim = hidden_dims[-1]
    return {
        "trunk": trunk,
        "mean": init_linear(mean_key, latent_dim, action_dim),
        "log_std": init_linear(log_std_key, latent_dim, action_dim),
    }


def sample_actor(actor_params, observations, random_key):
    latent = apply_hidden_layers(actor_params["trunk"], observations)
    mean = apply_linear(actor_params["mean"], latent)
    log_std = jnp.clip(apply_linear(actor_params["log_std"], latent), -20.0, 2.0)
    noise = jax.random.normal(random_key, mean.shape)
    unsquashed_actions = mean + jnp.exp(log_std) * noise
    actions = jnp.tanh(unsquashed_actions)
    gaussian_log_prob = -0.5 * (jnp.square(noise) + 2.0 * log_std + math.log(2.0 * math.pi))
    log_prob = jnp.sum(gaussian_log_prob - jnp.log(1.0 - jnp.square(actions) + 1e-6), axis=1, keepdims=True)
    return actions, log_prob


def init_q_network(random_key, observation_dim, action_dim, hidden_dims):
    hidden_key, output_key = jax.random.split(random_key)
    hidden = init_hidden_layers(hidden_key, observation_dim + action_dim, hidden_dims)
    return {
        "hidden": hidden,
        "output": init_linear(output_key, hidden_dims[-1], 1),
    }


def apply_q_network(q_params, observations, actions):
    inputs = jnp.concatenate((observations, actions), axis=1)
    latent = apply_hidden_layers(q_params["hidden"], inputs)
    return apply_linear(q_params["output"], latent)


def apply_twin_critics(critic_params, observations, actions):
    return tuple(apply_q_network(q_params, observations, actions) for q_params in critic_params)


def make_batch(random_key, batch_size, observation_dim, action_dim):
    observation_key, action_key, next_observation_key, reward_key, done_key = jax.random.split(random_key, 5)
    return Batch(
        observations=jax.random.normal(observation_key, (batch_size, observation_dim)),
        actions=jax.random.uniform(action_key, (batch_size, action_dim), minval=-1.0, maxval=1.0),
        next_observations=jax.random.normal(next_observation_key, (batch_size, observation_dim)),
        rewards=jax.random.normal(reward_key, (batch_size, 1)),
        dones=jax.random.bernoulli(done_key, 0.05, (batch_size, 1)).astype(jnp.float32),
    )


def polyak_update(target_params, source_params, tau):
    return jax.tree.map(lambda target, source: (1.0 - tau) * target + tau * source, target_params, source_params)


def build_update_step(batch, actor_optimizer, critic_optimizer, ent_coef_optimizer, gamma, tau, target_entropy):
    def update_step(state):
        random_key, actor_key, next_actor_key = jax.random.split(state.random_key, 3)
        entropy_coefficient = jnp.exp(state.log_ent_coef)
        _, policy_log_prob = sample_actor(state.actor_params, batch.observations, actor_key)

        def entropy_coefficient_loss(log_ent_coef):
            return -jnp.mean(log_ent_coef * jax.lax.stop_gradient(policy_log_prob + target_entropy))

        _, entropy_coefficient_gradients = jax.value_and_grad(entropy_coefficient_loss)(state.log_ent_coef)
        ent_coef_updates, ent_coef_optimizer_state = ent_coef_optimizer.update(
            entropy_coefficient_gradients,
            state.ent_coef_optimizer_state,
            state.log_ent_coef,
        )
        log_ent_coef = optax.apply_updates(state.log_ent_coef, ent_coef_updates)

        next_actions, next_log_prob = sample_actor(state.actor_params, batch.next_observations, next_actor_key)
        next_q_values = apply_twin_critics(state.target_critic_params, batch.next_observations, next_actions)
        next_q_value = jnp.minimum(*next_q_values) - entropy_coefficient * next_log_prob
        target_q_value = batch.rewards + (1.0 - batch.dones) * gamma * next_q_value
        target_q_value = jax.lax.stop_gradient(target_q_value)

        def critic_loss(critic_params):
            current_q_values = apply_twin_critics(critic_params, batch.observations, batch.actions)
            return 0.5 * sum(jnp.mean(jnp.square(current_q_value - target_q_value)) for current_q_value in current_q_values)

        _, critic_gradients = jax.value_and_grad(critic_loss)(state.critic_params)
        critic_updates, critic_optimizer_state = critic_optimizer.update(
            critic_gradients,
            state.critic_optimizer_state,
            state.critic_params,
        )
        critic_params = optax.apply_updates(state.critic_params, critic_updates)

        def actor_loss(actor_params):
            policy_actions, log_prob = sample_actor(actor_params, batch.observations, actor_key)
            q_values = apply_twin_critics(critic_params, batch.observations, policy_actions)
            return jnp.mean(entropy_coefficient * log_prob - jnp.minimum(*q_values))

        _, actor_gradients = jax.value_and_grad(actor_loss)(state.actor_params)
        actor_updates, actor_optimizer_state = actor_optimizer.update(
            actor_gradients,
            state.actor_optimizer_state,
            state.actor_params,
        )
        actor_params = optax.apply_updates(state.actor_params, actor_updates)
        target_critic_params = polyak_update(state.target_critic_params, critic_params, tau)

        return TrainState(
            actor_params=actor_params,
            critic_params=critic_params,
            target_critic_params=target_critic_params,
            log_ent_coef=log_ent_coef,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
            ent_coef_optimizer_state=ent_coef_optimizer_state,
            random_key=random_key,
        )

    return update_step


def initialize_state(random_key, observation_dim, action_dim, hidden_dims, learning_rate):
    actor_key, first_critic_key, second_critic_key, state_key = jax.random.split(random_key, 4)
    actor_params = init_actor(actor_key, observation_dim, action_dim, hidden_dims)
    critic_params = (
        init_q_network(first_critic_key, observation_dim, action_dim, hidden_dims),
        init_q_network(second_critic_key, observation_dim, action_dim, hidden_dims),
    )
    actor_optimizer = optax.adam(learning_rate, eps=1e-5)
    critic_optimizer = optax.adam(learning_rate, eps=1e-5)
    ent_coef_optimizer = optax.adam(learning_rate, eps=1e-5)
    log_ent_coef = jnp.zeros((), dtype=jnp.float32)
    state = TrainState(
        actor_params=actor_params,
        critic_params=critic_params,
        target_critic_params=critic_params,
        log_ent_coef=log_ent_coef,
        actor_optimizer_state=actor_optimizer.init(actor_params),
        critic_optimizer_state=critic_optimizer.init(critic_params),
        ent_coef_optimizer_state=ent_coef_optimizer.init(log_ent_coef),
        random_key=state_key,
    )
    return state, actor_optimizer, critic_optimizer, ent_coef_optimizer


def measure_updates(operation, state, call_count, updates_per_call):
    state = operation(state)
    state.log_ent_coef.block_until_ready()
    started = time.perf_counter()
    for _ in range(call_count):
        state = operation(state)
    state.log_ent_coef.block_until_ready()
    elapsed_seconds = time.perf_counter() - started
    update_count = call_count * updates_per_call
    return state, elapsed_seconds / update_count


def parameter_count(params):
    return sum(int(np.prod(array.shape)) for array in jax.tree.leaves(params))


@hydra.main(version_base=None, config_path="../conf", config_name="benchmark_sac_jax")
def benchmark(cfg: DictConfig):
    observation_dim = 29 + int(cfg.waypoint.cartesian_action_dim)
    action_dim = 6
    hidden_dims = tuple(int(hidden_dim) for hidden_dim in cfg.policy.net_arch)
    batch_size = int(cfg.training.batch_size)
    warmup_iterations = int(cfg.benchmark.warmup_iterations)
    measured_iterations = int(cfg.benchmark.measured_iterations)
    updates_per_compiled_block = int(cfg.benchmark.updates_per_compiled_block)
    learning_rate = float(cfg.training.learning_rate)

    random_key = jax.random.PRNGKey(0)
    state, actor_optimizer, critic_optimizer, ent_coef_optimizer = initialize_state(
        random_key,
        observation_dim,
        action_dim,
        hidden_dims,
        learning_rate,
    )
    batch = make_batch(random_key, batch_size, observation_dim, action_dim)
    update_step = build_update_step(
        batch,
        actor_optimizer,
        critic_optimizer,
        ent_coef_optimizer,
        float(cfg.training.gamma),
        float(cfg.training.tau),
        -float(action_dim),
    )
    compiled_update = jax.jit(update_step)

    def update_block(current_state):
        return jax.lax.fori_loop(0, updates_per_compiled_block, lambda _, loop_state: update_step(loop_state), current_state)

    compiled_update_block = jax.jit(update_block)

    for _ in range(warmup_iterations):
        state = compiled_update(state)
    state.log_ent_coef.block_until_ready()
    state, single_update_seconds = measure_updates(compiled_update, state, measured_iterations, 1)

    warmup_blocks = math.ceil(warmup_iterations / updates_per_compiled_block)
    for _ in range(warmup_blocks):
        state = compiled_update_block(state)
    state.log_ent_coef.block_until_ready()
    measured_blocks = math.ceil(measured_iterations / updates_per_compiled_block)
    state, blocked_update_seconds = measure_updates(compiled_update_block, state, measured_blocks, updates_per_compiled_block)

    print(f"JAX backend: {jax.default_backend()}")
    print(f"Device: {jax.devices()[0]}")
    print(f"Batch size: {batch_size}")
    print(f"Hidden dimensions: {list(hidden_dims)}")
    print(f"Actor parameters: {parameter_count(state.actor_params)}")
    print(f"Twin critic parameters: {parameter_count(state.critic_params)}")
    print(f"Warm-up iterations: {warmup_iterations}")
    print(f"Measured iterations: {measured_iterations}")
    print(f'{"One JIT call per update":<34} {single_update_seconds * 1_000:8.3f} ms  {1.0 / single_update_seconds:8.1f} updates/s')
    print(
        f'{f"{updates_per_compiled_block} updates per JIT call":<34} '
        f"{blocked_update_seconds * 1_000:8.3f} ms  {1.0 / blocked_update_seconds:8.1f} updates/s"
    )


if __name__ == "__main__":
    benchmark()