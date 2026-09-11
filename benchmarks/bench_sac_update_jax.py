import math
import time

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from robot_arm.training.jax_sac import build_update_step, initialize_state
from robot_arm.training.replay_buffer import Batch
from robot_arm.robot_schema import POLICY_OBSERVATION_NAMES, policy_observation_sizes


def make_batch(random_key, batch_size, observation_sizes, action_dim):
    group_count = len(POLICY_OBSERVATION_NAMES)
    random_keys = jax.random.split(random_key, group_count * 2 + 3)
    return Batch(
        observations={
            name: jax.random.normal(random_keys[index], (batch_size, observation_sizes[name]))
            for index, name in enumerate(POLICY_OBSERVATION_NAMES)
        },
        actions=jax.random.uniform(random_keys[group_count * 2], (batch_size, action_dim), minval=-1.0, maxval=1.0),
        next_observations={
            name: jax.random.normal(random_keys[index + group_count], (batch_size, observation_sizes[name]))
            for index, name in enumerate(POLICY_OBSERVATION_NAMES)
        },
        rewards=jax.random.normal(random_keys[group_count * 2 + 1], (batch_size, 1)),
        dones=jax.random.bernoulli(random_keys[group_count * 2 + 2], 0.05, (batch_size, 1)).astype(jnp.float32),
        is_real=jnp.zeros(batch_size, dtype=bool),
    )


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
    configured_history_steps = cfg.control.frequencies.joint * cfg.control.policy_history_seconds
    assert configured_history_steps >= 2, "policy_history_seconds must span at least two low-level control intervals"
    assert float(configured_history_steps).is_integer(), "policy_history_seconds must contain an integer number of low-level control intervals"
    history_steps = int(configured_history_steps)
    observation_sizes = policy_observation_sizes(int(cfg.waypoint.cartesian_action_dim), history_steps)
    architecture = {
        "history_encoder": tuple(int(hidden_dim) for hidden_dim in cfg.policy.history_encoder),
        "state_encoder": tuple(int(hidden_dim) for hidden_dim in cfg.policy.state_encoder),
        "goal_encoder": tuple(int(hidden_dim) for hidden_dim in cfg.policy.goal_encoder),
        "fusion": tuple(int(hidden_dim) for hidden_dim in cfg.policy.fusion),
        "forward_head": tuple(int(hidden_dim) for hidden_dim in cfg.policy.forward_head),
    }
    action_dim = 6
    batch_size = int(cfg.training.batch_size)
    warmup_iterations = int(cfg.benchmark.warmup_iterations)
    measured_iterations = int(cfg.benchmark.measured_iterations)
    updates_per_compiled_block = int(cfg.benchmark.updates_per_compiled_block)

    random_key = jax.random.PRNGKey(0)
    state, actor_optimizer, critic_optimizer, ent_coef_optimizer = initialize_state(
        random_key,
        observation_sizes,
        action_dim,
        architecture,
        float(cfg.training.learning_rate),
    )
    batch = make_batch(random_key, batch_size, observation_sizes, action_dim)
    update_step = build_update_step(
        actor_optimizer,
        critic_optimizer,
        ent_coef_optimizer,
        float(cfg.training.gamma),
        float(cfg.training.tau),
        -float(action_dim),
        float(cfg.training.actor_forward_loss_weight),
        float(cfg.training.critic_forward_loss_weight),
    )

    def single_update(current_state):
        return update_step(current_state, batch)[0]

    def update_block(current_state):
        return jax.lax.fori_loop(
            0,
            updates_per_compiled_block,
            lambda _, loop_state: update_step(loop_state, batch)[0],
            current_state,
        )

    compiled_update_block = jax.jit(update_block, donate_argnums=(0,))

    for _ in range(warmup_iterations):
        state = single_update(state)
    state.log_ent_coef.block_until_ready()
    state, single_update_seconds = measure_updates(single_update, state, measured_iterations, 1)

    warmup_blocks = math.ceil(warmup_iterations / updates_per_compiled_block)
    for _ in range(warmup_blocks):
        state = compiled_update_block(state)
    state.log_ent_coef.block_until_ready()
    measured_blocks = math.ceil(measured_iterations / updates_per_compiled_block)
    state, blocked_update_seconds = measure_updates(compiled_update_block, state, measured_blocks, updates_per_compiled_block)

    print(f"JAX backend: {jax.default_backend()}")
    print(f"Device: {jax.devices()[0]}")
    print(f"Batch size: {batch_size}")
    print(f"Observation groups: {observation_sizes}")
    print(f"Architecture: {architecture}")
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
