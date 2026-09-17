import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robot_arm.training.jax_sac import actor_distribution as jax_actor_distribution
from robot_arm.training.jax_sac import JaxSAC, apply_q_network, build_update_step, forward_motion_loss, init_actor, initialize_state, predict_forward_motion
from robot_arm.training.replay_buffer import Batch
from robot_arm.policies.numpy_policy import actor_distribution as numpy_actor_distribution
from robot_arm.policies.numpy_policy import load_numpy_policy
from robot_arm.robot_schema import POLICY_OBSERVATION_NAMES, policy_observation_sizes


def test_numpy_actor_distribution_matches_jax():
    history_steps = 10
    observation_sizes = policy_observation_sizes(7, history_steps)
    action_dim = 6
    actor_params = init_actor(
        jax.random.PRNGKey(0),
        observation_sizes,
        action_dim,
        (32, 16),
        (16, 8),
        (16, 8),
        (32, 16),
        (16,),
    )
    numpy_actor_params = jax.tree.map(np.asarray, actor_params)
    random = np.random.default_rng(0)
    observations = {
        name: random.standard_normal((8, observation_sizes[name])).astype(np.float32)
        for name in POLICY_OBSERVATION_NAMES
    }

    with jax.default_matmul_precision("highest"):
        jax_mean, jax_log_std = jax_actor_distribution(actor_params, jax.tree.map(jnp.asarray, observations))
    numpy_mean, numpy_log_std = numpy_actor_distribution(numpy_actor_params, observations)

    np.testing.assert_allclose(numpy_mean, np.asarray(jax_mean), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(numpy_log_std, np.asarray(jax_log_std), rtol=1e-5, atol=1e-6)


@pytest.fixture
def forward_case():
    sizes = policy_observation_sizes(7, 10)
    architecture = {
        "history_encoder": (16, 8),
        "state_encoder": (8,),
        "goal_encoder": (8,),
        "fusion": (16,),
        "forward_head": (16,),
    }
    state, actor_optimizer, critic_optimizer, entropy_optimizer = initialize_state(
        jax.random.PRNGKey(42), sizes, 6, architecture, 0.0003,
    )
    random = np.random.default_rng(42)
    observations = {name: jnp.asarray(random.standard_normal((8, size)), dtype=jnp.float32) for name, size in sizes.items()}
    next_observations = {name: jnp.full((8, size), 99.0) for name, size in sizes.items()}
    next_observations["state"] = next_observations["state"].at[:, 6:12].set(2.0).at[:, 15:21].set(3.0)
    batch = Batch(observations, jnp.asarray(random.uniform(-1, 1, (8, 6)), dtype=jnp.float32), next_observations, jnp.zeros((8, 1)), jnp.ones((8, 1)), jnp.zeros(8, dtype=bool))
    return state, (actor_optimizer, critic_optimizer, entropy_optimizer), batch, sizes, architecture


def test_forward_targets_use_only_interval_joint_and_tcp_velocities(forward_case):
    state, _, batch, _, _ = forward_case
    params = dict(state.actor_params)
    params["forward_output"] = jax.tree.map(jnp.zeros_like, params["forward_output"])
    assert predict_forward_motion(params, batch.observations, batch.actions).shape == (8, 12)
    loss, joint_loss, tcp_loss = forward_motion_loss(params, batch)
    np.testing.assert_allclose([loss, joint_loss, tcp_loss], [6.5, 4.0, 9.0])


@pytest.mark.parametrize("network_index", [-1, 0, 1])
def test_forward_gradient_reaches_history_state_and_head_only(forward_case, network_index):
    state, _, batch, _, _ = forward_case
    params = state.actor_params if network_index == -1 else state.critic_params[network_index]
    gradients = jax.grad(lambda network: forward_motion_loss(network, batch)[0])(params)
    for branch in ("history_encoder", "state_encoder", "forward_head", "forward_output"):
        assert any(np.any(np.asarray(value) != 0) for value in jax.tree.leaves(gradients[branch]))
    output_branches = ("mean", "log_std") if network_index == -1 else ("output",)
    for branch in ("goal_encoder", "fusion", *output_branches):
        for value in jax.tree.leaves(gradients[branch]):
            np.testing.assert_array_equal(value, jnp.zeros_like(value))
    changed_goal = {**batch.observations, "goal": batch.observations["goal"] * 100}
    changed_state = {**batch.observations, "state": batch.observations["state"] * 100}
    expected = predict_forward_motion(params, batch.observations, batch.actions)
    np.testing.assert_array_equal(expected, predict_forward_motion(params, changed_goal, batch.actions))
    assert not np.allclose(expected, predict_forward_motion(params, changed_state, batch.actions))
    assert not np.allclose(expected, predict_forward_motion(params, batch.observations, -batch.actions))


@pytest.mark.parametrize("actor_weight,critic_weight", [(0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (0.1, 0.1)])
def test_update_reports_weighted_losses_and_updates_independent_heads(forward_case, actor_weight, critic_weight):
    state, optimizers, batch, _, _ = forward_case
    before = jax.tree.map(
        lambda value: np.array(value, copy=True),
        tuple(params["forward_output"] for params in (state.actor_params, *state.critic_params)),
    )
    update = build_update_step(*optimizers, 1.0, 0.005, -6.0, actor_weight, critic_weight)
    updated, metrics = update(state, batch)
    assert bool(metrics.finite)
    np.testing.assert_allclose(metrics.actor_total_loss, metrics.actor_loss + actor_weight * metrics.forward_loss, rtol=1e-6)
    np.testing.assert_allclose(metrics.critic_total_loss, metrics.critic_loss + critic_weight * metrics.critic_forward_loss, rtol=1e-6)
    np.testing.assert_allclose(metrics.forward_loss, (metrics.forward_joint_loss + metrics.forward_tcp_loss) / 2, rtol=1e-6)
    np.testing.assert_allclose(metrics.critic_forward_loss, (metrics.critic_forward_joint_loss + metrics.critic_forward_tcp_loss) / 2, rtol=1e-6)
    for original, params, weight in zip(before, (updated.actor_params, *updated.critic_params), (actor_weight, critic_weight, critic_weight)):
        if weight == 0.0:
            for name in original:
                np.testing.assert_array_equal(original[name], params["forward_output"][name])
        else:
            assert any(not np.array_equal(original[name], params["forward_output"][name]) for name in original)


def test_critic_value_does_not_consume_forward_predictions(forward_case):
    state, _, batch, _, _ = forward_case
    for params in state.critic_params:
        changed = {**params, "forward_output": jax.tree.map(lambda value: value + 100, params["forward_output"])}
        np.testing.assert_array_equal(
            apply_q_network(params, batch.observations, batch.actions),
            apply_q_network(changed, batch.observations, batch.actions),
        )
        assert not np.allclose(
            predict_forward_motion(params, batch.observations, batch.actions),
            predict_forward_motion(changed, batch.observations, batch.actions),
        )


def test_forward_checkpoint_roundtrip_and_inference_export(forward_case, tmp_path):
    state, _, batch, sizes, architecture = forward_case
    model = JaxSAC.__new__(JaxSAC)
    model.state = state
    model.observation_sizes = sizes
    model.action_dim = 6
    model.architecture = architecture
    model.random_seed = 42
    model.device = jax.devices()[0]
    checkpoint_path = model.save(str(tmp_path / "policy"))
    policy = load_numpy_policy(str(checkpoint_path.with_suffix(".actor.npz")))
    assert "forward_head" not in policy.actor_params
    with jax.default_matmul_precision("highest"):
        expected = jax_actor_distribution(state.actor_params, batch.observations)
    actual = numpy_actor_distribution(policy.actor_params, jax.tree.map(np.asarray, batch.observations))
    for expected_value, actual_value in zip(expected, actual):
        np.testing.assert_allclose(actual_value, expected_value, rtol=1e-4, atol=1e-5)
    model.load(str(checkpoint_path))
    for original, restored in zip(jax.tree.leaves(state), jax.tree.leaves(model.state)):
        np.testing.assert_array_equal(original, restored)
    checkpoint = pickle.loads(checkpoint_path.read_bytes())
    original_bytes = checkpoint_path.read_bytes()
    checkpoint["state"].actor_params["forward_head"][0]["weight"] = checkpoint["state"].actor_params["forward_head"][0]["weight"][:-8]
    checkpoint_path.write_bytes(pickle.dumps(checkpoint))
    with pytest.raises(ValueError, match="Checkpoint parameter"):
        model.load(str(checkpoint_path))
    checkpoint = pickle.loads(original_bytes)
    del checkpoint["state"].critic_params[0]["forward_head"]
    checkpoint_path.write_bytes(pickle.dumps(checkpoint))
    with pytest.raises(ValueError, match="Checkpoint parameter"):
        model.load(str(checkpoint_path))
    checkpoint = pickle.loads(original_bytes)
    del checkpoint["architecture"]["forward_head"]
    checkpoint_path.write_bytes(pickle.dumps(checkpoint))
    with pytest.raises(ValueError, match="Checkpoint architecture"):
        model.load(str(checkpoint_path))
