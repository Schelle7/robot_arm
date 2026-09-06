import jax
import jax.numpy as jnp
import numpy as np

from robot_arm.jax_sac import actor_distribution as jax_actor_distribution
from robot_arm.jax_sac import init_actor
from robot_arm.numpy_policy import actor_distribution as numpy_actor_distribution
from robot_arm.robot_schema import policy_observation_dim


def test_numpy_actor_distribution_matches_jax():
    observation_dim = policy_observation_dim(7)
    action_dim = 6
    hidden_dims = (32, 16)
    actor_params = init_actor(jax.random.PRNGKey(0), observation_dim, action_dim, hidden_dims)
    numpy_actor_params = jax.tree.map(np.asarray, actor_params)
    observations = np.random.default_rng(0).standard_normal((8, observation_dim)).astype(np.float32)

    with jax.default_matmul_precision("highest"):
        jax_mean, jax_log_std = jax_actor_distribution(actor_params, jnp.asarray(observations))
    numpy_mean, numpy_log_std = numpy_actor_distribution(numpy_actor_params, observations)

    np.testing.assert_allclose(numpy_mean, np.asarray(jax_mean), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(numpy_log_std, np.asarray(jax_log_std), rtol=1e-5, atol=1e-6)