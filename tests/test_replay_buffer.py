import jax
import numpy as np
import pytest

from robot_arm.training.replay_buffer import MixedReplayBuffer, NumpyReplayBuffer


SIZES = {"history": 3, "state": 2, "goal": 1}


def add_transition(buffer, marker):
    observation = {name: np.full(size, marker, dtype=np.float32) for name, size in SIZES.items()}
    next_observation = {name: values + 1 for name, values in observation.items()}
    buffer.add(observation, next_observation, np.full(6, marker), marker, False)


@pytest.mark.parametrize("fraction,real_count", [(0.0, 0), (0.25, 2), (1.0, 8)])
def test_mixed_sampling_preserves_source_and_transition_alignment(fraction, real_count):
    buffer = MixedReplayBuffer(4, 4, SIZES, 6, 42, fraction, jax.devices("cpu")[0])
    if fraction < 1:
        add_transition(buffer.sim, 1)
    if fraction > 0:
        add_transition(buffer.real, 2)
    batch = buffer.sample(8)
    flags = np.asarray(batch.is_real)
    assert np.count_nonzero(flags) == real_count
    expected = np.where(flags, 2, 1)
    np.testing.assert_array_equal(np.asarray(batch.rewards)[:, 0], expected)
    np.testing.assert_array_equal(np.asarray(batch.actions)[:, 0], expected)
    np.testing.assert_array_equal(np.asarray(batch.observations["state"])[:, 0], expected)
    np.testing.assert_array_equal(np.asarray(batch.next_observations["state"])[:, 0], expected + 1)


def test_simulation_overwrites_do_not_remove_real_data():
    buffer = MixedReplayBuffer(2, 2, SIZES, 6, 42, 0.5, jax.devices("cpu")[0])
    add_transition(buffer.real, 7)
    for marker in range(5):
        add_transition(buffer.sim, marker)
    assert buffer.sim.size == 2
    assert buffer.real.size == 1
    assert set(buffer.sim.rewards[:, 0]) == {3, 4}
    np.testing.assert_array_equal(buffer.real.sample(3)["rewards"], np.full((3, 1), 7))


def test_requested_real_samples_fail_when_real_buffer_is_empty():
    buffer = MixedReplayBuffer(2, 2, SIZES, 6, 42, 0.5, jax.devices("cpu")[0])
    add_transition(buffer.sim, 1)
    with pytest.raises(AssertionError, match="empty replay buffer"):
        buffer.sample(8)


def test_empty_buffer_cannot_sample():
    buffer = NumpyReplayBuffer(2, SIZES, 6, 42)
    with pytest.raises(AssertionError, match="empty replay buffer"):
        buffer.sample(1)
