from typing import NamedTuple

import jax
import numpy as np

from robot_arm.robot_schema import POLICY_OBSERVATION_NAMES


class Batch(NamedTuple):
    observations: dict[str, jax.Array]
    actions: jax.Array
    next_observations: dict[str, jax.Array]
    rewards: jax.Array
    dones: jax.Array
    is_real: jax.Array


class NumpyReplayBuffer:
    def __init__(self, capacity: int, observation_sizes: dict[str, int], action_dim: int, random_seed: int):
        assert capacity > 0
        self.capacity = capacity
        self.observations = {name: np.empty((capacity, observation_sizes[name]), dtype=np.float32) for name in POLICY_OBSERVATION_NAMES}
        self.next_observations = {name: np.empty((capacity, observation_sizes[name]), dtype=np.float32) for name in POLICY_OBSERVATION_NAMES}
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0
        self.random = np.random.default_rng(random_seed)

    def add(self, observation, next_observation, action, reward, done) -> None:
        for name in POLICY_OBSERVATION_NAMES:
            self.observations[name][self.position] = observation[name]
            self.next_observations[name][self.position] = next_observation[name]
        self.actions[self.position] = action
        self.rewards[self.position, 0] = reward
        self.dones[self.position, 0] = done
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        assert batch_size > 0
        assert self.size > 0, "Cannot sample an empty replay buffer"
        indices = self.random.integers(self.size, size=batch_size)
        return {
            "observations": {name: self.observations[name][indices] for name in POLICY_OBSERVATION_NAMES},
            "actions": self.actions[indices],
            "next_observations": {name: self.next_observations[name][indices] for name in POLICY_OBSERVATION_NAMES},
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
        }


class MixedReplayBuffer:
    def __init__(self, sim_capacity: int, real_capacity: int, observation_sizes: dict[str, int], action_dim: int, random_seed: int, real_fraction: float, device):
        assert 0.0 <= real_fraction <= 1.0
        self.sim = NumpyReplayBuffer(sim_capacity, observation_sizes, action_dim, random_seed)
        self.real = NumpyReplayBuffer(real_capacity, observation_sizes, action_dim, random_seed + 1)
        self.real_fraction = real_fraction
        self.device = device

    def sample(self, batch_size: int) -> Batch:
        assert batch_size > 0
        real_count = int(round(batch_size * self.real_fraction))
        sim_count = batch_size - real_count
        assert self.real_fraction == 0.0 or real_count > 0, "Batch size rounds the requested real fraction to zero"
        assert self.real_fraction == 1.0 or sim_count > 0, "Batch size rounds the requested simulation fraction to zero"
        samples = []
        for buffer, count, is_real in ((self.sim, sim_count, False), (self.real, real_count, True)):
            if count > 0:
                sample = buffer.sample(count)
                sample["is_real"] = np.full((count,), is_real, dtype=bool)
                samples.append(sample)
        combined = jax.tree.map(lambda *arrays: np.concatenate(arrays, axis=0), *samples)
        return Batch(**jax.device_put(combined, self.device))
