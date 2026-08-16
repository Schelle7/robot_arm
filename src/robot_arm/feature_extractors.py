import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class MLPFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, hidden_dims: list[int]):
        self.observation_keys = tuple(observation_space.spaces)
        input_dim = sum(int(np.prod(space.shape)) for space in observation_space.spaces.values())
        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(previous_dim, hidden_dim), nn.ReLU()])
            previous_dim = hidden_dim

        super().__init__(observation_space, features_dim=previous_dim)
        self.network = nn.Sequential(*layers)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        flattened_observations = [observations[key].flatten(start_dim=1) for key in self.observation_keys]
        return self.network(torch.cat(flattened_observations, dim=1))
