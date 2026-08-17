from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("cartesian_smolvla")
@dataclass
class CartesianSmolVLAConfig(SmolVLAConfig):
    pass