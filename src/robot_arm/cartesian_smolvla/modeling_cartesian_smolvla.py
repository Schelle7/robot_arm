from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from robot_arm.cartesian_smolvla.configuration_cartesian_smolvla import CartesianSmolVLAConfig
from robot_arm.cartesian_smolvla.pretrained import load_base_backbone
from robot_arm.robot_schema import CARTESIAN_ACTION_NAMES


class CartesianSmolVLAPolicy(SmolVLAPolicy):
    config_class = CartesianSmolVLAConfig
    name = "cartesian_smolvla"

    def __init__(self, config: CartesianSmolVLAConfig, **kwargs):
        assert config.chunk_size == config.n_action_steps == 1, (
            f"Cartesian VLA requires chunk_size=1 and n_action_steps=1; "
            f"got chunk_size={config.chunk_size}, n_action_steps={config.n_action_steps}"
        )
        assert config.max_action_dim == config.action_feature.shape[0] == len(CARTESIAN_ACTION_NAMES) + 1, (
            "Cartesian VLA requires eight action dimensions: seven commands and completion"
        )
        super().__init__(config, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, **kwargs):
        if str(pretrained_name_or_path) != "lerobot/smolvla_base":
            kwargs["strict"] = True
            return super().from_pretrained(pretrained_name_or_path, **kwargs)

        config = kwargs.pop("config")
        policy = cls(config, **kwargs)
        model_file = hf_hub_download(repo_id=str(pretrained_name_or_path), filename=SAFETENSORS_SINGLE_FILE)
        load_base_backbone(policy, model_file)
        policy.to(config.device)
        policy.eval()
        return policy
