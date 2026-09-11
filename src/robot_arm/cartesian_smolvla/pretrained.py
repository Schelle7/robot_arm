from safetensors import safe_open
from safetensors.torch import load_model
from torch import nn


def load_base_backbone(policy, model_file: str) -> None:
    action_in = policy.model.action_in_proj
    action_out = policy.model.action_out_proj
    with safe_open(model_file, framework="pt", device="cpu") as weights:
        in_shape = weights.get_slice("model.action_in_proj.weight").get_shape()
        out_shape = weights.get_slice("model.action_out_proj.weight").get_shape()

    # Load the base architecture strictly, then discard its action projections.
    backbone = nn.Module()
    backbone.model = policy.model
    backbone.model.action_in_proj = nn.Linear(in_shape[1], in_shape[0])
    backbone.model.action_out_proj = nn.Linear(out_shape[1], out_shape[0])
    load_model(backbone, model_file, strict=True)
    backbone.model.action_in_proj = action_in
    backbone.model.action_out_proj = action_out
