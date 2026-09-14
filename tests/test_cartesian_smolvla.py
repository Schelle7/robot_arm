from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from safetensors.torch import load_model, save_model

from robot_arm.cartesian_smolvla.modeling_cartesian_smolvla import CartesianSmolVLAPolicy
from robot_arm.cartesian_smolvla.pretrained import load_base_backbone


def projection_policy(action_dim: int) -> nn.Module:
    policy = nn.Module()
    policy.model = nn.Module()
    policy.model.backbone = nn.Linear(4, 4)
    policy.model.action_in_proj = nn.Linear(action_dim, 4)
    policy.model.action_out_proj = nn.Linear(4, action_dim)
    return policy


def test_base_loading_preserves_fresh_cartesian_projections(tmp_path):
    base = projection_policy(32)
    path = str(tmp_path / "base.safetensors")
    save_model(base, path)
    policy = projection_policy(8)
    fresh = {name: value.clone() for name, value in policy.state_dict().items()}

    load_base_backbone(policy, path)

    for name, value in policy.state_dict().items():
        expected = base.state_dict()[name] if name.startswith("model.backbone.") else fresh[name]
        torch.testing.assert_close(value, expected)
    assert policy.model.action_out_proj(policy.model.action_in_proj(torch.zeros(2, 1, 8))).shape == (2, 1, 8)

    trained_path = str(tmp_path / "cartesian.safetensors")
    save_model(policy, trained_path)
    restored = projection_policy(8)
    load_model(restored, trained_path, strict=True)
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


def test_base_loading_rejects_missing_backbone_weights(tmp_path):
    base = projection_policy(32)
    del base.model.backbone
    path = str(tmp_path / "incomplete.safetensors")
    save_model(base, path)

    with pytest.raises(RuntimeError, match="backbone"):
        load_base_backbone(projection_policy(8), path)


def test_cartesian_policy_rejects_padded_action_dimension():
    config = SimpleNamespace(chunk_size=1, n_action_steps=1, max_action_dim=32, action_feature=SimpleNamespace(shape=(7,)))
    with pytest.raises(AssertionError, match="eight action dimensions"):
        CartesianSmolVLAPolicy(config)


@pytest.mark.parametrize("score", [0.2, 0.8, 1.2])
def test_rollout_splits_completion_after_denormalizing(score, monkeypatch):
    from robot_arm.policies import cartesian

    policy = cartesian.VLACartesianPolicy.__new__(cartesian.VLACartesianPolicy)
    policy.device = "cpu"
    normalized = torch.tensor([[0.0] * 7 + [score - 1.0]])
    policy.policy = SimpleNamespace(select_action=lambda batch: normalized)
    policy.preprocessor = lambda batch: batch
    policy.postprocessor = lambda action: action + 1.0
    monkeypatch.setattr(cartesian, "build_vla_observation", lambda images, state, prompt: {})

    action = policy.get_action(None, {}, np.zeros(16), 0.0, False, SimpleNamespace(prompt="open gripper"))

    np.testing.assert_array_equal(action.cartesian_action, np.ones(7))
    assert action.diagnostics["completion_score"] == pytest.approx(score)
    assert action.completes_active_primitive == (score >= 0.5)
