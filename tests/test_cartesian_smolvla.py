from types import SimpleNamespace

import torch
from torch import nn

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching

from robot_arm.cartesian_smolvla.modeling_cartesian_smolvla import CartesianSmolVLAPolicy
from robot_arm.robot_schema import PRIMITIVE_COMPLETION


class VLMStub(nn.Module):
    def forward(self, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, fill_kv_cache):
        prefix_embs, suffix_embs = inputs_embeds
        return (prefix_embs + 1.0, suffix_embs + 2.0), None


class FlowModelStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(chunk_size=2)
        self.vlm_with_expert = VLMStub()
        self.action_out_proj = nn.Identity()

    def sample_noise(self, shape, device):
        raise AssertionError("The test supplies noise explicitly.")

    def sample_time(self, batch_size, device):
        raise AssertionError("The test supplies time explicitly.")

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state):
        prefix_embs = torch.stack([state[:, 0], state[:, 0] + 1.0], dim=1).unsqueeze(-1)
        prefix_pad_masks = torch.ones(state.shape[0], 2, dtype=torch.bool)
        prefix_att_masks = torch.tensor([[False, True]]).expand(state.shape[0], -1)
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    def embed_suffix(self, noisy_actions, time):
        suffix_pad_masks = torch.ones(noisy_actions.shape[:2], dtype=torch.bool)
        suffix_att_masks = torch.ones(noisy_actions.shape[:2], dtype=torch.bool)
        return noisy_actions, suffix_pad_masks, suffix_att_masks

    forward = VLAFlowMatching.forward


def make_policy() -> CartesianSmolVLAPolicy:
    policy = CartesianSmolVLAPolicy.__new__(CartesianSmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        adapt_to_pi_aloha=False,
        action_feature=SimpleNamespace(shape=(2,)),
        max_action_dim=2,
        chunk_size=2,
    )
    policy.model = FlowModelStub()
    policy.completion_head = nn.Linear(1, 1, bias=False)
    policy.completion_head.weight.data.fill_(0.5)
    policy.prepare_images = lambda batch: ([batch["image"]], [torch.ones(2, dtype=torch.bool)])
    policy.prepare_state = lambda batch: batch["state"]
    policy.prepare_action = lambda batch: batch["action"]
    return policy


def test_single_pass_flow_losses_match_stock_smolvla():
    policy = make_policy()
    batch = {
        "image": torch.zeros(2, 1),
        "state": torch.zeros(2, 1),
        "action": torch.zeros(2, 2, 3),
        "observation.language.tokens": torch.zeros(2, 1, dtype=torch.long),
        "observation.language.attention_mask": torch.ones(2, 1, dtype=torch.bool),
        "action_is_pad": torch.tensor([[False, True], [False, False]]),
        PRIMITIVE_COMPLETION: torch.tensor([0.0, 1.0]),
    }
    noise = torch.zeros_like(batch["action"])
    time = torch.zeros(2)

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    actions = policy.prepare_action(batch)
    stock_flow_losses = policy.model.forward(
        images,
        img_masks,
        batch["observation.language.tokens"],
        batch["observation.language.attention_mask"],
        state,
        actions,
        noise,
        time,
    )
    custom_flow_losses, completion_logits = policy._flow_losses_and_completion_logits(
        images,
        img_masks,
        batch["observation.language.tokens"],
        batch["observation.language.attention_mask"],
        state,
        actions,
        noise,
        time,
    )

    torch.testing.assert_close(custom_flow_losses, stock_flow_losses)
    assert completion_logits.shape == (2,)


def test_custom_forward_adds_completion_loss_to_stock_action_loss():
    policy = make_policy()
    batch = {
        "image": torch.zeros(2, 1),
        "state": torch.zeros(2, 1),
        "action": torch.zeros(2, 2, 3),
        "observation.language.tokens": torch.zeros(2, 1, dtype=torch.long),
        "observation.language.attention_mask": torch.ones(2, 1, dtype=torch.bool),
        "action_is_pad": torch.tensor([[False, True], [False, False]]),
        PRIMITIVE_COMPLETION: torch.tensor([0.0, 1.0]),
    }
    noise = torch.zeros_like(batch["action"])
    time = torch.zeros(2)

    stock_per_sample_loss, stock_loss_dict = SmolVLAPolicy.forward(
        policy,
        batch,
        noise=noise,
        time=time,
        reduction="none",
    )
    custom_per_sample_loss, custom_loss_dict = CartesianSmolVLAPolicy.forward(
        policy,
        batch,
        noise=noise,
        time=time,
        reduction="none",
    )
    completion_logits = policy._flow_losses_and_completion_logits(
        *policy.prepare_images(batch),
        batch["observation.language.tokens"],
        batch["observation.language.attention_mask"],
        policy.prepare_state(batch),
        policy.prepare_action(batch),
        noise,
        time,
    )[1]
    expected_completion_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        completion_logits,
        batch[PRIMITIVE_COMPLETION],
        reduction="none",
    )
    torch.testing.assert_close(custom_per_sample_loss, stock_per_sample_loss + expected_completion_loss)
    assert custom_loss_dict["action_loss"] == stock_loss_dict["loss"]
    assert custom_loss_dict["losses_after_forward"] == stock_loss_dict["losses_after_forward"]
    assert custom_loss_dict["losses_after_in_ep_bound"] == stock_loss_dict["losses_after_in_ep_bound"]
    assert custom_loss_dict["losses_after_rm_padding"] == stock_loss_dict["losses_after_rm_padding"]