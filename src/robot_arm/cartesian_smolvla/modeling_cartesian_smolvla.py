import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from robot_arm.cartesian_smolvla.configuration_cartesian_smolvla import CartesianSmolVLAConfig
from robot_arm.robot_schema import PRIMITIVE_COMPLETION


class CartesianSmolVLAPolicy(SmolVLAPolicy):
    config_class = CartesianSmolVLAConfig
    name = "cartesian_smolvla"

    def __init__(self, config: CartesianSmolVLAConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.completion_head = nn.Linear(self.model.vlm_with_expert.config.text_config.hidden_size, 1)

    def _flow_losses_and_completion_logits(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise,
        time,
    ) -> tuple[Tensor, Tensor]:
        if noise is None:
            noise = self.model.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.model.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        noisy_actions = time_expanded * noise + (1 - time_expanded) * actions
        target_velocity = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state=state,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.model.embed_suffix(noisy_actions, time)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        attention_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (prefix_out, suffix_out), _ = self.model.vlm_with_expert.forward(
            attention_mask=attention_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)
        predicted_velocity = self.model.action_out_proj(suffix_out)
        flow_losses = F.mse_loss(target_velocity, predicted_velocity, reduction="none")

        token_indices = torch.arange(prefix_pad_masks.shape[1], device=prefix_pad_masks.device)
        state_token_indices = token_indices.masked_fill(~prefix_pad_masks, -1).max(dim=1).values
        state_context = prefix_out[torch.arange(prefix_out.shape[0], device=prefix_out.device), state_token_indices]
        completion_logits = self.completion_head(state_context.to(dtype=torch.float32)).squeeze(-1)
        return flow_losses, completion_logits

    def _completion_logits(self, images, img_masks, lang_tokens, lang_masks, state) -> Tensor:
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state=state,
        )
        attention_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        (prefix_out, _), _ = self.model.vlm_with_expert.forward(
            attention_mask=attention_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            fill_kv_cache=True,
        )
        token_indices = torch.arange(prefix_pad_masks.shape[1], device=prefix_pad_masks.device)
        state_token_indices = token_indices.masked_fill(~prefix_pad_masks, -1).max(dim=1).values
        state_context = prefix_out[torch.arange(prefix_out.shape[0], device=prefix_out.device), state_token_indices]
        return self.completion_head(state_context.to(dtype=torch.float32)).squeeze(-1)

    def forward(
        self,
        batch: dict[str, Tensor],
        noise=None,
        time=None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, float]]:
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)
        flow_losses, completion_logits = self._flow_losses_and_completion_logits(
            images,
            img_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state,
            actions,
            noise,
            time,
        )

        action_dim = self.config.action_feature.shape[0]
        flow_losses = flow_losses[:, :, :action_dim]
        loss_dict = {"losses_after_forward": flow_losses.clone().mean().item()}
        actions_is_pad = batch.get("action_is_pad")
        if actions_is_pad is not None:
            flow_losses = flow_losses * (~actions_is_pad).unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = flow_losses.clone().mean().item()
        flow_losses = flow_losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = flow_losses.clone().mean().item()
        per_sample_action_loss = flow_losses.mean(dim=(1, 2))
        completion_targets = batch[PRIMITIVE_COMPLETION].float().reshape(-1)
        per_sample_completion_loss = F.binary_cross_entropy_with_logits(
            completion_logits,
            completion_targets,
            reduction="none",
        )
        per_sample_loss = per_sample_action_loss + per_sample_completion_loss
        loss_dict.update({
            "action_loss": per_sample_action_loss.mean().item(),
            "completion_loss": per_sample_completion_loss.mean().item(),
            "completion_accuracy": (
                (completion_logits >= 0) == completion_targets.bool()
            ).float().mean().item(),
            "loss": per_sample_loss.mean().item(),
        })
        if reduction == "none":
            return per_sample_loss, loss_dict
        return per_sample_loss.mean(), loss_dict

    @torch.no_grad()
    def select_action_with_completion(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        self.eval()
        batch = self._prepare_batch(batch)
        actions = self._get_action_chunk(batch)
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        completion_logits = self._completion_logits(
            images,
            img_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state,
        )
        return actions[:, 0], torch.sigmoid(completion_logits)
