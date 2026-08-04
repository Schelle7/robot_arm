from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def assert_matching_policy_constraints(
    current_cfg: DictConfig,
    saved_cfg: DictConfig,
) -> None:
    frequency_fields = ("high_level", "low_level", "mujoco")
    for field in frequency_fields:
        current_value = current_cfg.control.frequencies[field]
        saved_value = saved_cfg.control.frequencies[field]
        if current_value != saved_value:
            raise ValueError(
                f"Rollout frequency {field!r} ({current_value}) does not match "
                f"the policy frequency ({saved_value})."
            )

    safety_fields = (
        "max_position_radians",
        "min_position_radians",
        "max_temperature_celsius",
        "load_ema_alpha",
        "max_smoothed_load",
    )
    for field in safety_fields:
        current_value = current_cfg.safety[field]
        saved_value = saved_cfg.safety[field]
        if current_value != saved_value:
            raise ValueError(
                f"Rollout safety constraint {field!r} ({current_value}) does not "
                f"match the policy constraint ({saved_value})."
            )


def load_policy_config(checkpoint_path: str) -> DictConfig:
    checkpoint_directory = Path(checkpoint_path).resolve().parent.parent
    saved_config_path = checkpoint_directory / ".hydra" / "config.yaml"
    return OmegaConf.load(saved_config_path)


def policy_model_path(checkpoint_path: str, policy_cfg: DictConfig) -> str:
    checkpoint_directory = Path(checkpoint_path).resolve().parent.parent
    model_filename = Path(policy_cfg.model_path).name
    return str(checkpoint_directory / "model" / model_filename)
