from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from robot_arm.envs.factory import make_env
from robot_arm.recording.git_snapshot import snapshot_git_state
from robot_arm.recording.model_snapshot import snapshot_model_files
from robot_arm.policies.checkpoints import resolve_joint_checkpoint
from robot_arm.policies.numpy_policy import load_numpy_policy


def assert_matching_policy_constraints(
    current_cfg: DictConfig,
    saved_cfg: DictConfig,
) -> None:
    frequency_fields = ("cartesian", "joint", "mujoco")
    for field in frequency_fields:
        current_value = current_cfg.control.frequencies[field]
        saved_value = saved_cfg.control.frequencies[field]
        if current_value != saved_value:
            raise ValueError(f"Rollout frequency {field!r} ({current_value}) does not match the policy frequency ({saved_value}).")

    safety_fields = (
        "duty_ema_seconds",
        "max_smoothed_duty",
    )
    for field in safety_fields:
        current_value = current_cfg.safety[field]
        saved_value = saved_cfg.safety[field]
        if current_value != saved_value:
            raise ValueError(f"Rollout safety constraint {field!r} ({current_value}) does not match the policy constraint ({saved_value}).")


def load_policy_config(checkpoint_path: str) -> DictConfig:
    checkpoint_directory = Path(checkpoint_path).resolve().parent.parent
    saved_config_path = checkpoint_directory / ".hydra" / "config.yaml"
    return OmegaConf.load(saved_config_path)


def policy_model_path(checkpoint_path: str, policy_cfg: DictConfig) -> str:
    checkpoint_directory = Path(checkpoint_path).resolve().parent.parent
    model_filename = Path(policy_cfg.model_path).name
    return str(checkpoint_directory / "model" / model_filename)


def setup_rollout_context(cfg: DictConfig, run_dir: str):
    checkpoint_path = resolve_joint_checkpoint(cfg.policy_name)
    policy_cfg = load_policy_config(checkpoint_path)
    assert_matching_policy_constraints(cfg, policy_cfg)

    merged_cfg = OmegaConf.merge(policy_cfg, cfg)
    merged_cfg.policy_name = checkpoint_path
    merged_cfg.model_path = policy_model_path(checkpoint_path, policy_cfg)

    hydra_dir = Path(run_dir) / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(merged_cfg, hydra_dir / "config.yaml")

    snapshot_model_files(merged_cfg.model_path, run_dir)
    snapshot_git_state(run_dir)

    joint_policy = load_numpy_policy(checkpoint_path)
    env = make_env(merged_cfg, run_dir)
    return merged_cfg, env, joint_policy
