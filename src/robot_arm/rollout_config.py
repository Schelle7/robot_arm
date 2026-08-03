from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def build_rollout_config(current_cfg: DictConfig, checkpoint_path: str) -> DictConfig:
    checkpoint_directory = Path(checkpoint_path).resolve().parent.parent
    saved_config_path = checkpoint_directory / ".hydra" / "config.yaml"
    saved_cfg = OmegaConf.load(saved_config_path)

    if OmegaConf.to_container(current_cfg.safety, resolve=True) != OmegaConf.to_container(
        saved_cfg.safety, resolve=True
    ):
        raise ValueError(
            "Current rollout safety configuration does not match the saved experiment."
        )

    rollout_cfg = OmegaConf.create(
        {
            "model_path": current_cfg.model_path,
            "backend": current_cfg.backend,
            "hardware": OmegaConf.to_container(current_cfg.hardware, resolve=True),
            "camera": OmegaConf.to_container(current_cfg.camera, resolve=True),
            "runtime": OmegaConf.to_container(current_cfg.runtime, resolve=True),
            "safety": OmegaConf.to_container(current_cfg.safety, resolve=True),
            "reward": OmegaConf.to_container(current_cfg.reward, resolve=True),
            "waypoint": {
                "trajectory_length": saved_cfg.waypoint.trajectory_length,
                "trajectory_dim": saved_cfg.waypoint.trajectory_dim,
                "lift_height": current_cfg.waypoint.lift_height,
                "gripper_open": current_cfg.waypoint.gripper_open,
                "gripper_closed": current_cfg.waypoint.gripper_closed,
                "speed": current_cfg.waypoint.speed,
            },
            "control": {
                "frequencies": OmegaConf.to_container(
                    saved_cfg.control.frequencies, resolve=True
                ),
                "action_scale_radians": saved_cfg.control.action_scale_radians,
            },
            "training": {
                "detailed_metrics": current_cfg.training.detailed_metrics,
                "pose_delta_diagnostics_enabled": current_cfg.training.pose_delta_diagnostics_enabled,
            },
        }
    )

    return rollout_cfg
