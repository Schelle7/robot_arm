import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import logging
from hydra.core.hydra_config import HydraConfig
from pathlib import Path

from robot_arm.distributed import run_distributed_training

log = logging.getLogger(__name__)


def load_continuation_config(cfg: DictConfig) -> DictConfig:
    if "continue_from" not in cfg:
        return cfg
    if cfg.continue_from is None:
        raise ValueError("continue_from must point to a SAC checkpoint for continuation training.")

    checkpoint_path = Path(cfg.continue_from).resolve()
    previous_config_path = checkpoint_path.parent.parent / ".hydra" / "config.yaml"
    previous_cfg = OmegaConf.load(previous_config_path)
    continuation_cfg = OmegaConf.create(
        {
            "continue_from": str(checkpoint_path),
            "training": {
                "learning_starts": cfg.training.learning_starts,
                "total_training_steps": cfg.training.total_training_steps,
            },
        }
    )
    return OmegaConf.merge(previous_cfg, continuation_cfg)


def save_run_config(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir
    saved_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    saved_cfg.continue_from = False
    OmegaConf.save(saved_cfg, Path(output_dir) / ".hydra" / "config.yaml")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train_low_level(cfg: DictConfig):
    cfg = load_continuation_config(cfg)
    save_run_config(cfg)
    device = torch.device(cfg.device)
    output_dir = HydraConfig.get().runtime.output_dir

    print(f"Hydra run directory: {output_dir}", flush=True)
    print(f"tensorboard --logdir={output_dir}", flush=True)

    run_distributed_training(cfg, device)


if __name__ == "__main__":
    train_low_level()
