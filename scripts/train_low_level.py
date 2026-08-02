import hydra
from omegaconf import DictConfig
import torch
import logging

from robot_arm.distributed import run_distributed_training

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train_low_level(cfg: DictConfig):
    device = torch.device(cfg.device)

    run_distributed_training(cfg, device)


if __name__ == "__main__":
    train_low_level()
