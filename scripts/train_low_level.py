import hydra
from omegaconf import DictConfig
import torch
import logging
from hydra.core.hydra_config import HydraConfig

from robot_arm.distributed import run_distributed_training

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train_low_level(cfg: DictConfig):
    device = torch.device(cfg.device)
    output_dir = HydraConfig.get().runtime.output_dir

    print(f"Hydra run directory: {output_dir}", flush=True)
    print(f"tensorboard --logdir={output_dir}", flush=True)

    run_distributed_training(cfg, device)


if __name__ == "__main__":
    train_low_level()
