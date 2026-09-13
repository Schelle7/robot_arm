import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../conf", config_name="train_vla_dagger")
def main(cfg: DictConfig) -> None:
    os.environ.update(dict(cfg.environment))
    # Hugging Face reads offline settings during import.
    from robot_arm.training.dagger import run_dagger

    run_dagger(cfg, Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()
