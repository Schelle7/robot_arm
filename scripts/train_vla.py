import os
import subprocess
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../conf", config_name="train_vla")
def main(cfg: DictConfig) -> None:
    project_root = Path(__file__).resolve().parent.parent
    dataset_root = Path(to_absolute_path(cfg.dataset_root))
    output_dir = Path(to_absolute_path(cfg.output_dir))
    subprocess.run(
        [
            "lerobot-train",
            f"--policy.type={cfg.policy.type}",
            f"--policy.pretrained_path={cfg.policy.pretrained_path}",
            f"--policy.discover_packages_path={cfg.policy.discover_packages_path}",
            f"--policy.chunk_size={cfg.policy.chunk_size}",
            f"--policy.n_action_steps={cfg.policy.n_action_steps}",
            f"--dataset.repo_id={dataset_root.name}",
            f"--dataset.root={dataset_root}",
            f"--output_dir={output_dir}",
            f"--job_name={cfg.job_name}",
            f"--steps={cfg.steps}",
            f"--batch_size={cfg.batch_size}",
            f"--policy.push_to_hub={str(cfg.policy.push_to_hub).lower()}",
            f"--wandb.enable={str(cfg.wandb.enable).lower()}",
        ],
        check=True,
        env={**os.environ, **dict(cfg.environment)},
    )
    latest_run_file = project_root / "outputs" / "train_vla" / "latest_run.txt"
    latest_run_file.write_text(str(output_dir))


if __name__ == "__main__":
    main()
