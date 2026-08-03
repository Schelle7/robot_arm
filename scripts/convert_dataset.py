import os
import hydra
from omegaconf import DictConfig

from robot_arm.data.lerobot_converter import convert_to_lerobot


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Expects the source dir containing the raw rollouts.
    # e.g. python scripts/convert_dataset.py source_dir=outputs/data_collection/2026-07-27/12-00-00/recordings target_name=waypoint_vla_01

    if "source_dir" not in cfg:
        raise ValueError(
            "Must provide source_dir! Example: source_dir=outputs/data_collection/.../recordings"
        )

    target_name = cfg["target_name"]

    # Place target exactly in the datasets/ folder at the root project level
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    target_dir = os.path.join(base_dir, "datasets", target_name)

    convert_to_lerobot(
        source_dir=cfg.source_dir,
        target_dir=target_dir,
        fps=cfg.control.frequencies.high_level,
    )


if __name__ == "__main__":
    main()
