import os
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from robot_arm.recorder import EpisodeRecorder
from robot_arm.policies import SmolVLAPolicyWrapper, load_latest_low_level_policy
from robot_arm.runner import execute_episode


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Setup Policy
    task = "Grip the red box."

    # Initialize the high-level policy wrapper
    policy = SmolVLAPolicyWrapper()

    # Load the latest trained low-level RL model
    low_level_policy = load_latest_low_level_policy()

    # Initialize recorder inside Hydra's output directory
    hydra_cfg = HydraConfig.get()
    output_dir = os.path.join(hydra_cfg.runtime.output_dir, "recordings")

    recorder = EpisodeRecorder(
        output_dir=output_dir,
        cfg=cfg,
        episode_name="vla_run_01",
    )

    # Execute
    execute_episode(
        cfg=cfg,
        policy=policy,
        low_level_policy=low_level_policy,
        recorder=recorder,
        task=task,
        generate_waypoints=False,
    )


if __name__ == "__main__":
    main()
