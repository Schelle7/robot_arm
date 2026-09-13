from pathlib import Path
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from robot_arm.data.collection import collect_episodes
from robot_arm.policies.cartesian import ScriptedCartesianPolicy
from robot_arm.rollout_config import setup_rollout_context


@hydra.main(version_base=None, config_path="../conf", config_name="collect_data")
def main(collect_cfg: DictConfig):
    hydra_cfg = HydraConfig.get()
    run_dir = hydra_cfg.runtime.output_dir

    merged_cfg, env, low_level_policy = setup_rollout_context(collect_cfg, run_dir)

    cartesian_policy = ScriptedCartesianPolicy(merged_cfg)
    collect_episodes(merged_cfg, env, low_level_policy, cartesian_policy, Path(run_dir), int(collect_cfg.num_episodes))


if __name__ == "__main__":
    main()
