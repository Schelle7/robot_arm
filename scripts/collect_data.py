import os
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from robot_arm.recorder import EpisodeRecorder
from robot_arm.policies import ScriptedCartesianPolicy
from robot_arm.primitive_policy import ScriptedPrimitiveGeneratorPolicy
from robot_arm.episode_runner import EpisodeRunner
from robot_arm.rollout_config import setup_rollout_context


@hydra.main(version_base=None, config_path="../conf", config_name="collect_data")
def main(collect_cfg: DictConfig):
    hydra_cfg = HydraConfig.get()
    run_dir = hydra_cfg.runtime.output_dir

    merged_cfg, env, low_level_policy = setup_rollout_context(collect_cfg, run_dir)

    cartesian_policy = ScriptedCartesianPolicy(merged_cfg)
    primitive_policy = ScriptedPrimitiveGeneratorPolicy(merged_cfg)

    recordings_dir = os.path.join(run_dir, "recordings")
    num_episodes = int(collect_cfg.num_episodes)

    print(f"Starting data collection for {num_episodes} episodes into {recordings_dir}...")

    for ep_idx in range(num_episodes):
        episode_name = f"episode_{ep_idx:04d}"
        recorder = EpisodeRecorder(
            output_dir=recordings_dir,
            cfg=merged_cfg,
            episode_name=episode_name,
        )

        runner = EpisodeRunner(
            cfg=merged_cfg,
            env=env,
            low_level_policy=low_level_policy,
            primitive_policy=primitive_policy,
            cartesian_policy=cartesian_policy,
            training=False,
            recorder=recorder,
            replay_buffer=None,
            metrics_queue=None,
            weights_queue=None,
        )

        runner.run_episode(generate_primitives=True)
        print(f"Finished episode {ep_idx + 1}/{num_episodes} ({episode_name})")

    print(f"Data collection complete! All {num_episodes} episodes saved.")


if __name__ == "__main__":
    main()
