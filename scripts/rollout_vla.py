import os
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from robot_arm.recorder import EpisodeRecorder
from robot_arm.policies import VLACartesianPolicy
from robot_arm.primitive_policy import ScriptedPrimitiveGeneratorPolicy
from robot_arm.episode_runner import EpisodeRunner
from robot_arm.rollout_config import setup_rollout_context


@hydra.main(version_base=None, config_path="../conf", config_name="rollout")
def main(rollout_cfg: DictConfig):
    hydra_cfg = HydraConfig.get()
    run_dir = hydra_cfg.runtime.output_dir

    merged_cfg, env, low_level_policy = setup_rollout_context(rollout_cfg, run_dir)

    cartesian_policy = VLACartesianPolicy()
    primitive_policy = ScriptedPrimitiveGeneratorPolicy(merged_cfg)

    output_dir = os.path.join(run_dir, "recordings")
    recorder = EpisodeRecorder(
        output_dir=output_dir,
        cfg=merged_cfg,
        episode_name="vla_run_01",
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

    raise NotImplementedError(
        "SmolVLA rollout needs a trained primitive-completion output before it can advance the scripted primitive policy."
    )


if __name__ == "__main__":
    main()
