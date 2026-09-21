import os
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from robot_arm.recording.recorder import EpisodeRecorder
from robot_arm.policies.cartesian import ScriptedCartesianPolicy
from robot_arm.policies.primitive_generator import ScriptedPrimitiveGeneratorPolicy
from robot_arm.episode_runner import EpisodeRunner
from robot_arm.rollout_config import setup_rollout_context


@hydra.main(version_base=None, config_path="../conf", config_name="rollout")
def main(rollout_cfg: DictConfig):
    hydra_cfg = HydraConfig.get()
    run_dir = hydra_cfg.runtime.output_dir

    merged_cfg, env, joint_policy = setup_rollout_context(rollout_cfg, run_dir)

    with env:
        cartesian_policy = ScriptedCartesianPolicy(merged_cfg)
        primitive_policy = ScriptedPrimitiveGeneratorPolicy(merged_cfg)

        if not merged_cfg.runtime.capture_camera:
            print("WARNING: Camera capture is disabled; recording numeric episode data without images.")

        output_dir = os.path.join(run_dir, "waypoint_recording")
        recorder = EpisodeRecorder(
            output_dir=output_dir,
            cfg=merged_cfg,
            episode_name="waypoint_sanity_check",
        )

        runner = EpisodeRunner(
            cfg=merged_cfg,
            env=env,
            joint_policy=joint_policy,
            primitive_policy=primitive_policy,
            cartesian_policy=cartesian_policy,
            training=False,
            recorder=recorder,
            replay_buffer=None,
            metrics_queue=None,
            weights_queue=None,
            progress=None,
        )

        runner.run_episode(generate_primitives=True)


if __name__ == "__main__":
    main()
