import json
import os
from pathlib import Path

import hydra
import mujoco
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from robot_arm.envs.factory import make_env
from robot_arm.episode_runner import EpisodeRunner
from robot_arm.git_snapshot import snapshot_git_state
from robot_arm.model_snapshot import snapshot_model_files
from robot_arm.policies import FixedDutyPolicy, ScriptedCartesianPolicy
from robot_arm.primitive_policy import ScriptedPrimitiveGeneratorPolicy
from robot_arm.recorder import EpisodeRecorder
from robot_arm.replay import load_recorded_timing, recorded_model_path
from robot_arm.robot_schema import MOTOR_ORDER


@hydra.main(version_base=None, config_path="../conf", config_name="rollout_fixed_duty")
def main(cfg: DictConfig):
    request = json.loads(Path(cfg.branch_request_path).read_text())
    source_episode_path = request["episode_path"]
    source_cfg = load_recorded_timing(source_episode_path)
    branch_cfg = OmegaConf.create(OmegaConf.to_container(source_cfg, resolve=True))
    duties = np.array([request["duties"][motor] for motor in MOTOR_ORDER], dtype=np.float32)
    duration_seconds = float(request["duration_seconds"])
    frame_index = int(request["frame_index"])

    run_dir = HydraConfig.get().runtime.output_dir
    branch_cfg.control.max_seconds = duration_seconds
    branch_cfg.model_path = recorded_model_path(source_episode_path, source_cfg)
    branch_cfg.policy_name = "fixed_duty"

    hydra_dir = Path(run_dir) / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(branch_cfg, hydra_dir / "config.yaml")
    snapshot_model_files(branch_cfg.model_path, run_dir)
    snapshot_git_state(run_dir)

    source_data = np.load(source_episode_path, allow_pickle=True)
    env = make_env(branch_cfg, run_dir)
    mujoco.mj_forward(env.arm.model, env.arm.data)

    recorder = EpisodeRecorder(
        output_dir=os.path.join(run_dir, "recordings"),
        cfg=branch_cfg,
        episode_name="fixed_duty_branch",
    )
    runner = EpisodeRunner(
        cfg=branch_cfg,
        env=env,
        low_level_policy=FixedDutyPolicy(duties),
        primitive_policy=ScriptedPrimitiveGeneratorPolicy(branch_cfg),
        cartesian_policy=ScriptedCartesianPolicy(branch_cfg),
        training=False,
        recorder=recorder,
        replay_buffer=None,
        metrics_queue=None,
        weights_queue=None,
    )
    runner.run_episode_from_sim_state(
        generate_primitives=True,
        qpos=source_data["qpos"][frame_index],
        qvel=source_data["qvel"][frame_index],
    )


if __name__ == "__main__":
    main()