import logging
from contextlib import ExitStack
from pathlib import Path
from omegaconf import DictConfig

from robot_arm.arms.sim_arm import SimArm
from robot_arm.arms.real_arm import RealArm
from robot_arm.envs.env import RobotEnv

log = logging.getLogger(__name__)


def make_env(cfg: DictConfig, output_dir: str):
    """
    Creates a standardized instance of the underlying arm and RobotEnv wrapper
    based on the loaded DictConfig. Helper to avoid duplicating this setup between
    the learner and the workers.
    """
    if cfg.backend == "sim":
        arm = SimArm(cfg)
    elif cfg.backend == "real":
        arm = RealArm(cfg)
    else:
        raise ValueError(f"Unknown backend requested: {cfg.backend}")

    with ExitStack() as cleanup:
        cleanup.callback(arm.disconnect)
        if cfg.backend == "real":
            arm.communication.start(Path(output_dir) / "hardware_communication.jsonl")
        env = RobotEnv(arm=arm, cfg=cfg, output_dir=output_dir)
        cleanup.pop_all()
        return env
