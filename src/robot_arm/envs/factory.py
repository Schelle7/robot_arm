import logging
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
        # Imports protected to avoid needing lerobot/hardware on simulation-only machines
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        # Instantiate the LeRobot follower using settings from our Hydra config
        follower = SO101Follower(SO101FollowerConfig(port=cfg.hardware.port, id=cfg.hardware.calibration_id))
        follower.connect(calibrate=True)

        # Initialize our wrapper using the raw connected bus natively
        arm = RealArm(bus=follower.bus, cfg=cfg)
        # After connect, because SO101Follower.configure writes Operating_Mode back to position.
        arm.set_pwm_mode()

        # Prevent garbage collection of the follower object
        arm.follower_keepalive = follower
        arm.configure_read_timeout(cfg.hardware.read_timeout_margin_ms)
    else:
        raise ValueError(f"Unknown backend requested: {cfg.backend}")

    env = RobotEnv(
        arm=arm,
        cfg=cfg,
        output_dir=output_dir,
    )
    return env
