import logging
from omegaconf import DictConfig

from robot_arm.backends.sim_arm import SimBackend
from robot_arm.backends.real_arm import RealArm
from robot_arm.envs.env import RobotEnv
from robot_arm.envs.safety import SafeArmWrapper

log = logging.getLogger(__name__)


def make_env(cfg: DictConfig, output_dir: str):
    """
    Creates a standardized instance of the underlying backend and RobotEnv wrapper
    based on the loaded DictConfig. Helper to avoid duplicating this setup between
    the learner and the workers.
    """
    height = cfg.camera.height
    width = cfg.camera.width

    if cfg.backend == "sim":
        backend = SimBackend(
            model_path=cfg.model_path,
            height=height,
            width=width,
            initial_joint_range_percent=cfg.control.initial_joints.range_percent,
        )
    elif cfg.backend == "real":
        # Imports protected to avoid needing lerobot/hardware on simulation-only machines
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        # Instantiate the LeRobot follower using settings from our Hydra config
        follower = SO101Follower(
            SO101FollowerConfig(port=cfg.hardware.port, id=cfg.hardware.calibration_id)
        )
        follower.connect(calibrate=True)

        # Initialize our wrapper using the raw connected bus natively
        backend = RealArm(bus=follower.bus, model_path=cfg.model_path)
        # Prevent garbage collection of the follower object
        backend.follower_keepalive = follower
    else:
        raise ValueError(f"Unknown backend requested: {cfg.backend}")

    safe_backend = SafeArmWrapper(
        backend_arm=backend,
        min_pos=cfg.safety.min_position_radians,
        max_pos=cfg.safety.max_position_radians,
        max_temperature=cfg.safety.max_temperature_celsius,
        load_ema_alpha=cfg.safety.load_ema_alpha,
        max_smoothed_load=cfg.safety.max_smoothed_load,
    )
    env = RobotEnv(
        arm=safe_backend,
        cfg=cfg,
        output_dir=output_dir,
    )
    return env
