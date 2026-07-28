import numpy as np
import logging
from omegaconf import DictConfig

from robot_arm.backends.sim_arm import SimBackend
from robot_arm.backends.real_arm import RealArm
from robot_arm.envs.env import RobotEnv
from robot_arm.envs.safety import SafeArmWrapper

log = logging.getLogger(__name__)


def make_env(cfg: DictConfig, minimize_visuals=False):
    """
    Creates a standardized instance of the underlying backend and RobotEnv wrapper
    based on the loaded DictConfig. Helper to avoid duplicating this setup between
    the learner and the workers.
    """
    height = 10 if minimize_visuals else cfg.camera.height
    width = 10 if minimize_visuals else cfg.camera.width

    if cfg.backend == "sim":
        backend = SimBackend(model_path=cfg.model_path, height=height, width=width)
    elif cfg.backend == "real":
        # Imports protected to avoid needing lerobot/hardware on simulation-only machines
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        
        # Instantiate the LeRobot follower using settings from our Hydra config
        follower = SO101Follower(SO101FollowerConfig(port=cfg.hardware.port, id=cfg.hardware.calibration_id))
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
        max_seconds=cfg.max_seconds,
        trajectory_length=cfg.trajectory_length,
        trajectory_dim=cfg.trajectory_dim,
        pose_distance_weights=np.array(cfg.pose_distance_weights, dtype=np.float32),
        high_level_hz=cfg.frequencies.high_level,
        low_level_hz=cfg.frequencies.low_level,
        delta_action_scale=cfg.training.waypoint_speed / cfg.frequencies.low_level,
        violation_penalty_factor=cfg.safety.violation_penalty_factor,
    )
    return env
