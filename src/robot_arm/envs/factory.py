import numpy as np
import logging
from omegaconf import DictConfig

from robot_arm.backends.sim_arm import SimBackend
from robot_arm.envs.env import RobotEnv
from robot_arm.envs.safety import SafeArmWrapper

log = logging.getLogger(__name__)

def make_env(cfg: DictConfig, minimize_visuals=False):
    """
    Creates a standardized instance of the underlying SimBackend and RobotEnv wrapper 
    based on the loaded DictConfig. Helper to avoid duplicating this setup between 
    the learner and the workers.
    """
    height = 10 if minimize_visuals else cfg.camera.height
    width = 10 if minimize_visuals else cfg.camera.width
    
    sim_backend = SimBackend(
        model_path=cfg.model_path, height=height, width=width
    )
    safe_backend = SafeArmWrapper(
        backend_arm=sim_backend,
        min_pos=cfg.safety.min_position_radians,
        max_pos=cfg.safety.max_position_radians
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
