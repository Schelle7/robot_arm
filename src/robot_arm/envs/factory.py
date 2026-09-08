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
    if cfg.backend == "sim":
        backend = SimBackend(
            model_path=cfg.model_path,
            camera_configs=cfg.camera.cameras,
            initial_joint_mode=cfg.control.initial_joints.mode,
            initial_joint_range_percent=cfg.control.initial_joints.range_percent,
            initial_joint_positions=cfg.control.initial_joints.positions_radians,
            disable_box_collisions=cfg.runtime.disable_box_collisions,
            object_placement=cfg.scene.object_placement,
            mujoco_steps_per_control_step=cfg.control.frequencies.mujoco // cfg.control.frequencies.joint,
            servo=cfg.servo,
        )
    elif cfg.backend == "real":
        # Imports protected to avoid needing lerobot/hardware on simulation-only machines
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        # Instantiate the LeRobot follower using settings from our Hydra config
        follower = SO101Follower(SO101FollowerConfig(port=cfg.hardware.port, id=cfg.hardware.calibration_id))
        follower.connect(calibrate=True)

        # Initialize our wrapper using the raw connected bus natively
        backend = RealArm(
            bus=follower.bus,
            model_path=cfg.model_path,
            control_step_seconds=1.0 / cfg.control.frequencies.joint,
        )
        # After connect, because SO101Follower.configure writes Operating_Mode back to position.
        backend.set_pwm_mode()

        # Prevent garbage collection of the follower object
        backend.follower_keepalive = follower
        backend.configure_read_timeout(cfg.hardware.read_timeout_margin_ms)
    else:
        raise ValueError(f"Unknown backend requested: {cfg.backend}")

    safe_backend = SafeArmWrapper(
        backend_arm=backend,
        max_temperature=cfg.safety.max_temperature_celsius,
        duty_ema_seconds=cfg.safety.duty_ema_seconds,
        max_smoothed_duty=cfg.safety.max_smoothed_duty,
        read_hz=cfg.control.frequencies.joint,
    )
    env = RobotEnv(
        arm=safe_backend,
        cfg=cfg,
        output_dir=output_dir,
    )
    return env
