import os
import hydra
from hydra.core.hydra_config import HydraConfig
import logging
from omegaconf import DictConfig
import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv

from robot_arm.sim_arm import SimBackend
from robot_arm.env import RobotEnv
from robot_arm.policies import WaypointPolicy
from robot_arm.coordinator import Coordinator
from robot_arm.safety import SafeArmWrapper

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train_low_level(cfg: DictConfig):
    # 1. Initialize Backend and Env Builder
    log.info(f"Initializing {cfg.training.parallel_envs} Parallel Simulation Environments...")

    def make_env():
        # Each subprocess needs its own isolated instance of MuJoCo and the wrapper
        sim_backend = SimBackend(
            model_path=cfg.model_path, height=cfg.camera.height, width=cfg.camera.width
        )
        safe_backend = SafeArmWrapper(
            backend_arm=sim_backend,
            min_pos=cfg.safety.min_position_radians,
            max_pos=cfg.safety.max_position_radians
        )
        return RobotEnv(
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

    # Wrap in SubprocVecEnv to spin up multi-CPU execution natively
    vec_env = SubprocVecEnv([make_env for _ in range(cfg.training.parallel_envs)])

    # 2. Get device
    device = torch.device(cfg.experiment.device)

    # 3. Setup SAC agent
    log.info("Initializing SAC...")

    model = SAC(
        "MultiInputPolicy",
        vec_env,
        learning_rate=cfg.training.learning_rate,
        buffer_size=cfg.training.buffer_size,
        learning_starts=cfg.training.learning_starts,
        batch_size=cfg.training.batch_size,
        tau=cfg.training.tau,
        train_freq=cfg.training.train_freq,
        gradient_steps=cfg.training.gradient_steps,
        gamma=1.0,  # gamma=1.0 because our task is finite horizon (the chunk)
        verbose=1,
        device=device,
    )

    # Initialize the SB3 logger explicitly since we are hijacking the training loop
    hydra_cfg = HydraConfig.get()
    from stable_baselines3.common.logger import configure

    logger = configure(hydra_cfg.runtime.output_dir, ["stdout", "csv"])
    model.set_logger(logger)

    # 4. Initialize synthetic Waypoint Policy
    log.info("Initializing Waypoint Policy...")

    high_level_policy = WaypointPolicy(
        trajectory_length=cfg.trajectory_length,
        speed=cfg.training.waypoint_speed,
    )

    coordinator = Coordinator(
        env=vec_env,
        high_level_policy=high_level_policy,
        low_level_policy=model,
        high_level_hz=cfg.frequencies.high_level,
        low_level_hz=cfg.frequencies.low_level,
        training=True,
    )

    log.info(f"Starting training for {cfg.training.num_episodes} episodes...")

    for episode in range(cfg.training.num_episodes):
        obs = vec_env.reset()
        
        episode_reward = 0.0
        terminated = False
        truncated = False

        from robot_arm.waypoints import generate_grab_waypoints
        
        # Pull box pose from the first environment's attributes directly since 
        # `vec_env.reset()` only returns `obs` in standard SB3 wrappers (not `obs, info`).
        box_pose_6d = vec_env.env_method("get_privileged_box_pose_6d")[0]

        high_level_policy.waypoints = generate_grab_waypoints(
            box_pose_6d=box_pose_6d,
            lift_height=cfg.training.lift_height,
            gripper_open=cfg.training.gripper_open,
            gripper_closed=cfg.training.gripper_closed,
        )
        high_level_policy.current_wp_idx = 0
        
        # In SB3, vec_env.step() returns info automatically, but vec_env.reset() only returns obs.
        # However, VecEnv provides a method for getting the info natively.
        info = vec_env.reset_infos
        
        done = np.zeros(vec_env.num_envs, dtype=bool)

        while not np.any(done):
            # Coordinator naturally executes training bounds natively
            obs, reward, terminated, truncated, info = coordinator.step(
                obs, info, instruction="grab the box"
            )
            done = terminated | truncated
            # episode_reward is now an array of size vec_env.num_envs
            episode_reward += reward

            if coordinator.global_step % cfg.training.save_freq == 0:
                hydra_cfg = HydraConfig.get()
                output_dir = os.path.join(hydra_cfg.runtime.output_dir, "checkpoints")
                os.makedirs(output_dir, exist_ok=True)
                model.save(
                    os.path.join(
                        output_dir, f"sac_manual_step_{coordinator.global_step}"
                    )
                )

        log.info(
            f"Episode {episode} | Reward: {np.mean(episode_reward):.2f} (std: {np.std(episode_reward):.2f}) | Steps: {coordinator.global_step}"
        )


if __name__ == "__main__":
    train_low_level()
