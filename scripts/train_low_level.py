import hydra
import logging
from omegaconf import DictConfig
import numpy as np
import torch

from stable_baselines3 import SAC

from robot_arm.sim_arm import SimBackend
from robot_arm.env import RobotEnv
from robot_arm.policies import WaypointPolicy
from robot_arm.coordinator import Coordinator

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train_low_level(cfg: DictConfig):
    # 1. Initialize Backend and Env
    log.info("Initializing Simulation...")
    sim_backend = SimBackend(model_path=cfg.model_path)
    env = RobotEnv(
        arm=sim_backend,
        max_seconds=cfg.max_seconds,
        trajectory_length=cfg.trajectory_length,
        trajectory_dim=cfg.trajectory_dim,
        pose_distance_weights=np.array(cfg.pose_distance_weights, dtype=np.float32),
    )

    # 2. Get device
    device = torch.device(cfg.experiment.device)

    # 3. Setup SAC agent
    log.info("Initializing SAC...")
    # Because SAC needs to know the observation space of the rl_obs dict, we have to mock it.
    # In Coordinator we have: start_joint_positions, current_joint_positions, high_level_action, time_left
    rl_obs_space = gym.spaces.Dict(
        {
            "start_joint_positions": env.observation_space.spaces[
                "start_joint_positions"
            ],
            "current_joint_positions": env.observation_space.spaces["joint_positions"],
            "high_level_action": env.observation_space.spaces["high_level_action"],
            "time_left": env.observation_space.spaces["time_left"],
        }
    )

    # We create a dummy env just to instantiate SAC, since we are doing manual insertion
    dummy_env = gym.Env()
    dummy_env.observation_space = rl_obs_space
    dummy_env.action_space = env.action_space

    model = SAC(
        "MultiInputPolicy",
        dummy_env,
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
    replay_buffer = model.replay_buffer

    # 4. Initialize synthetic Waypoint Policy
    log.info("Initializing Waypoint Policy...")
    high_level_policy = WaypointPolicy(
        lift_height=cfg.training.lift_height,
        trajectory_length=cfg.trajectory_length,
        gripper_open=cfg.training.gripper_open,
        gripper_closed=cfg.training.gripper_closed,
        speed=cfg.training.waypoint_speed,
    )

    coordinator = Coordinator(
        env=env,
        high_level_policy=high_level_policy,
        low_level_policy=None,  # We don't want the coordinator to step it automatically here, we need the transitions
        high_level_hz=cfg.frequencies.high_level,
        low_level_hz=cfg.frequencies.low_level,
    )

    log.info(f"Starting training for {cfg.training.num_episodes} episodes...")
    global_step = 0

    for episode in range(cfg.training.num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0

        while True:
            # High Level Inference
            high_level_action = high_level_policy.get_action(
                obs, info, instruction="grab the box"
            )

            start_joint_positions = env.current_joint_angles
            env.update_path(high_level_action)

            # Manual Low Level Loop
            rl_obs = {
                "start_joint_positions": start_joint_positions,
                "current_joint_positions": env.current_joint_angles,
                "high_level_action": high_level_action,
                "time_left": np.array([coordinator.skip_frames - 1], dtype=np.float32),
            }

            for step_idx in range(coordinator.skip_frames):
                time_left = coordinator.skip_frames - step_idx - 1

                # Get action from SAC
                action, _ = model.predict(rl_obs)

                # Step env
                next_obs, reward, terminated, truncated, env_info = env.step(action)
                episode_reward += reward
                global_step += 1

                next_rl_obs = {
                    "start_joint_positions": start_joint_positions,
                    "current_joint_positions": env.current_joint_angles,
                    "high_level_action": high_level_action,
                    "time_left": np.array(
                        [time_left - 1 if time_left > 0 else 0], dtype=np.float32
                    ),
                }

                # CRITICAL: We enforce termination exactly at chunk completion to sever the bootstrap.
                chunk_terminated = time_left == 0
                actual_terminated = terminated or chunk_terminated

                # Add to SB3 buffer directly
                state_dict = {k: np.array([v]) for k, v in rl_obs.items()}
                next_state_dict = {k: np.array([v]) for k, v in next_rl_obs.items()}

                replay_buffer.add(
                    state_dict,
                    next_state_dict,
                    np.array([action]),
                    np.array([reward]),
                    np.array([actual_terminated]),
                    [env_info],
                )

                rl_obs = next_rl_obs

                # Manual optimization step
                if (
                    global_step > cfg.training.learning_starts
                    and global_step % cfg.training.train_freq == 0
                ):
                    for _ in range(cfg.training.gradient_steps):
                        model.train(
                            gradient_steps=1, batch_size=cfg.training.batch_size
                        )

                if global_step % cfg.training.save_freq == 0:
                    model.save(f"outputs/sac_manual_step_{global_step}")

                if terminated or truncated:
                    break

            if terminated or truncated:
                break

        log.info(
            f"Episode {episode} | Reward: {episode_reward:.2f} | Steps: {global_step}"
        )


if __name__ == "__main__":
    train_low_level()
