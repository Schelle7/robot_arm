import os
import hydra
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from hydra.core.hydra_config import HydraConfig

from robot_arm.sim_arm import SimArm
from robot_arm.env import RobotEnv


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    if not os.path.exists(cfg.model_path):
        raise FileNotFoundError(f"Simulation model not found at {cfg.model_path}")

    # 1. Initialize simulation backend
    print("Initializing SimArm...")
    arm = SimArm(model_path=cfg.model_path)

    # 2. Wrap in Gym environment
    print("Creating RobotEnv...")
    env = RobotEnv(arm=arm, max_steps=cfg.max_steps)

    # 3. Setup PPO Agent
    print(f"Setting up PPO on {cfg.experiment.ppo.device}...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=cfg.experiment.ppo.learning_rate,
        n_steps=cfg.experiment.ppo.n_steps,
        batch_size=cfg.experiment.ppo.batch_size,
        device=cfg.experiment.ppo.device
    )

    # Automatically save checkpoints during training inside hydra's working directory
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000, 
        save_path=checkpoint_dir,
        name_prefix="rl_model"
    )

    # 4. Train the agent
    print(f"Starting training for {cfg.total_timesteps} timesteps...")
    
    model.learn(
        total_timesteps=cfg.total_timesteps, 
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # 5. Save the final model
    final_model_path = os.path.join(output_dir, "ppo_robot_arm_final")
    model.save(final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()
