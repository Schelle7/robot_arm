import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from robot_arm.envs.factory import make_env


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir
    env = make_env(cfg, output_dir)
    try:
        observation = env.reset()
        print("Staging complete. Final joint positions:")
        for name, position in zip(env.motor_order, observation["joint_positions"]):
            print(f"  {name}: {position:.4f} rad")
    finally:
        env.arm.disconnect()


if __name__ == "__main__":
    main()
