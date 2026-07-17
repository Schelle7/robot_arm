import os
import numpy as np

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from robot_arm.recorder import EpisodeRecorder
from robot_arm.policies import ReplayPolicy
from robot_arm.runner import execute_episode

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # 3. Setup Policy
    instruction = "Move slightly to the side and up."
    
    # For now, hardcode the ReplayPolicy setup as before. 
    # Later this is decided by hydra config (cfg.policy)
    start_pos = np.zeros(6, dtype=np.float32)
    end_pos = np.array([0.5, 0.5, -0.5, 0.0, 0.0, 0.2], dtype=np.float32)
    trajectory = np.linspace(start_pos, end_pos, cfg.max_steps)
    policy = ReplayPolicy(trajectory=trajectory)

    # 4. Initialize recorder inside Hydra's output directory
    hydra_cfg = HydraConfig.get()
    output_dir = os.path.join(hydra_cfg.runtime.output_dir, "recordings")
    
    recorder = EpisodeRecorder(
        output_dir=output_dir, 
        jpeg_quality=cfg.camera.jpeg_quality,
        episode_name="synthetic_01"
    )
    
    # 5. Execute
    execute_episode(
        cfg=cfg,
        policy=policy,
        recorder=recorder,
        instruction=instruction
    )
    
if __name__ == "__main__":
    main()
