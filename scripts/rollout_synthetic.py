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
    instruction = "Reach for the red box."
    
    # For now, hardcode the ReplayPolicy setup as before. 
    # Later this is decided by hydra config (cfg.policy)
    # The box is at (x=0.3, y=0.0, z=0.05). Start at 0, target a reach.
    start_pos = np.zeros(6, dtype=np.float32)
    # Target joint angles: shoulder_pan=0, shoulder_lift=0.6, elbow_flex=1.2, wrist_flex=-0.8, wrist_roll=0, gripper=0
    end_pos = np.array([0.0, 0.0, 1.2, -0.8, 0.0, 0.0], dtype=np.float32)
    max_steps_calc = int(cfg.max_seconds * cfg.frequencies.high_level)
    policy = ReplayPolicy(start_pos=start_pos, end_pos=end_pos, num_steps=max_steps_calc)

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
