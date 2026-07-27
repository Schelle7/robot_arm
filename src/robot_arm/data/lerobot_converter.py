import os
import glob
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

def convert_to_lerobot(source_dir: str, target_dir: str, fps: int):
    """
    Parses flat .npz tracking outputs and corresponding jpegs, 
    and converts them into LeRobot/Hugging Face format using LeRobotDataset.create().
    """
    # 1. Define the dataset features.
    # We map "privileged_end_effector_pose_7d" -> observation.state
    # We map "high_level_action" -> action
    features = {
        "observation.images.camera1": {
            "dtype": "video",
            "shape": (480, 640, 3), # Matches config.yaml
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (50, 7), # traj_len=50, traj_dim=7
            "names": ["time", "action_dim"],
        },
    }

    # Use the target_dir name as the repo_id (e.g. "robot_arm_vla_dataset")
    repo_id = os.path.basename(os.path.normpath(target_dir))
    
    # Create the dataset root
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=target_dir,
        features=features,
        use_videos=True,
    )
    
    # 2. Find all episodes in the source directory
    # Expecting structures like: source_dir/waypoint_dataset_01/episode.npz
    episodes = sorted(glob.glob(os.path.join(source_dir, "**", "episode.npz"), recursive=True))
    if not episodes:
        raise FileNotFoundError(f"No episode.npz files found in {source_dir}")

    print(f"Found {len(episodes)} episodes. Converting to LeRobot dataset at {target_dir}...")

    # 3. Process each episode
    for ep_idx, ep_path in enumerate(tqdm(episodes)):
        ep_dir = os.path.dirname(ep_path)
        
        # Load the numeric data
        data = np.load(ep_path, allow_pickle=True)
        num_frames = len(data["step"])
        
        for frame_idx in range(num_frames - 1):
            # Load the corresponding image. The NPZ contains the relative path.
            # We align observation t with action t (which is technically recorded at step+1 in our loop).
            image_rel_path = data["image_path"][frame_idx]
            image_abs_path = os.path.join(ep_dir, image_rel_path)
            
            img = Image.open(image_abs_path).convert("RGB")
            
            # State t
            state = torch.tensor(data["privileged_end_effector_pose_7d"][frame_idx], dtype=torch.float32)
            
            # Action t (the action decided after seeing state t, recorded in frame t+1's metadata)
            action_raw = data["high_level_action"][frame_idx + 1]
            action = torch.tensor(action_raw, dtype=torch.float32)
                
            task_instruction = str(data["instruction"][frame_idx])

            # Add frame to the dataset
            dataset.add_frame(
                {
                    "observation.images.camera1": img,
                    "observation.state": state,
                    "action": action,
                }
            )
            
        # Complete the episode
        dataset.save_episode(task=task_instruction)

    # Finalize consolidation of parquets and info.json
    dataset.consolidate()
    print("Dataset conversion complete!")

