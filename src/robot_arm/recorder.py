import os
import numpy as np
from typing import Dict, List, Any
from PIL import Image


class EpisodeRecorder:
    """
    Records interactions for a single episode and saves them to disk.
    Images are saved as JPEGs or PNGs, and numeric data is saved in a JSON lines file
    or basic NPZ format for easy loading into dataset wrappers later.
    """

    def __init__(
        self, output_dir: str, jpeg_quality: int, episode_name: str = "episode_0"
    ):
        self.output_dir = output_dir
        self.episode_dir = os.path.join(output_dir, episode_name)
        self.images_dir = os.path.join(self.episode_dir, "images")
        self.jpeg_quality = jpeg_quality
        self.frames: List[Dict[str, Any]] = []

        os.makedirs(self.images_dir, exist_ok=True)

    def record_reset(
        self,
        obs: Dict[str, np.ndarray],
        info: Dict[str, Any],
        instruction: str = "",
    ):
        """
        Record the initial state immediately after resetting the environment.
        """
        step_idx = -1
        image_path = self._save_image(step_idx, info["image"])

        frame_data = {
            "step": step_idx,
            "instruction": instruction,
            "image_path": image_path,
            "joint_positions": obs["joint_positions"].copy(),
            "privileged_end_effector_pose_7d": info[
                "privileged_end_effector_pose_7d"
            ].copy(),
            "high_level_action": np.array([]),  # No action taken yet
            "reward": 0.0,
            "dense_trajectory": [],
        }
        self.frames.append(frame_data)

    def step(
        self,
        step_idx: int,
        obs: Dict[str, np.ndarray],
        reward: float,
        info: Dict[str, Any],
        instruction: str = "",
    ):
        """
        Record a single transition step in the environment.
        """
        image_path = self._save_image(step_idx, info["image"])

        # Buffer numeric state
        frame_data = {
            "step": step_idx,
            "instruction": instruction,
            "image_path": image_path,
            "joint_positions": obs["joint_positions"].copy(),
            "privileged_end_effector_pose_7d": info[
                "privileged_end_effector_pose_7d"
            ].copy(),
            "high_level_action": info["high_level_action"].copy(),
            "reward": float(reward),
            "dense_trajectory": [
                {
                    "global_step": step["global_step"],
                    "chunk_step": step["chunk_step"],
                    "joint_positions": step["joint_positions"].copy(),
                    "action": step["action"].copy(),
                }
                for step in info["dense_trajectory"]
            ],
        }
        self.frames.append(frame_data)

    def _save_image(self, step_idx: int, pixels: np.ndarray) -> str:
        """Saves the camera frame to disk and returns the relative path."""
        image_path = f"images/frame_{step_idx:04d}.jpg"
        abs_image_path = os.path.join(self.episode_dir, image_path)

        img_array = pixels
        img = Image.fromarray(img_array)
        img.save(abs_image_path, format="JPEG", quality=self.jpeg_quality)

        return image_path

    def save(self):
        """
        Write the buffered data to disk as a compressed .npz archive.
        """
        episode_path = os.path.join(self.episode_dir, "episode.npz")

        data_dict = {
            "step": np.array([f["step"] for f in self.frames], dtype=np.int32),
            "instruction": np.array([f["instruction"] for f in self.frames], dtype=str),
            "image_path": np.array([f["image_path"] for f in self.frames], dtype=str),
            "joint_positions": np.array(
                [f["joint_positions"] for f in self.frames], dtype=np.float32
            ),
            "privileged_end_effector_pose_7d": np.array(
                [f["privileged_end_effector_pose_7d"] for f in self.frames],
                dtype=np.float32,
            ),
            "high_level_action": np.array(
                [f["high_level_action"] for f in self.frames], dtype=object
            ),
            "reward": np.array([f["reward"] for f in self.frames], dtype=np.float32),
            "dense_trajectory": np.array(
                [f["dense_trajectory"] for f in self.frames], dtype=object
            ),
        }

        np.savez_compressed(episode_path, **data_dict)
