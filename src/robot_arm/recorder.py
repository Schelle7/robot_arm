import os
import json
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
            "joint_positions": obs["joint_positions"].tolist(),
            "reward": float(reward),
            "dense_trajectory": [
                {
                    "joint_positions": step["joint_positions"].tolist(),
                    "action": step["action"].tolist(),
                }
                for step in info.get("dense_trajectory", [])
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
        Write the buffered metadata to disk to complete the episode.
        """
        metadata_path = os.path.join(self.episode_dir, "metadata.jsonl")
        with open(metadata_path, "w") as f:
            for frame in self.frames:
                f.write(json.dumps(frame) + "\n")
