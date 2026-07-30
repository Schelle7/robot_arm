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
        self,
        output_dir: str,
        jpeg_quality: int,
        chunk_size: int,
        episode_name: str,
        record_sim_state: bool,
    ):
        self.output_dir = output_dir
        self.episode_dir = os.path.join(output_dir, episode_name)
        self.images_dir = os.path.join(self.episode_dir, "images")
        self.jpeg_quality = jpeg_quality
        self.chunk_size = chunk_size
        self.record_sim_state = record_sim_state
        self.frames: List[Dict[str, Any]] = []
        self.dense_trajectory_buffer: List[Dict[str, Any]] = []
        self.waypoints = None

        os.makedirs(self.images_dir, exist_ok=True)

    def save_waypoints(self, waypoints: np.ndarray):
        """
        Record the raw waypoints provided by the high level policy layout.
        """
        self.waypoints = (
            waypoints.copy()
        )

    def record_high_level(
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
            "privileged_end_effector_pose": info[
                "privileged_end_effector_pose"
            ].as_10d(),
            "high_level_action": info["high_level_action"].copy(),
            "reward": float(reward),
            "dense_trajectory": self.dense_trajectory_buffer.copy(),
        }
        
        if self.record_sim_state:
            frame_data["qpos"] = info["sim_state"]["qpos"].copy()
            frame_data["qvel"] = info["sim_state"]["qvel"].copy()
            
        self.frames.append(frame_data)
        self.dense_trajectory_buffer.clear()

    def append_low_level_transition(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        terminated: bool,
    ):
        """
        Record a single low-level RL transition step in the environment.
        Currently, this method just caches dense steps on the most recently added frame
        so that when we save, we can view the whole micro-trajectory that occurred during the high-level step.
        """
        if len(self.dense_trajectory_buffer) >= self.chunk_size:
            raise RuntimeError(
                f"Dense trajectory buffer overflow! Max chunk size is {self.chunk_size}."
            )

        self.dense_trajectory_buffer.append(
            {
                "obs": {
                    k: v.copy() for k, v in obs.items() if isinstance(v, np.ndarray)
                },
                "next_obs": {
                    k: v.copy()
                    for k, v in next_obs.items()
                    if isinstance(v, np.ndarray)
                },
                "action": action.copy(),
                "reward": float(reward),
                "terminated": terminated,
            }
        )

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
            "privileged_end_effector_pose": np.array(
                [f["privileged_end_effector_pose"] for f in self.frames],
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
        
        if self.record_sim_state:
            data_dict["qpos"] = np.array([f["qpos"] for f in self.frames], dtype=np.float32)
            data_dict["qvel"] = np.array([f["qvel"] for f in self.frames], dtype=np.float32)

        if self.waypoints:
            data_dict["waypoints"] = self.waypoints

        np.savez_compressed(episode_path, **data_dict)
        print(f"Saved to: {episode_path}")
