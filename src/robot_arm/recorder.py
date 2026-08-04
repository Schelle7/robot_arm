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
        cfg: Any,
        episode_name: str,
    ):
        self.output_dir = output_dir
        self.episode_dir = os.path.join(output_dir, episode_name)
        self.images_dir = os.path.join(self.episode_dir, "images")
        self.jpeg_quality = cfg.camera.jpeg_quality
        self.chunk_size = (
            cfg.control.frequencies.low_level // cfg.control.frequencies.high_level
        )
        self.record_sim_state = cfg.runtime.record_sim_state
        self.states: List[Dict[str, Any]] = []
        self.transitions: List[Dict[str, Any]] = []
        self.dense_trajectory_buffer: List[Dict[str, Any]] = []
        self.waypoints = None

        os.makedirs(self.images_dir, exist_ok=True)

    def save_waypoints(self, waypoints: np.ndarray):
        """
        Record the raw waypoints provided by the high level policy layout.
        """
        self.waypoints = waypoints.copy()

    def record_initial_state(
        self,
        obs: Dict[str, np.ndarray],
        pose,
        sim_state: Dict[str, np.ndarray] | None,
        image: np.ndarray,
    ):
        self.states.append(
            self._make_state(
                step_idx=0,
                obs=obs,
                pose=pose,
                sim_state=sim_state,
                image=image,
            )
        )

    def record_transition(
        self,
        step_idx: int,
        next_obs: Dict[str, np.ndarray],
        reward: float,
        action: np.ndarray,
        pose,
        sim_state: Dict[str, np.ndarray] | None,
        image: np.ndarray,
        task: str = "",
    ):
        self.states.append(
            self._make_state(
                step_idx=step_idx + 1,
                obs=next_obs,
                pose=pose,
                sim_state=sim_state,
                image=image,
            )
        )
        self.transitions.append(
            {
                "step": step_idx,
                "task": task,
                "action": action.copy(),  # the action indexes are all wrong by one too much, maybe put in separate list
                "reward": float(reward),
                "dense_trajectory": self.dense_trajectory_buffer.copy(),
            }
        )
        self.dense_trajectory_buffer.clear()

    def _make_state(self, step_idx, obs, pose, sim_state, image):
        return {
            "step": step_idx,
            "image_path": self._save_image(step_idx, image),
            "joint_positions": obs["joint_positions"].copy(),
            "privileged_end_effector_pose": pose.as_10d(),
            "sim_state": sim_state,
        }

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

        # TODO(lerobot): Review this state/transition layout against LeRobot's dataset schema.

        data_dict = {
            "state_step": np.array([s["step"] for s in self.states], dtype=np.int32),
            "task": np.array([t["task"] for t in self.transitions], dtype=str),
            "image_path": np.array(
                [s["image_path"] for s in self.states], dtype=str
            ),
            "privileged_end_effector_pose": np.array(
                [s["privileged_end_effector_pose"] for s in self.states],
                dtype=np.float32,
            ),
            "joint_positions": np.array(
                [s["joint_positions"] for s in self.states],
                dtype=np.float32,
            ),
            "high_level_delta_action": np.array(
                [t["action"] for t in self.transitions], dtype=object
            ),
            "reward": np.array(
                [t["reward"] for t in self.transitions], dtype=np.float32
            ),
            "dense_trajectory": np.array(
                [t["dense_trajectory"] for t in self.transitions], dtype=object
            ),
        }

        if self.record_sim_state:
            data_dict["qpos"] = np.array(
                [s["sim_state"]["qpos"] for s in self.states], dtype=np.float32
            )
            data_dict["qvel"] = np.array(
                [s["sim_state"]["qvel"] for s in self.states], dtype=np.float32
            )

        if self.waypoints:
            data_dict["waypoints"] = self.waypoints

        np.savez_compressed(episode_path, **data_dict)
        print(f"Saved to: {episode_path}")
