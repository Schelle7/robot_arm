import os
import copy
import numpy as np
from typing import Dict, List, Any
from PIL import Image
from robot_arm.robot_schema import MOTOR_ORDER


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
        self.chunk_size = cfg.control.frequencies.low_level // cfg.control.frequencies.mid_level
        self.record_sim_state = cfg.runtime.record_sim_state
        self.record_policy_debug = cfg.runtime.record_policy_debug
        self.motor_order = MOTOR_ORDER
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

    def record_transition(
        self,
        state_idx: int,
        obs: Dict[str, np.ndarray],
        sensor_state: Dict[str, Any],
        reward: float,
        cartesian_action_path: np.ndarray,
        pose,
        sim_state: Dict[str, np.ndarray] | None,
        image: np.ndarray,
        vla_input_state: np.ndarray,
        primitive_prompt: str,
        primitive_index: int,
        diagnostics: Dict[str, Any],
        completes_active_primitive: bool,
    ):
        self.states.append(
            self._make_state(
                state_idx=state_idx,
                primitive_index=primitive_index,
                obs=obs,
                sensor_state=sensor_state,
                pose=pose,
                sim_state=sim_state,
                image=image,
            )
        )
        self.transitions.append(
            {
                "step": state_idx,
                "primitive_prompt": primitive_prompt,
                "vla_input_state": vla_input_state.copy(),
                "reward": float(reward),
                "cartesian_action_path": cartesian_action_path.copy(),
                "diagnostics": diagnostics.copy(),
                "completes_active_primitive": bool(completes_active_primitive),
                "dense_trajectory": self.dense_trajectory_buffer.copy(),
            }
        )
        self.dense_trajectory_buffer.clear()

    def record_final_state(
        self,
        state_idx: int,
        primitive_index: int,
        obs: Dict[str, np.ndarray],
        sensor_state: Dict[str, Any],
        pose,
        sim_state: Dict[str, np.ndarray] | None,
        image: np.ndarray,
    ):
        self.states.append(
            self._make_state(
                state_idx=state_idx,
                primitive_index=primitive_index,
                obs=obs,
                sensor_state=sensor_state,
                pose=pose,
                sim_state=sim_state,
                image=image,
            )
        )

    def _make_state(self, state_idx, primitive_index, obs, sensor_state, pose, sim_state, image):
        return {
            "step": state_idx,
            "primitive_index": primitive_index,
            "image_path": self._save_image(state_idx, image),
            "joint_positions": obs["joint_positions"].copy(),
            "joint_velocities": obs["joint_velocities"].copy(),
            "sensor_state": sensor_state,
            "privileged_end_effector_pose": pose.as_10d(),
            "sim_state": sim_state,
        }

    def append_low_level_transition(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        reward_breakdown: Dict[str, float],
        terminated: bool,
        state,
        next_state,
    ):
        """
        Record a single low-level RL transition step in the environment.
        Currently, this method just caches dense steps on the most recently added frame
        so that when we save, we can view the whole micro-trajectory that occurred during the high-level step.
        """
        if len(self.dense_trajectory_buffer) >= self.chunk_size:
            raise RuntimeError(f"Dense trajectory buffer overflow! Max chunk size is {self.chunk_size}.")

        self.dense_trajectory_buffer.append(
            {
                "obs": {k: v.copy() for k, v in obs.items() if isinstance(v, np.ndarray)},
                "next_obs": {k: v.copy() for k, v in next_obs.items() if isinstance(v, np.ndarray)},
                "action": action.copy(),
                "reward": float(reward),
                "reward_breakdown": {key: float(value) for key, value in reward_breakdown.items()},
                "terminated": terminated,
                "privileged_state": copy.deepcopy(state.privileged_state),
                "next_privileged_state": copy.deepcopy(next_state.privileged_state),
            }
        )

    def _save_image(self, state_idx: int, pixels: np.ndarray) -> str:
        """Saves the camera frame to disk and returns the relative path."""
        image_path = f"images/frame_{state_idx:04d}.jpg"
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
            "step": np.array([s["step"] for s in self.states], dtype=np.int32),
            "primitive_prompt": np.array([t["primitive_prompt"] for t in self.transitions], dtype=str),
            "primitive_index": np.array([s["primitive_index"] for s in self.states], dtype=np.int32),
            "image_path": np.array([s["image_path"] for s in self.states], dtype=str),
            "vla_input_state": np.array(
                [t["vla_input_state"] for t in self.transitions],
                dtype=np.float32,
            ),
            "privileged_end_effector_pose": np.array(
                [s["privileged_end_effector_pose"] for s in self.states],
                dtype=np.float32,
            ),
            "joint_positions": np.array(
                [s["joint_positions"] for s in self.states],
                dtype=np.float32,
            ),
            "joint_velocities": np.array(
                [s["joint_velocities"] for s in self.states],
                dtype=np.float32,
            ),
            "cartesian_action_path": np.array([t["cartesian_action_path"] for t in self.transitions], dtype=object),
            "cartesian_action_diagnostics": np.array([t["diagnostics"] for t in self.transitions], dtype=object),
            "completes_active_primitive": np.array([t["completes_active_primitive"] for t in self.transitions], dtype=bool),
            "reward": np.array([t["reward"] for t in self.transitions], dtype=np.float32),
            "dense_trajectory": np.array([t["dense_trajectory"] for t in self.transitions], dtype=object),
        }

        sensor_names = ("Present_Temperature", "Present_Load", "Present_Voltage")
        for sensor_name in sensor_names:
            data_dict[f"sensor_{sensor_name.removeprefix('Present_').lower()}"] = np.array(
                [[s["sensor_state"][sensor_name][motor] for motor in self.motor_order] for s in self.states],
                dtype=np.float32,
            )

        if self.record_sim_state:
            data_dict["qpos"] = np.array([s["sim_state"]["qpos"] for s in self.states], dtype=np.float32)
            data_dict["qvel"] = np.array([s["sim_state"]["qvel"] for s in self.states], dtype=np.float32)

        if self.waypoints:
            data_dict["waypoints"] = self.waypoints

        np.savez_compressed(episode_path, **data_dict)
        print(f"Saved to: {episode_path}")
