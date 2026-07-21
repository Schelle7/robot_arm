from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import numpy as np


class Policy(ABC):
    """
    Base interface for all high-level control policies.
    """

    @abstractmethod
    def get_action(
        self,
        obs: Dict[str, np.ndarray],
        info: Dict[str, Any],
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """
        Given the current observation, privileged info, and optional language instruction, output an action.
        """
        pass


class ReplayPolicy(Policy):
    """
    A simple policy that blindly plays back a pre-generated sequence of actions.
    Useful for testing, hardcoded trajectories, or replaying offline data.
    """

    def __init__(self, start_pos: np.ndarray, end_pos: np.ndarray, num_steps: int):
        self.trajectory = np.linspace(start_pos, end_pos, num_steps)
        self.current_step = 0

    def get_action(
        self,
        obs: Dict[str, np.ndarray],
        info: Dict[str, Any],
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        action = self.trajectory[self.current_step]
        self.current_step += 1
        return action


class SmolVLAPolicyWrapper(Policy):
    """
    Wrapper for the lerobot SmolVLAPolicy.
    """

    def __init__(self, model_id: str = "lerobot/smolvla_base"):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = SmolVLAPolicy.from_pretrained(model_id).to(self.device)
        self.policy.eval()

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=model_id,
        )

    def get_action(
        self,
        obs: Dict[str, np.ndarray],
        info: Dict[str, Any],
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        import torch

        # Convert HWC image to CHW directly natively.
        img_chw = np.transpose(obs["pixels"].astype(np.float32), (2, 0, 1))

        # Build raw unbatched transition data expected by lerobot pipelines
        raw_obs = {
            "observation.state": obs["joint_positions"].astype(np.float32),
            "observation.images.camera1": img_chw,
        }
        if instruction is not None:
            raw_obs["task"] = instruction

        with torch.inference_mode():
            # Add batch dim manually as preprocessor expects it
            batch = {
                k: (
                    torch.tensor(v).unsqueeze(0).to(self.device)
                    if isinstance(v, np.ndarray)
                    else [v]
                )
                for k, v in raw_obs.items()
            }

            # Preprocess (tokenizes instruction, normalizes images/state, sends to device)
            processed_batch = self.preprocessor(batch)

            # Action selection
            out = self.policy.select_action(processed_batch)

            # Postprocess (unnormalizes action back to radians/raw units)
            action = self.postprocessor(out)

        return action.squeeze(0).cpu().numpy()


class WaypointPolicy(Policy):
    """
    A policy that takes a sequence of 7D waypoints and generates chunks of deltas.
    It moves towards the current waypoint at a fixed speed per step.
    Once a waypoint is reached, it automatically targets the next one.
    """

    def __init__(self, waypoints: list[np.ndarray], chunk_size: int, speed: float):
        """
        waypoints: List of 7D target poses [x, y, z, roll, pitch, yaw, gripper]
        chunk_size: Number of future steps to project in each chunk
        speed: Maximum 7D distance to move per chunk step
        """
        self.waypoints = [np.array(wp, dtype=np.float32) for wp in waypoints]
        self.chunk_size = chunk_size
        self.speed = speed
        self.current_wp_idx = 0

    def get_action(
        self,
        obs: Dict[str, np.ndarray],
        info: Dict[str, Any],
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        # We need the 7D pose from the privileged info dict
        current_pose = info["privileged_end_effector_pose_7d"]
        chunk = np.zeros((self.chunk_size, 7), dtype=np.float32)

        # If we exhausted waypoints, just stay where we are (zero deltas)
        if self.current_wp_idx >= len(self.waypoints):
            return chunk  # TODO I might wan to make this an error instead

        target = self.waypoints[self.current_wp_idx]
        diff = target - current_pose
        dist = np.linalg.norm(diff)

        # Determine the speed of this entire chunk
        # Max distance this chunk can cover at nominal speed
        max_chunk_dist = self.speed * self.chunk_size

        if dist <= max_chunk_dist:
            # Close enough to finish in this chunk.
            # Scale the per-step velocity so the final step lands exactly on the waypoint.
            step_vector = diff / self.chunk_size
            self.current_wp_idx += 1
        else:
            # Far away. Go straight toward the waypoint at full speed.
            step_vector = diff * (self.speed / dist)

        # Build the chunk as cumulative deltas pushing outward from the current pose
        for i in range(self.chunk_size):
            chunk[i] = step_vector * (i + 1)

        return chunk
