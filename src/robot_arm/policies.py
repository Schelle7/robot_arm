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
        img_chw = np.transpose(info["image"].astype(np.float32), (2, 0, 1))

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
    A policy that follows pre-defined waypoints to grab a target box.
    It moves towards the current waypoint at a fixed speed per step.
    Once a waypoint is reached, it automatically targets the next one.
    """

    def __init__(
        self,
        trajectory_length: int,
        speed: float,
    ):
        self.chunk_size = trajectory_length
        self.speed = speed
        self.waypoints = []
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

def load_latest_low_level_policy():
    """
    Loads the most recent low-level SAC policy from the outputs/ directory.
    Searches the directory structure for the newest final checkpoint.
    """
    # TODO I might have to check the cfgs match in the relevant fields
    import os
    import glob
    from stable_baselines3 import SAC
    
    outputs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "outputs"))
    if not os.path.exists(outputs_dir):
        raise FileNotFoundError(f"Outputs directory not found at {outputs_dir}.")

    # Search for all "sac_manual_step_final_*.zip" checkpoints inside checkpoints directories
    search_pattern = os.path.join(outputs_dir, "*", "*", "checkpoints", "sac_manual_step_final_*.zip")
    checkpoints = glob.glob(search_pattern)

    if not checkpoints:
        raise FileNotFoundError("No final low-level policy checkpoints found in any outputs directory.")

    # Sort by the YYYY-MM-DD and HH-MM-SS folder names implicitly found in the path
    # Path structure: .../outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/sac...zip
    def extract_datetime_key(filepath):
        parts = filepath.split(os.sep)
        return (parts[-4], parts[-3])

    latest_checkpoint = max(checkpoints, key=extract_datetime_key)
    
    print(f"Loading latest low level policy from: {latest_checkpoint}")
    return SAC.load(latest_checkpoint)
