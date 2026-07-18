from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class Policy(ABC):
    """
    Base interface for all high-level control policies.
    """
    @abstractmethod
    def get_action(self, obs: Dict[str, np.ndarray], instruction: Optional[str] = None) -> np.ndarray:
        """
        Given the current observation and optional language instruction, output an action.
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

    def get_action(self, obs: Dict[str, np.ndarray], instruction: Optional[str] = None) -> np.ndarray:
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

    def get_action(self, obs: Dict[str, np.ndarray], instruction: Optional[str] = None) -> np.ndarray:
        import torch
        
        # Convert HWC image to CHW directly natively.
        img_chw = np.transpose(obs["pixels"].astype(np.float32), (2, 0, 1))
            
        # Build raw unbatched transition data expected by lerobot pipelines
        raw_obs = {
            "observation.state": obs["agent_pos"].astype(np.float32),
            "observation.images.camera1": img_chw
        }
        if instruction is not None:
            raw_obs["task"] = instruction

        with torch.inference_mode():
            # Add batch dim manually as preprocessor expects it
            batch = {k: torch.tensor(v).unsqueeze(0).to(self.device) if isinstance(v, np.ndarray) else [v] for k, v in raw_obs.items()}
            
            # Preprocess (tokenizes instruction, normalizes images/state, sends to device)
            processed_batch = self.preprocessor(batch)
            
            # Action selection
            out = self.policy.select_action(processed_batch)
            
            # Postprocess (unnormalizes action back to radians/raw units)
            action = self.postprocessor(out)
            
        return action.squeeze(0).cpu().numpy()
