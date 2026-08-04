import os
import glob
from stable_baselines3 import SAC
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import numpy as np

from robot_arm.pose import Pose


def make_waypoint_pose(
    position: np.ndarray,
    angles: np.ndarray,
    gripper: float,
    seq: str,
    degrees: bool,
) -> Pose:
    euler_pose = Pose.from_euler(position, angles, gripper, seq, degrees)
    rotation_matrix = euler_pose.rotation.as_matrix()
    return Pose.from_tcp_axes(
        position,
        rotation_matrix[:, 1],
        rotation_matrix[:, 2],
        gripper,
    )


class Policy(ABC):
    """
    Base interface for all high-level control policies.
    """

    @abstractmethod
    def get_action(
        self,
        obs: Dict[str, np.ndarray],
        info: Dict[str, Any],
        task: Optional[str] = None,
    ) -> np.ndarray:
        """
        Given the current observation, privileged info, and optional language task, output an action.
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
        task: Optional[str] = None,
    ) -> np.ndarray:
        import torch

        # Convert HWC image to CHW directly natively.
        img_chw = np.transpose(info["image"].astype(np.float32), (2, 0, 1))

        # Build raw unbatched transition data expected by lerobot pipelines
        raw_obs = {
            "observation.state": obs["joint_positions"].astype(np.float32),
            "observation.images.camera1": img_chw,
        }
        if task is not None:
            raw_obs["task"] = task

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

            # Preprocess (tokenizes task, normalizes images/state, sends to device)
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
        low_level_hz: int,
        position_speed_meters_per_second: float,
        rotation_speed_radians_per_second: float,
        gripper_speed_units_per_second: float,
    ):
        self.chunk_size = trajectory_length
        self.max_step_delta = np.array(
            [
                position_speed_meters_per_second / low_level_hz,
                position_speed_meters_per_second / low_level_hz,
                position_speed_meters_per_second / low_level_hz,
                rotation_speed_radians_per_second / low_level_hz,
                rotation_speed_radians_per_second / low_level_hz,
                rotation_speed_radians_per_second / low_level_hz,
                gripper_speed_units_per_second / low_level_hz,
            ],
            dtype=np.float32,
        )
        self.waypoints = []
        self.current_wp_idx = 0

    def generate_grab_waypoints(
        self,
        box_pose: Pose,
        lift_height: float,
        gripper_open: float,
        gripper_closed: float,
    ) -> None:
        """
        Generates a generic 4-waypoint sequence for grasping a given 6D pose:
        0. Move to a neutral, safe position to orient the arm gracefully.
        1. Move to the target with the gripper completely open.
        2. Close the gripper while remaining in place.
        3. Move straight up along the Z-axis with the gripper closed.
        """
        pregrasp_position = [
            np.random.uniform(0.3, 0.5),
            np.random.uniform(-0.15, 0.15),
            np.random.uniform(0.2, 0.4),
        ]
        pregrasp_x_rotation = np.random.uniform(-np.pi / 4, np.pi / 4)
        wp0 = make_waypoint_pose(
            pregrasp_position,
            [pregrasp_x_rotation, 0.0, 0.0],
            gripper_open,
            "XYZ",
            False,
        ).as_10d()

        wp1 = make_waypoint_pose(
            box_pose.position,
            [0.0, 0.0, 0.0],
            gripper_open,
            "XYZ",
            False,
        ).as_10d()

        wp2 = wp1.copy()
        wp2[9] = gripper_closed

        lift_x_rotation = np.random.choice([-np.pi / 2, np.pi / 2])
        wp3 = make_waypoint_pose(
            [wp2[0], wp2[1], wp2[2] + lift_height],
            [lift_x_rotation, 0.0, 0.0],
            gripper_closed,
            "XYZ",
            False,
        ).as_10d()

        self.waypoints = [wp0, wp1, wp2, wp3]
        self.current_wp_idx = 0

    def get_action(
        self,
        obs: Dict[str, np.ndarray],
        privileged_end_effector_pose: Pose,
        task: Optional[str] = None,
    ) -> np.ndarray:
        current_flat = privileged_end_effector_pose.as_10d()

        chunk = np.zeros((self.chunk_size, 7), dtype=np.float32)

        # If we exhausted waypoints, just stay where we are (zero deltas)
        if self.current_wp_idx >= len(self.waypoints):
            print("WARNING waypoints exhausted")
            return chunk

        target = Pose.from_10d(self.waypoints[self.current_wp_idx])
        target_rotation_delta = (
            privileged_end_effector_pose.rotation.inv() * target.rotation
        ).as_rotvec()
        diff = np.concatenate(
            [
                target.position - privileged_end_effector_pose.position,
                target_rotation_delta,
                [target.gripper - privileged_end_effector_pose.gripper],
            ]
        )
        max_chunk_delta = self.max_step_delta * self.chunk_size
        reaches_waypoint = np.all(np.abs(diff) <= max_chunk_delta)

        if reaches_waypoint:
            step_vector = diff / self.chunk_size
            self.current_wp_idx += 1
        else:
            step_vector = np.clip(
                diff,
                -self.max_step_delta,
                self.max_step_delta,
            )

        # Build the chunk as cumulative deltas pushing outward from the current pose
        for i in range(self.chunk_size):
            chunk[i] = np.minimum(
                np.abs(diff), np.abs(step_vector) * (i + 1)
            ) * np.sign(diff)

        return chunk


def find_latest_low_level_checkpoint() -> str:
    """
    Loads the most recent low-level SAC policy from the outputs/ directory.
    Searches the directory structure for the newest final checkpoint.
    """
    outputs_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..", "outputs")
    )
    if not os.path.exists(outputs_dir):
        raise FileNotFoundError(f"Outputs directory not found at {outputs_dir}.")

    # Search for all "sac_manual_step_final_*.zip" checkpoints inside checkpoints directories
    search_pattern = os.path.join(
        outputs_dir,
        "train_low_level",
        "*",
        "*",
        "checkpoints",
        "sac_manual_step_final_*.zip",
    )
    checkpoints = glob.glob(search_pattern)

    if not checkpoints:
        raise FileNotFoundError(
            "No final low-level policy checkpoints found in any outputs directory."
        )

    # Sort by the YYYY-MM-DD and HH-MM-SS folder names implicitly found in the path
    # Path structure: .../outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/sac...zip
    def extract_datetime_key(filepath):
        parts = filepath.split(os.sep)
        return (parts[-4], parts[-3])

    latest_checkpoint = max(checkpoints, key=extract_datetime_key)

    return latest_checkpoint


def load_low_level_policy(checkpoint_path: str):
    print(f"Loading low level policy from: {checkpoint_path}")
    return SAC.load(checkpoint_path)


def load_latest_low_level_policy():
    return load_low_level_policy(find_latest_low_level_checkpoint())
