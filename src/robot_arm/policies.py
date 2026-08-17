import os
import glob
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from stable_baselines3 import SAC
from typing import Dict, Any
import numpy as np
from omegaconf import DictConfig

from robot_arm.pose import Pose, axis_angular_distance
from robot_arm.robot_schema import CARTESIAN_ACTION_NAMES


def latest_vla_checkpoint_path() -> str:
    latest_run_file = Path(__file__).resolve().parents[2] / "outputs" / "train_vla" / "latest_run.txt"
    training_output_dir = Path(latest_run_file.read_text().strip())
    checkpoint_path = training_output_dir / "checkpoints" / "last" / "pretrained_model"
    return str(checkpoint_path)


def waypoint_action_limits(
    trajectory_length: int,
    low_level_hz: int,
    position_speed_meters_per_second: float,
    rotation_speed_radians_per_second: float,
    gripper_speed_radians_per_second: float,
) -> tuple[float, float, float]:
    return (
        position_speed_meters_per_second * trajectory_length / low_level_hz,
        rotation_speed_radians_per_second * trajectory_length / low_level_hz,
        gripper_speed_radians_per_second * trajectory_length / low_level_hz,
    )


def waypoint_action_scale(
    trajectory_length: int,
    low_level_hz: int,
    position_speed_meters_per_second: float,
    rotation_speed_radians_per_second: float,
    gripper_speed_radians_per_second: float,
) -> np.ndarray:
    position_limit, rotation_limit, gripper_limit = waypoint_action_limits(
        trajectory_length,
        low_level_hz,
        position_speed_meters_per_second,
        rotation_speed_radians_per_second,
        gripper_speed_radians_per_second,
    )
    return np.array(
        [
            position_limit,
            position_limit,
            position_limit,
            rotation_limit,
            rotation_limit,
            rotation_limit,
            gripper_limit,
        ],
        dtype=np.float32,
    )


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


@dataclass
class CartesianAction:
    cartesian_action_path: np.ndarray
    diagnostics: Dict[str, Any]
    completes_active_primitive: bool


class CartesianPolicy(ABC):
    @abstractmethod
    def get_action(
        self,
        current_pose: Pose,
        image: np.ndarray,
        vla_input_state: np.ndarray,
        primitive_prompt: str,
        privileged_target_pose: Pose | None,
    ) -> CartesianAction:
        raise NotImplementedError


class VLACartesianPolicy(CartesianPolicy):
    """
    Wrapper for the lerobot SmolVLAPolicy.

    TODO: Define how VLA orientation actions should be evaluated with primary and secondary axis angular distances.
    the problem is that not every pose can be reached
    """

    def __init__(self, model_path: str):
        from robot_arm.cartesian_smolvla.modeling_cartesian_smolvla import CartesianSmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = CartesianSmolVLAPolicy.from_pretrained(model_path, strict=True).to(self.device)
        self.policy.eval()

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=model_path,
        )

    def get_action(
        self,
        current_pose: Pose,
        image: np.ndarray,
        vla_input_state: np.ndarray,
        primitive_prompt: str,
        privileged_target_pose: Pose | None,
    ) -> CartesianAction:
        import torch

        img_chw = np.transpose(image.astype(np.float32), (2, 0, 1))

        raw_obs = {
            "observation.state": vla_input_state.astype(np.float32),
            "observation.images.camera1": img_chw,
            "task": primitive_prompt,
        }

        with torch.inference_mode():
            batch = {k: (torch.tensor(v).unsqueeze(0).to(self.device) if isinstance(v, np.ndarray) else [v]) for k, v in raw_obs.items()}
            processed_batch = self.preprocessor(batch)
            out, completion_probability = self.policy.select_action_with_completion(processed_batch)
            action = self.postprocessor(out)

        return CartesianAction(
            cartesian_action_path=action.squeeze(0).cpu().numpy().reshape(-1, len(CARTESIAN_ACTION_NAMES)),
            diagnostics={"completion_probability": float(completion_probability.item())},
            completes_active_primitive=bool(completion_probability.item() >= 0.5),
        )


class ScriptedCartesianPolicy(CartesianPolicy):
    """
    Produces Cartesian action paths toward privileged primitive targets.
    """

    def __init__(self, cfg: DictConfig):
        trajectory_length = cfg.waypoint.trajectory_length
        low_level_hz = cfg.control.frequencies.low_level
        position_speed_meters_per_second = cfg.waypoint.position_speed_meters_per_second
        rotation_speed_radians_per_second = cfg.waypoint.rotation_speed_radians_per_second
        gripper_speed_radians_per_second = cfg.waypoint.gripper_speed_radians_per_second

        self.action_horizon = trajectory_length
        self.max_position_delta = position_speed_meters_per_second / low_level_hz
        self.max_rotation_delta = rotation_speed_radians_per_second / low_level_hz
        self.max_gripper_delta = gripper_speed_radians_per_second / low_level_hz
        (
            self.max_position_delta_over_horizon,
            self.max_orientation_distance_over_horizon,
            self.max_gripper_delta_over_horizon,
        ) = waypoint_action_limits(
            trajectory_length,
            low_level_hz,
            position_speed_meters_per_second,
            rotation_speed_radians_per_second,
            gripper_speed_radians_per_second,
        )
    def _evaluate_target(self, current_pose: Pose, target_pose: Pose):
        waypoint_delta = current_pose.delta_to(target_pose)
        position_distance = float(np.linalg.norm(waypoint_delta[:3]))
        primary_orientation_distance = axis_angular_distance(
            current_pose.closing_axis,
            target_pose.closing_axis,
        )
        secondary_orientation_distance = axis_angular_distance(
            current_pose.secondary_axis,
            target_pose.secondary_axis,
        )
        gripper_distance = float(abs(waypoint_delta[6]))
        completes_active_primitive = (
            position_distance <= self.max_position_delta_over_horizon
            and primary_orientation_distance <= self.max_orientation_distance_over_horizon
            and secondary_orientation_distance <= self.max_orientation_distance_over_horizon
            and gripper_distance <= self.max_gripper_delta_over_horizon
        )

        diagnostics = {
            "position_distance": position_distance,
            "position_threshold": float(self.max_position_delta_over_horizon),
            "primary_orientation_distance": primary_orientation_distance,
            "secondary_orientation_distance": secondary_orientation_distance,
            "orientation_threshold": float(self.max_orientation_distance_over_horizon),
            "gripper_distance": gripper_distance,
            "gripper_threshold": float(self.max_gripper_delta_over_horizon),
        }
        return waypoint_delta, diagnostics, completes_active_primitive

    def get_action(
        self,
        current_pose: Pose,
        image: np.ndarray,
        vla_input_state: np.ndarray,
        primitive_prompt: str,
        privileged_target_pose: Pose | None,
    ) -> CartesianAction:
        action_sequence = np.zeros((self.action_horizon, 7), dtype=np.float32)

        assert privileged_target_pose is not None
        waypoint_delta, diagnostics, completes_active_primitive = self._evaluate_target(current_pose, privileged_target_pose)

        if completes_active_primitive:
            step_vector = waypoint_delta / self.action_horizon
        else:
            position_step = waypoint_delta[:3].copy()
            position_step_norm = np.linalg.norm(position_step)
            if position_step_norm > self.max_position_delta:
                position_step *= self.max_position_delta / position_step_norm

            step_vector = np.empty(7, dtype=np.float32)
            step_vector[:3] = position_step
            step_vector[3:6] = np.clip(waypoint_delta[3:6], -self.max_rotation_delta, self.max_rotation_delta)
            step_vector[6] = np.clip(waypoint_delta[6], -self.max_gripper_delta, self.max_gripper_delta)

        # Build the action sequence as cumulative deltas pushing outward from the current pose
        for i in range(self.action_horizon):
            action_sequence[i] = np.minimum(np.abs(waypoint_delta), np.abs(step_vector) * (i + 1)) * np.sign(waypoint_delta)

        return CartesianAction(
            cartesian_action_path=action_sequence,
            diagnostics=diagnostics,
            completes_active_primitive=completes_active_primitive,
        )


def find_latest_low_level_checkpoint() -> str:
    """
    Loads the most recent low-level SAC policy from the outputs/ directory.
    Searches the directory structure for the newest final checkpoint.
    """
    outputs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "outputs"))
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
        raise FileNotFoundError("No final low-level policy checkpoints found in any outputs directory.")

    # Sort by the YYYY-MM-DD and HH-MM-SS folder names implicitly found in the path
    # Path structure: .../outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/sac...zip
    def extract_datetime_key(filepath):
        parts = filepath.split(os.sep)
        return (parts[-4], parts[-3])

    latest_checkpoint = max(checkpoints, key=extract_datetime_key)

    return latest_checkpoint


def resolve_low_level_checkpoint(policy_name: str) -> str:
    if policy_name == "latest":
        return find_latest_low_level_checkpoint()

    policy_path = Path(policy_name)
    if not policy_path.is_absolute():
        policy_path = Path(__file__).resolve().parents[2] / policy_path
    return str(policy_path.resolve())


def load_low_level_policy(checkpoint_path: str):
    print(f"Loading low level policy from: {checkpoint_path}")
    return SAC.load(checkpoint_path)


def load_latest_low_level_policy():
    return load_low_level_policy(find_latest_low_level_checkpoint())
