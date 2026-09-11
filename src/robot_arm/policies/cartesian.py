from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import numpy as np
from omegaconf import DictConfig

from robot_arm.pose import Pose, axis_angular_distance
from robot_arm.action_primitives import ActionPrimitive
from robot_arm.robot_schema import CAMERA_NAMES, CARTESIAN_ACTION_NAMES


def latest_vla_checkpoint_path() -> str:
    latest_run_file = Path(__file__).resolve().parents[3] / "outputs" / "train_vla" / "latest_run.txt"
    training_output_dir = Path(latest_run_file.read_text().strip())
    checkpoint_path = training_output_dir / "checkpoints" / "last" / "pretrained_model"
    return str(checkpoint_path)


def waypoint_action_limits(
    cartesian_hz: int,
    position_speed_meters_per_second: float,
    rotation_speed_radians_per_second: float,
    gripper_speed_radians_per_second: float,
) -> tuple[float, float, float]:
    return (
        position_speed_meters_per_second / cartesian_hz,
        rotation_speed_radians_per_second / cartesian_hz,
        gripper_speed_radians_per_second / cartesian_hz,
    )


def waypoint_action_scale(
    cartesian_hz: int,
    position_speed_meters_per_second: float,
    rotation_speed_radians_per_second: float,
    gripper_speed_radians_per_second: float,
) -> np.ndarray:
    position_limit, rotation_limit, gripper_limit = waypoint_action_limits(
        cartesian_hz,
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
    cartesian_action: np.ndarray
    diagnostics: Dict[str, Any]
    completes_active_primitive: bool


class CartesianPolicy(ABC):
    @abstractmethod
    def get_action(
        self,
        current_pose: Pose,
        images: dict[str, np.ndarray],
        vla_input_state: np.ndarray,
        gripper_duty: float,
        grasp_confirmed: bool,
        primitive: ActionPrimitive,
    ) -> CartesianAction:
        raise NotImplementedError


def build_vla_observation(images: dict[str, np.ndarray], vla_input_state: np.ndarray, primitive_prompt: str) -> Dict[str, Any]:
    """
    Builds the observation the rollout hands the shared LeRobot preprocessor. It has to match what
    LeRobotDataset yields for a recorded frame, or inference silently disagrees with training.
    LeRobot serves images in [0, 1] and SmolVLA normalizes VISUAL features as identity, so nothing
    downstream rescales what is passed in here.
    """
    assert tuple(images) == CAMERA_NAMES, f"Expected cameras {CAMERA_NAMES}, got {tuple(images)}."
    external_camera_image = images["external_camera"]
    wrist_camera_image = images["wrist_camera"]
    assert external_camera_image.dtype == np.uint8, f"external_camera frames must be uint8, got {external_camera_image.dtype}."
    assert wrist_camera_image.dtype == np.uint8, f"wrist_camera frames must be uint8, got {wrist_camera_image.dtype}."
    return {
        "observation.state": vla_input_state.astype(np.float32),
        "observation.images.external_camera": np.transpose(external_camera_image.astype(np.float32) / 255.0, (2, 0, 1)),
        "observation.images.wrist_camera": np.transpose(wrist_camera_image.astype(np.float32) / 255.0, (2, 0, 1)),
        "task": primitive_prompt,
    }


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
        images: dict[str, np.ndarray],
        vla_input_state: np.ndarray,
        gripper_duty: float,
        grasp_confirmed: bool,
        primitive: ActionPrimitive,
    ) -> CartesianAction:
        import torch

        raw_obs = build_vla_observation(images, vla_input_state, primitive.prompt)

        with torch.inference_mode():
            batch = {k: (torch.tensor(v).unsqueeze(0).to(self.device) if isinstance(v, np.ndarray) else [v]) for k, v in raw_obs.items()}
            processed_batch = self.preprocessor(batch)
            out, completion_probability = self.policy.select_action_with_completion(processed_batch)
            action = self.postprocessor(out)

        return CartesianAction(
            cartesian_action=action.squeeze(0).cpu().numpy().reshape(len(CARTESIAN_ACTION_NAMES)),
            diagnostics={"completion_probability": float(completion_probability.item())},
            completes_active_primitive=bool(completion_probability.item() >= 0.5)
            and (primitive.prompt != "close gripper" or grasp_confirmed),
        )


class ScriptedCartesianPolicy(CartesianPolicy):
    """
    Produces Cartesian actions toward the primitive's target pose.
    """

    def __init__(self, cfg: DictConfig):
        cartesian_hz = cfg.control.frequencies.cartesian
        position_speed_meters_per_second = cfg.waypoint.position_speed_meters_per_second
        rotation_speed_radians_per_second = cfg.waypoint.rotation_speed_radians_per_second
        gripper_speed_radians_per_second = cfg.waypoint.gripper_speed_radians_per_second

        self.duty_completion_tolerance = float(cfg.waypoint.duty_completion_tolerance)

        tolerance = cfg.waypoint.completion_tolerance
        self.position_tolerance = float(tolerance.position_meters)
        self.primary_rotation_tolerance = float(tolerance.primary_rotation_radians)
        self.secondary_rotation_tolerance = float(tolerance.secondary_rotation_radians)
        self.gripper_tolerance = float(tolerance.gripper_radians)

        (
            self.max_position_delta,
            self.max_rotation_delta,
            self.max_gripper_delta,
        ) = waypoint_action_limits(
            cartesian_hz,
            position_speed_meters_per_second,
            rotation_speed_radians_per_second,
            gripper_speed_radians_per_second,
        )

    def _evaluate_target(self, current_pose: Pose, target_pose: Pose, gripper_duty: float, primitive: ActionPrimitive):
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
        gripper_duty_distance = abs(primitive.desired_gripper_duty - gripper_duty)

        if primitive.desired_gripper_duty_active:
            gripper_channel_reached = gripper_duty_distance <= self.duty_completion_tolerance
        else:
            gripper_channel_reached = gripper_distance <= self.gripper_tolerance

        completes_active_primitive = (
            position_distance <= self.position_tolerance
            and primary_orientation_distance <= self.primary_rotation_tolerance
            and secondary_orientation_distance <= self.secondary_rotation_tolerance
            and gripper_channel_reached
        )

        diagnostics = {
            "position_distance": position_distance,
            "position_threshold": float(self.position_tolerance),
            "primary_orientation_distance": primary_orientation_distance,
            "secondary_orientation_distance": secondary_orientation_distance,
            "primary_orientation_threshold": float(self.primary_rotation_tolerance),
            "secondary_orientation_threshold": float(self.secondary_rotation_tolerance),
            "gripper_distance": gripper_distance,
            "gripper_threshold": float(self.gripper_tolerance),
            "gripper_duty_distance": gripper_duty_distance,
            "duty_threshold": float(self.duty_completion_tolerance),
        }
        return waypoint_delta, diagnostics, completes_active_primitive

    def get_action(
        self,
        current_pose: Pose,
        images: dict[str, np.ndarray],
        vla_input_state: np.ndarray,
        gripper_duty: float,
        grasp_confirmed: bool,
        primitive: ActionPrimitive,
    ) -> CartesianAction:
        waypoint_delta, diagnostics, completes_active_primitive = self._evaluate_target(
            current_pose,
            primitive.target_pose,
            gripper_duty,
            primitive,
        )
        completes_active_primitive = completes_active_primitive and (
            primitive.prompt != "close gripper" or grasp_confirmed
        )

        # The completion tolerances are set independently of the speeds, so a completing step can
        # still ask for more than one command may travel. Shortening is therefore unconditional.
        # maybe therefore simplify the existing setup?
        position = waypoint_delta[:3]
        position_norm = np.linalg.norm(position)
        if position_norm > self.max_position_delta:
            position *= self.max_position_delta / position_norm

        rotation = waypoint_delta[3:6]
        rotation_norm = np.linalg.norm(rotation)
        if rotation_norm > self.max_rotation_delta:
            rotation *= self.max_rotation_delta / rotation_norm

        waypoint_delta[6] = np.clip(waypoint_delta[6], -self.max_gripper_delta, self.max_gripper_delta)

        return CartesianAction(
            cartesian_action=waypoint_delta,
            diagnostics=diagnostics,
            completes_active_primitive=completes_active_primitive,
        )
