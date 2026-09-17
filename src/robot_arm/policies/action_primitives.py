import mujoco
import numpy as np
from omegaconf import DictConfig

from robot_arm.arms.sim_arm import object_color
from robot_arm.control_types import ActionPrimitive
from robot_arm.geometry.pose import Pose
from robot_arm.robot_schema import TILE_BODY_NAME
from robot_arm.geometry.waypoints import generate_oriented_waypoint, position_from_base_rotation, shoulder_pan_position


def _find_body_position(model, data, body_name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    assert body_id != -1, f"Body {body_name!r} not found in MuJoCo model."
    return data.xpos[body_id].copy()


def _generate_random_position(model, data, random_pose_cfg) -> np.ndarray:
    pivot = shoulder_pan_position(model, data)
    height = float(np.random.uniform(*random_pose_cfg.height_meters))
    vertical_distance = height - pivot[2]
    min_distance, max_distance = random_pose_cfg.shoulder_distance_meters
    assert abs(vertical_distance) <= max_distance

    shoulder_distance = float(np.random.uniform(max(min_distance, abs(vertical_distance)), max_distance))
    planar_radius = np.sqrt(shoulder_distance**2 - vertical_distance**2)
    return position_from_base_rotation(
        model=model,
        data=data,
        radius=planar_radius,
        base_rotation_degrees=float(np.random.uniform(*random_pose_cfg.base_rotation_degrees)),
        height=height,
    )


def _sample_displacement_centimeters(value_range) -> int:
    low, high = round(value_range[0] * 100), round(value_range[1] * 100)
    return int(np.random.randint(low, high + 1))


def _is_reachable_position(model, data, cfg: DictConfig, position: np.ndarray) -> bool:
    min_height, max_height = cfg.waypoint.random_pose.height_meters
    _, max_distance = cfg.waypoint.random_pose.shoulder_distance_meters
    shoulder_distance = float(np.linalg.norm(position - shoulder_pan_position(model, data)))
    return min_height <= position[2] <= max_height and shoulder_distance <= max_distance


def _relative_move_prompt(x_cm: int, y_cm: int, z_cm: int) -> str:
    axes = (("x", x_cm), ("y", y_cm), ("z", z_cm))
    parts = [f"{centimeters}cm along {axis}" for axis, centimeters in axes if centimeters != 0]
    if not parts:
        return "hold position"
    return f"move {' and '.join(parts)}"


def relative_move_primitive(
    model,
    data,
    cfg: DictConfig,
    start_pose: Pose,
) -> ActionPrimitive:
    rel_cfg = cfg.waypoint.relative_move
    x_cm = _sample_displacement_centimeters(rel_cfg.dx_range_meters)
    y_cm = _sample_displacement_centimeters(rel_cfg.dy_range_meters)
    z_cm = _sample_displacement_centimeters(rel_cfg.dz_range_meters)

    # Shortened by whole centimetres rather than clipped, so the prompt states the commanded offset exactly.
    target_pos = start_pose.position + np.array([x_cm, y_cm, z_cm], dtype=np.float32) / 100.0
    while not _is_reachable_position(model, data, cfg, target_pos) and (x_cm or y_cm or z_cm):
        x_cm -= int(np.sign(x_cm))
        y_cm -= int(np.sign(y_cm))
        z_cm -= int(np.sign(z_cm))
        target_pos = start_pose.position + np.array([x_cm, y_cm, z_cm], dtype=np.float32) / 100.0

    target_pose = Pose.from_tcp_axes(
        position=target_pos,
        closing_axis=start_pose.closing_axis,
        secondary_axis=start_pose.secondary_axis,
        gripper=start_pose.gripper,
    )

    primitive = ActionPrimitive(
        start_pose=start_pose,
        target_pose=target_pose,
        prompt=_relative_move_prompt(x_cm, y_cm, z_cm),
        include_target_offset=True,
        desired_gripper_duty=0.0,
        desired_gripper_duty_active=False,
    )

    return primitive


def random_waypoint_primitive(
    model,
    data,
    cfg: DictConfig,
    start_pose: Pose,
) -> ActionPrimitive:
    random_pose_cfg = cfg.waypoint.random_pose
    target_position = _generate_random_position(model, data, random_pose_cfg)
    target_pose = generate_oriented_waypoint(
        model=model,
        data=data,
        position=target_position,
        pointing_axis_tilt_degrees=float(np.random.uniform(*random_pose_cfg.pointing_axis_tilt_degrees)),
        pointing_axis_rotation_degrees=float(np.random.uniform(*random_pose_cfg.pointing_axis_rotation_degrees)),
        gripper=float(np.random.uniform(*random_pose_cfg.gripper_radians)),
    )
    return ActionPrimitive(
        start_pose=start_pose,
        target_pose=target_pose,
        prompt="move according to the provided target delta",
        include_target_offset=True,
        desired_gripper_duty=0.0,
        desired_gripper_duty_active=False,
    )


class PickAndPlacePrimitives:
    def __init__(self, model, data, cfg: DictConfig, box_body_name: str):
        self.model = model
        self.data = data
        self.cfg = cfg.waypoint.pick_and_place
        self.box_color = object_color(model, box_body_name)
        self.tile_color = object_color(model, TILE_BODY_NAME)
        self.box_position = _find_body_position(model, data, box_body_name)
        self.tile_position = _find_body_position(model, data, TILE_BODY_NAME)
        self.grasp_height = float(self.box_position[2] + self.cfg.grasp_z_offset_meters)
        self.lift_height = float(self.box_position[2] + self.cfg.lift_z_offset_meters)

    def _target_pose(self, position: np.ndarray, gripper: float) -> Pose:
        return generate_oriented_waypoint(
            model=self.model,
            data=self.data,
            position=position,
            pointing_axis_tilt_degrees=45.0,
            pointing_axis_rotation_degrees=90.0,
            gripper=gripper,
        )

    def _primitive(self, start_pose: Pose, target_pose: Pose, prompt: str, gripping: bool) -> ActionPrimitive:
        return ActionPrimitive(
            start_pose=start_pose,
            target_pose=target_pose,
            prompt=prompt,
            include_target_offset=False,
            desired_gripper_duty=float(self.cfg.desired_gripper_duty) if gripping else 0.0,
            desired_gripper_duty_active=gripping,
        )

    def open_gripper(self, start_pose: Pose) -> ActionPrimitive:
        target = Pose.from_tcp_axes(
            position=start_pose.position,
            closing_axis=start_pose.closing_axis,
            secondary_axis=start_pose.secondary_axis,
            gripper=float(self.cfg.gripper_open_radians),
        )
        return self._primitive(start_pose, target, "open gripper", False)

    def _box_approach_pose(self) -> Pose:
        position = np.array([*self.box_position[:2], self.grasp_height], dtype=np.float32)
        return self._target_pose(position, float(self.cfg.gripper_open_radians))

    def move_above_box(self, start_pose: Pose) -> ActionPrimitive:
        approach = self._box_approach_pose()
        target = Pose.from_tcp_axes(
            position=np.array([*self.box_position[:2], self.lift_height], dtype=np.float32),
            closing_axis=approach.closing_axis,
            secondary_axis=approach.secondary_axis,
            gripper=float(self.cfg.gripper_open_radians),
        )
        return self._primitive(start_pose, target, f"move above {self.box_color} box", False)

    def move_to_box(self, start_pose: Pose) -> ActionPrimitive:
        return self._primitive(start_pose, self._box_approach_pose(), f"move to {self.box_color} box", False)

    def close_gripper(self, start_pose: Pose) -> ActionPrimitive:
        target = self._target_pose(start_pose.position, float(self.cfg.gripper_closed_radians))
        return self._primitive(start_pose, target, "close gripper", True)

    def lift_object(self, start_pose: Pose) -> ActionPrimitive:
        position = np.array([*self.box_position[:2], self.lift_height], dtype=np.float32)
        target = self._target_pose(position, float(self.cfg.gripper_closed_radians))
        return self._primitive(start_pose, target, "lift object", True)

    def move_above_tile(self, start_pose: Pose) -> ActionPrimitive:
        position = np.array([*self.tile_position[:2], self.lift_height], dtype=np.float32)
        target = self._target_pose(position, float(self.cfg.gripper_closed_radians))
        return self._primitive(start_pose, target, f"move {self.box_color} box above the {self.tile_color} tile", True)

    def lower_object(self, start_pose: Pose) -> ActionPrimitive:
        position = np.array([*self.tile_position[:2], self.grasp_height], dtype=np.float32)
        target = self._target_pose(position, float(self.cfg.gripper_closed_radians))
        return self._primitive(start_pose, target, f"lower {self.box_color} box onto the {self.tile_color} tile", True)

    def release_object(self, start_pose: Pose) -> ActionPrimitive:
        target = self._target_pose(start_pose.position, float(self.cfg.gripper_open_radians))
        return self._primitive(start_pose, target, "open gripper", False)
