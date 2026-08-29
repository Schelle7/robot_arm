from dataclasses import dataclass
from typing import List
import mujoco
import numpy as np
from omegaconf import DictConfig

from robot_arm.pose import Pose
from robot_arm.experimental_waypoints import (
    generate_oriented_waypoint,
    position_from_base_rotation,
    shoulder_pan_position,
)


@dataclass
class ActionPrimitive:
    start_pose: Pose
    target_pose: Pose
    prompt: str
    has_explicit_goal: bool


def _find_target_box_position(model, data) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_box")
    assert body_id != -1, "Target box body 'target_box' not found in MuJoCo model."
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


def generate_pick_and_place(
    model,
    data,
    cfg: DictConfig,
    start_pose: Pose,
) -> List[ActionPrimitive]:
    box_pos = _find_target_box_position(model, data)
    pnp_cfg = cfg.waypoint.pick_and_place

    grasp_height = float(box_pos[2] + pnp_cfg.grasp_z_offset_meters)
    lift_height = float(box_pos[2] + pnp_cfg.lift_z_offset_meters)

    # 1. Move to box with open gripper
    approach_pose = generate_oriented_waypoint(
        model=model,
        data=data,
        position=np.array([box_pos[0], box_pos[1], grasp_height], dtype=np.float32),
        pointing_axis_tilt_degrees=0.0,
        pointing_axis_rotation_degrees=0.0,
        gripper=float(pnp_cfg.gripper_open_radians),
    )
    p1 = ActionPrimitive(
        start_pose=start_pose,
        target_pose=approach_pose,
        prompt="move to red box and open gripper",
        has_explicit_goal=False,
    )

    # 2. Close gripper and lift box
    # TODO: Potentially also sample a feasible lift orientation while keeping the gripper closed.
    lift_pose = generate_oriented_waypoint(
        model=model,
        data=data,
        position=np.array([box_pos[0], box_pos[1], lift_height], dtype=np.float32),
        pointing_axis_tilt_degrees=0.0,
        pointing_axis_rotation_degrees=0.0,
        gripper=float(pnp_cfg.gripper_closed_radians),
    )
    p2 = ActionPrimitive(
        start_pose=approach_pose,
        target_pose=lift_pose,
        prompt="close gripper and lift red box",
        has_explicit_goal=True,
    )

    # 3. Transport to target location
    random_pose_cfg = cfg.waypoint.random_pose
    target_tilt = float(np.random.uniform(*random_pose_cfg.pointing_axis_tilt_degrees))
    target_rotation = float(np.random.uniform(*random_pose_cfg.pointing_axis_rotation_degrees))
    target_place_pos = _generate_random_position(model, data, random_pose_cfg)
    transport_pose = generate_oriented_waypoint(
        model=model,
        data=data,
        position=target_place_pos,
        pointing_axis_tilt_degrees=target_tilt,
        pointing_axis_rotation_degrees=target_rotation,
        gripper=float(pnp_cfg.gripper_closed_radians),
    )
    p3 = ActionPrimitive(
        start_pose=lift_pose,
        target_pose=transport_pose,
        prompt="move red box to target location",
        has_explicit_goal=True,
    )

    # 4. Lower and release
    place_down_pos = np.array([target_place_pos[0], target_place_pos[1], grasp_height], dtype=np.float32)
    place_pose = generate_oriented_waypoint(
        model=model,
        data=data,
        position=place_down_pos,
        pointing_axis_tilt_degrees=0.0,
        pointing_axis_rotation_degrees=0.0,
        gripper=float(pnp_cfg.gripper_open_radians),
    )
    p4 = ActionPrimitive(
        start_pose=transport_pose,
        target_pose=place_pose,
        prompt="place red box on target",
        has_explicit_goal=False,
    )

    return [p1, p2, p3, p4]


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


def generate_relative_moves(
    model,
    data,
    cfg: DictConfig,
    start_pose: Pose,
) -> List[ActionPrimitive]:
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
        has_explicit_goal=True,
    )

    return [primitive]


def generate_random_waypoint(
    model,
    data,
    cfg: DictConfig,
    start_pose: Pose,
) -> List[ActionPrimitive]:
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
    return [
        ActionPrimitive(
            start_pose=start_pose,
            target_pose=target_pose,
            prompt="move according to the provided target delta",
            has_explicit_goal=True,
        )
    ]


def generate_action_primitives(
    model,
    data,
    cfg: DictConfig,
    start_pose: Pose,
) -> List[ActionPrimitive]:
    probabilities = cfg.waypoint.primitive_probabilities
    task_probabilities = np.array(
        [
            probabilities.pick_and_place,
            probabilities.relative_move,
            probabilities.random_waypoint,
        ],
        dtype=np.float64,
    )
    assert np.all(task_probabilities >= 0.0)
    assert np.isclose(task_probabilities.sum(), 1.0)

    task = np.random.choice(
        ["pick_and_place", "relative_move", "random_waypoint"],
        p=task_probabilities,
    )
    if task == "pick_and_place":
        return generate_pick_and_place(model, data, cfg, start_pose)
    elif task == "relative_move":
        return generate_relative_moves(model, data, cfg, start_pose)
    elif task == "random_waypoint":
        return generate_random_waypoint(model, data, cfg, start_pose)
    raise ValueError(f"Unknown primitive task: {task}")
