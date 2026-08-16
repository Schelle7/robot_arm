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


def generate_relative_moves(
    model,
    data,
    cfg: DictConfig,
    start_pose: Pose,
) -> List[ActionPrimitive]:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "pixel_cam")
    assert cam_id != -1, "Camera 'pixel_cam' not found in MuJoCo model."
    cam_rot = data.cam_xmat[cam_id].reshape(3, 3)

    rel_cfg = cfg.waypoint.relative_move
    dx = float(np.random.uniform(*rel_cfg.dx_range_meters))
    dy = float(np.random.uniform(*rel_cfg.dy_range_meters))
    dz = float(np.random.uniform(*rel_cfg.dz_range_meters))

    min_disp = float(rel_cfg.min_displacement_threshold_meters)
    parts = []
    if abs(dx) >= min_disp:
        parts.append(f"{int(round(abs(dx) * 100))}cm {'right' if dx > 0 else 'left'}")
    if abs(dz) >= min_disp:
        parts.append(f"{int(round(abs(dz) * 100))}cm {'forward' if dz > 0 else 'backward'}")
    if abs(dy) >= min_disp:
        parts.append(f"{int(round(abs(dy) * 100))}cm {'up' if dy > 0 else 'down'}")

    if not parts:
        dx = float(rel_cfg.fallback_displacement_meters)
        parts.append(f"{int(round(dx * 100))}cm right")

    prompt = f"move {' and '.join(parts)} relative to camera"

    # MuJoCo camera coordinate convention: optical axis points in -z direction
    cam_delta = np.array([dx, dy, -dz], dtype=np.float32)
    world_delta = cam_rot @ cam_delta

    target_pos = start_pose.position + world_delta

    # Bound target position to reachable workspace
    pivot = shoulder_pan_position(model, data)
    diff = target_pos - pivot
    min_h, max_h = cfg.waypoint.random_pose.height_meters
    target_pos[2] = np.clip(target_pos[2], min_h, max_h)
    diff = target_pos - pivot
    min_distance, max_distance = cfg.waypoint.random_pose.shoulder_distance_meters
    assert abs(diff[2]) <= max_distance
    min_planar_radius = np.sqrt(max(min_distance**2 - diff[2] ** 2, 0.0))
    max_planar_radius = np.sqrt(max_distance**2 - diff[2] ** 2)
    planar_radius = np.linalg.norm(diff[:2])
    if planar_radius < min_planar_radius or planar_radius > max_planar_radius:
        target_pos[:2] = pivot[:2] + diff[:2] / planar_radius * np.clip(
            planar_radius,
            min_planar_radius,
            max_planar_radius,
        )

    target_pose = Pose.from_tcp_axes(
        position=target_pos,
        closing_axis=start_pose.closing_axis,
        secondary_axis=start_pose.secondary_axis,
        gripper=start_pose.gripper,
    )

    primitive = ActionPrimitive(
        start_pose=start_pose,
        target_pose=target_pose,
        prompt=prompt,
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
