import mujoco
import numpy as np

from robot_arm.geometry.pose import Pose
from robot_arm.geometry.gripper_geometry import gripper_geometry_at_opening, align_gripper_to_target


def shoulder_pan_position(model, data) -> np.ndarray:
    shoulder_pan_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_pan")
    assert shoulder_pan_joint_id != -1
    return data.xanchor[shoulder_pan_joint_id]


def base_rotation_for_position(model, data, position: np.ndarray) -> float:
    pivot = shoulder_pan_position(model, data)
    return float(np.arctan2(position[1] - pivot[1], position[0] - pivot[0]))


def position_from_base_rotation(model, data, radius: float, base_rotation_degrees: float, height: float) -> np.ndarray:
    pivot = shoulder_pan_position(model, data)
    base_rotation = np.deg2rad(base_rotation_degrees)
    return np.array(
        [
            pivot[0] + radius * np.cos(base_rotation),
            pivot[1] + radius * np.sin(base_rotation),
            height,
        ],
        dtype=np.float32,
    )


def generate_oriented_waypoint(
    model,
    data,
    position: np.ndarray,
    pointing_axis_tilt_degrees: float,
    pointing_axis_rotation_degrees: float,
    gripper: float,
) -> Pose:
    position = np.asarray(position, dtype=np.float32)
    pivot = shoulder_pan_position(model, data)
    pointing_axis = position - pivot
    pointing_axis = pointing_axis / np.linalg.norm(pointing_axis)
    base_rotation = base_rotation_for_position(model, data, position)
    secondary_axis = np.array([-np.sin(base_rotation), np.cos(base_rotation), 0.0], dtype=np.float32)
    closing_axis = np.cross(secondary_axis, pointing_axis)
    radial_frame = np.column_stack((closing_axis, secondary_axis, pointing_axis))
    local_pose = gripper_geometry_at_opening(model, data, gripper)
    aligned_pose, _ = align_gripper_to_target(
        local_pose,
        np.linalg.norm(position - pivot),
        pointing_axis_tilt_degrees,
        pointing_axis_rotation_degrees,
    )
    return Pose.from_tcp_axes(
        position,
        radial_frame @ aligned_pose.closing_axis,
        radial_frame @ aligned_pose.secondary_axis,
        gripper,
    )
