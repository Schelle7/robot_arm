import mujoco
import numpy as np

from robot_arm.pose import Pose


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
    pointing_axis_tilt = np.deg2rad(pointing_axis_tilt_degrees)
    pointing_axis_rotation = np.deg2rad(pointing_axis_rotation_degrees)
    pivot = shoulder_pan_position(model, data)
    pointing_axis = position - pivot
    pointing_axis = pointing_axis / np.linalg.norm(pointing_axis)
    base_rotation = base_rotation_for_position(model, data, position)
    secondary_axis = np.array([-np.sin(base_rotation), np.cos(base_rotation), 0.0], dtype=np.float32)
    closing_axis = np.cross(secondary_axis, pointing_axis)
    tilted_pointing_axis = (
        np.cos(pointing_axis_tilt) * pointing_axis
        + np.sin(pointing_axis_tilt) * np.cross(secondary_axis, pointing_axis)
    )
    tilted_closing_axis = (
        np.cos(pointing_axis_tilt) * closing_axis
        + np.sin(pointing_axis_tilt) * np.cross(secondary_axis, closing_axis)
    )
    rotated_closing_axis = (
        np.cos(pointing_axis_rotation) * tilted_closing_axis
        + np.sin(pointing_axis_rotation) * np.cross(tilted_pointing_axis, tilted_closing_axis)
    )
    rotated_secondary_axis = (
        np.cos(pointing_axis_rotation) * secondary_axis
        + np.sin(pointing_axis_rotation) * np.cross(tilted_pointing_axis, secondary_axis)
    )

    return Pose.from_tcp_axes(position, rotated_closing_axis, rotated_secondary_axis, gripper)
