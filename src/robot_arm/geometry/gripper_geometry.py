import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from robot_arm.geometry.pose import Pose


def get_tcp_geometry(model, data):
    fixed = data.site_xpos[model.site("fixed_finger_tip").id].copy()
    moving = data.site_xpos[model.site("moving_finger_tip").id].copy()
    frame_rotation = data.site_xmat[model.site("gripperframe").id].reshape(3, 3)
    closing = fixed - moving
    closing = closing / np.linalg.norm(closing)
    secondary = frame_rotation[:, 1]
    secondary = secondary - np.dot(secondary, closing) * closing
    secondary = secondary / np.linalg.norm(secondary)
    gripper_qpos = model.jnt_qposadr[model.joint("gripper").id]
    pose = Pose.from_tcp_axes(
        (fixed + moving) / 2.0,
        closing,
        secondary,
        float(data.qpos[gripper_qpos]),
    )
    return pose, fixed, moving


def gripper_geometry_at_opening(model, data, gripper: float) -> Pose:
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.qpos[model.jnt_qposadr[model.joint("gripper").id]] = gripper
    mujoco.mj_kinematics(model, scratch)
    pose, fixed, _ = get_tcp_geometry(model, scratch)
    frame_rotation = scratch.site_xmat[model.site("gripperframe").id].reshape(3, 3)
    # Local Z points along the fixed finger, with local X across the gap toward it.
    local_to_world = frame_rotation @ np.diag([-1.0, 1.0, -1.0])
    return Pose.from_tcp_axes(
        local_to_world.T @ (pose.position - fixed),
        local_to_world.T @ pose.closing_axis,
        local_to_world.T @ pose.secondary_axis,
        gripper,
    )


def align_gripper_to_target(
    local_pose: Pose,
    target_distance: float,
    tilt_degrees: float,
    rotation_degrees: float,
) -> tuple[Pose, np.ndarray]:
    tilt = Rotation.from_rotvec(np.array([0.0, np.deg2rad(tilt_degrees), 0.0])).as_matrix()
    roll = Rotation.from_rotvec(np.array([0.0, 0.0, np.deg2rad(rotation_degrees)])).as_matrix()
    orientation = tilt @ roll
    offset = orientation @ local_pose.position
    assert target_distance > 0.0, "A waypoint at the base cannot define a radial orientation."
    radial_squared = target_distance**2 - np.dot(offset[:2], offset[:2])
    assert radial_squared >= 0.0, "Target distance is smaller than the gripper's transverse TCP offset."
    reference_distance = np.sqrt(radial_squared) - offset[2]
    assert reference_distance >= 0.0, "Fixed-fingertip reference would lie behind the base."
    reference = np.array([0.0, 0.0, reference_distance])
    direction = (reference + offset) / target_distance

    # Shortest rotation onto +Z moves the reference and gripper together around the base.
    cross = np.cross(direction, np.array([0.0, 0.0, 1.0]))
    x, y, z = cross
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    alignment = np.eye(3) + skew + (skew @ skew) / (1.0 + direction[2])
    orientation = alignment @ orientation
    reference = alignment @ reference
    return Pose.from_tcp_axes(
        reference + orientation @ local_pose.position,
        orientation @ local_pose.closing_axis,
        orientation @ local_pose.secondary_axis,
        local_pose.gripper,
    ), reference
