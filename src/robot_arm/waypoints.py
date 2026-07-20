import numpy as np


def generate_grab_waypoints(
    box_pose_6d: np.ndarray,
    lift_height: float,
    gripper_open: float,
    gripper_closed: float,
) -> list[np.ndarray]:
    """
    Generates a generic 3-waypoint sequence for grasping a given 6D pose:
    1. Move to the target with the gripper completely open.
    2. Close the gripper while remaining in place.
    3. Move straight up along the Z-axis with the gripper closed.

    box_pose_6d: [x, y, z, roll, pitch, yaw]
    """
    # 1. Move to target (Max open gripper)
    wp1 = np.zeros(7, dtype=np.float32)
    wp1[:6] = box_pose_6d
    wp1[6] = gripper_open

    # 2. Close gripper (Remain still)
    wp2 = wp1.copy()
    wp2[6] = gripper_closed

    # 3. Lift straight up (Keep closed)
    wp3 = wp2.copy()
    wp3[2] += lift_height

    return [wp1, wp2, wp3]
