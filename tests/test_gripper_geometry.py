import mujoco
import numpy as np
import pytest

from robot_arm.geometry.gripper_geometry import align_gripper_to_target, get_tcp_geometry, gripper_geometry_at_opening
from robot_arm.geometry.waypoints import generate_oriented_waypoint, shoulder_pan_position


@pytest.fixture
def scene():
    model = mujoco.MjModel.from_xml_path("models/so101/scene.xml")
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    return model, data


def test_requested_opening_preserves_live_state_and_ignores_arm_orientation(scene):
    model, data = scene
    before_qpos = data.qpos.copy()
    before_sites = data.site_xpos.copy()
    opened = gripper_geometry_at_opening(model, data, 0.8)
    closed = gripper_geometry_at_opening(model, data, 0.0)
    np.testing.assert_array_equal(data.qpos, before_qpos)
    np.testing.assert_array_equal(data.site_xpos, before_sites)
    assert not np.allclose(opened.position, closed.position)
    assert not np.allclose(opened.closing_axis, closed.closing_axis)

    data.qpos[model.jnt_qposadr[model.joint("wrist_roll").id]] = 0.7
    data.qpos[model.jnt_qposadr[model.joint("shoulder_pan").id]] = -0.4
    mujoco.mj_kinematics(model, data)
    rotated = gripper_geometry_at_opening(model, data, 0.8)
    np.testing.assert_allclose(rotated.as_10d(), opened.as_10d(), atol=1e-6)


@pytest.mark.parametrize(
    "opening,tilt,roll,distance",
    [
        (0.0, 0.0, 0.0, 0.3),
        (0.8, 45.0, 90.0, 0.25),
        (1.7, -10.0, -150.0, 0.4),
        (0.5, 90.0, 0.0, 0.1),
    ],
)
def test_spherical_placement_preserves_finger_geometry_and_hits_target(scene, opening, tilt, roll, distance):
    model, data = scene
    local = gripper_geometry_at_opening(model, data, opening)
    aligned, fixed = align_gripper_to_target(local, distance, tilt, roll)
    np.testing.assert_allclose(aligned.position, [0.0, 0.0, distance], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(aligned.position - fixed), np.linalg.norm(local.position), atol=1e-6)
    moving = 2.0 * aligned.position - fixed
    direction = fixed - moving
    np.testing.assert_allclose(aligned.closing_axis, direction / np.linalg.norm(direction), atol=2e-6)
    np.testing.assert_allclose(np.dot(aligned.closing_axis, aligned.secondary_axis), 0.0, atol=1e-6)


def test_waypoint_geometry_can_be_reproduced_by_rigidly_placing_the_open_gripper(scene):
    model, data = scene
    data.qpos[model.jnt_qposadr[model.joint("gripper").id]] = 0.8
    mujoco.mj_kinematics(model, data)
    measured, fixed, moving = get_tcp_geometry(model, data)
    target = shoulder_pan_position(model, data) + np.array([0.2, -0.15, 0.1])
    waypoint = generate_oriented_waypoint(model, data, target, 45.0, 90.0, 0.8)
    rotation = waypoint.as_matrix() @ measured.as_matrix().T
    placed_fixed = target + rotation @ (fixed - measured.position)
    placed_moving = target + rotation @ (moving - measured.position)
    np.testing.assert_allclose((placed_fixed + placed_moving) / 2.0, target, atol=1e-6)
    direction = placed_fixed - placed_moving
    np.testing.assert_allclose(waypoint.closing_axis, direction / np.linalg.norm(direction), atol=1e-6)

    closed_waypoint = generate_oriented_waypoint(model, data, target, 45.0, 90.0, 0.0)
    assert not np.allclose(waypoint.closing_axis, closed_waypoint.closing_axis)


def test_impossible_transverse_offset_fails_explicitly(scene):
    model, data = scene
    local = gripper_geometry_at_opening(model, data, 0.8)
    with pytest.raises(AssertionError, match="transverse TCP offset"):
        align_gripper_to_target(local, 0.001, 0.0, 0.0)
