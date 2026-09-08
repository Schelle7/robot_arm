import mujoco
import numpy as np
import pytest
from omegaconf import OmegaConf

from robot_arm.waypoints import base_rotation_for_position, generate_oriented_waypoint
from robot_arm.policies import ScriptedCartesianPolicy
from robot_arm.pose import Pose
from robot_arm.primitive_policy import ScriptedPrimitiveGeneratorPolicy
from robot_arm.primitives import ActionPrimitive, generate_pick_and_place, generate_relative_moves


def make_policy():
    cfg = OmegaConf.create(
        {
            "waypoint": {
                "duty_completion_tolerance": 0.1,
                "completion_tolerance": {
                    "position_meters": 1.0,
                    "primary_rotation_radians": 1.0,
                    "secondary_rotation_radians": 1.0,
                    "gripper_radians": 1.0,
                },
                "position_speed_meters_per_second": 1.0,
                "rotation_speed_radians_per_second": 1.0,
                "gripper_speed_radians_per_second": 1.0,
            },
            "control": {
                "frequencies": {
                    "cartesian": 1,
                }
            },
        }
    )
    return ScriptedCartesianPolicy(cfg)


def make_primitive(start_pose: Pose, target_pose: Pose, prompt: str = "follow waypoint", include_target_offset: bool = True) -> ActionPrimitive:
    return ActionPrimitive(
        start_pose=start_pose,
        target_pose=target_pose,
        prompt=prompt,
        include_target_offset=include_target_offset,
        desired_gripper_duty=0.0,
        desired_gripper_duty_active=False,
    )


def get_scripted_action(policy, current_pose: Pose, target_pose: Pose):
    return policy.get_action(
        current_pose=current_pose,
        images={
            "external_camera": np.zeros((1, 1, 3), dtype=np.uint8),
            "wrist_camera": np.zeros((1, 1, 3), dtype=np.uint8),
        },
        vla_input_state=np.zeros(16, dtype=np.float32),
        gripper_duty=0.0,
        primitive=make_primitive(current_pose, target_pose),
    )


def test_waypoint_translation_is_limited_by_vector_length():
    policy = make_policy()
    current_pose = Pose.from_euler([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "XYZ", False)
    target_pose = Pose.from_euler([2.0, 2.0, 2.0], [0.0, 0.0, 0.0], 0.0, "XYZ", False)

    output = get_scripted_action(policy, current_pose, target_pose)

    np.testing.assert_allclose(np.linalg.norm(output.cartesian_action[:3]), 1.0)


def test_waypoint_translation_preserves_direction():
    policy = make_policy()
    current_pose = Pose.from_euler([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "XYZ", False)
    target_pose = Pose.from_euler([2.0, 1.0, 0.0], [0.0, 0.0, 0.0], 0.0, "XYZ", False)

    output = get_scripted_action(policy, current_pose, target_pose)

    np.testing.assert_allclose(output.cartesian_action[:3], np.array([2.0, 1.0, 0.0]) / np.sqrt(5.0))


def test_waypoint_rotation_is_limited_by_vector_length():
    policy = make_policy()
    current_pose = Pose.from_euler([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "XYZ", False)
    rotation_delta = np.full(3, 1.5 / np.sqrt(3.0))
    target_pose = current_pose.apply_delta(np.concatenate([np.zeros(3), rotation_delta, [0.0]]))

    # Every component sits inside the per-axis limit, so only a norm-based limit constrains this rotation.
    assert np.all(np.abs(rotation_delta) < policy.max_rotation_delta)

    output = get_scripted_action(policy, current_pose, target_pose)

    np.testing.assert_allclose(
        np.linalg.norm(output.cartesian_action[3:6]),
        policy.max_rotation_delta,
        rtol=1e-6,
    )


def test_scripted_primitive_policy_builds_current_vla_context_and_advances_immediately():
    primitive_policy = ScriptedPrimitiveGeneratorPolicy(OmegaConf.create({}))
    start_pose = Pose.from_euler([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "XYZ", False)
    target_pose = Pose.from_euler([0.1, 0.0, 0.0], [0.0, 0.0, 0.0], 0.2, "XYZ", False)
    next_target_pose = Pose.from_euler([0.2, 0.0, 0.0], [0.0, 0.0, 0.0], 0.2, "XYZ", False)
    primitive_policy.primitives = [
        make_primitive(start_pose, target_pose, "move right"),
        make_primitive(target_pose, next_target_pose, "move right again"),
    ]

    primitive_index, primitive = primitive_policy.get_next_primitive(start_pose)
    vla_input_state = primitive_policy.build_vla_input_state(primitive, start_pose, 0.25)

    np.testing.assert_allclose(vla_input_state[:7], start_pose.as_7d())
    np.testing.assert_allclose(vla_input_state[7:10], [0.1, 0.0, 0.0])
    assert vla_input_state[14] == 1.0
    assert vla_input_state[15] == 0.25
    assert primitive.prompt == "move right"
    assert primitive_index == 0
    assert primitive_policy.next_primitive_index == 1

    actual_next_start_pose = Pose.from_euler([0.09, 0.0, 0.0], [0.0, 0.0, 0.0], 0.19, "XYZ", False)
    _, next_primitive = primitive_policy.get_next_primitive(actual_next_start_pose)
    next_vla_input_state = primitive_policy.build_vla_input_state(next_primitive, actual_next_start_pose, 0.0)
    np.testing.assert_allclose(next_vla_input_state[:7], actual_next_start_pose.as_7d())


def test_scripted_primitive_policy_updates_remaining_target_offset():
    primitive_policy = ScriptedPrimitiveGeneratorPolicy(OmegaConf.create({}))
    start_pose = Pose.from_euler([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "XYZ", False)
    current_pose = Pose.from_euler([0.04, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "XYZ", False)
    target_pose = Pose.from_euler([0.1, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "XYZ", False)
    primitive = make_primitive(start_pose, target_pose, "move right")

    vla_input_state = primitive_policy.build_vla_input_state(primitive, current_pose, 0.0)

    np.testing.assert_allclose(vla_input_state[:7], current_pose.as_7d())
    np.testing.assert_allclose(vla_input_state[7:10], [0.06, 0.0, 0.0])


def test_scripted_primitive_policy_hides_privileged_target_offset():
    primitive_policy = ScriptedPrimitiveGeneratorPolicy(OmegaConf.create({}))
    current_pose = Pose.from_euler([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "XYZ", False)
    target_pose = Pose.from_euler([0.1, 0.2, 0.3], [0.0, 0.0, 0.0], 0.0, "XYZ", False)
    primitive = make_primitive(current_pose, target_pose, "move above the red tile", include_target_offset=False)

    vla_input_state = primitive_policy.build_vla_input_state(primitive, current_pose, 0.25)

    np.testing.assert_array_equal(vla_input_state[7:14], np.zeros(7, dtype=np.float32))
    assert vla_input_state[14] == 0.0
    assert primitive.target_pose is target_pose


def test_pick_and_place_uses_single_purpose_primitives():
    model = mujoco.MjModel.from_xml_path("models/so101/scene.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cfg = OmegaConf.create(
        {
            "waypoint": {
                "pick_and_place": {
                    "grasp_z_offset_meters": 0.01,
                    "lift_z_offset_meters": 0.08,
                    "gripper_open_radians": 0.8,
                    "gripper_closed_radians": 0.0,
                    "desired_gripper_duty": 0.35,
                }
            }
        }
    )
    start_pose = Pose.from_euler([0.35, 0.0, 0.25], [0.0, 0.0, 0.0], 0.0, "XYZ", False)

    primitives = generate_pick_and_place(model, data, cfg, start_pose)

    assert len(primitives) == 8
    assert primitives[0].prompt == "open gripper"
    assert primitives[1].prompt.startswith("move above ")
    assert primitives[2].prompt.startswith("move to ")
    assert primitives[3].prompt == "close gripper"
    assert primitives[4].prompt == "lift object"
    assert " above the " in primitives[5].prompt
    assert primitives[6].prompt.startswith("lower ")
    assert primitives[7].prompt == "open gripper"
    above_pose = primitives[1].target_pose
    grasp_pose = primitives[2].target_pose
    np.testing.assert_allclose(above_pose.position[:2], grasp_pose.position[:2])
    np.testing.assert_allclose(above_pose.position[2] - grasp_pose.position[2], 0.07)
    np.testing.assert_allclose(above_pose.closing_axis, grasp_pose.closing_axis)
    np.testing.assert_allclose(above_pose.secondary_axis, grasp_pose.secondary_axis)
    assert above_pose.gripper == grasp_pose.gripper
    assert not any(primitive.include_target_offset for primitive in primitives)
    assert [primitive.desired_gripper_duty_active for primitive in primitives] == [
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        False,
    ]
    for current, following in zip(primitives, primitives[1:]):
        np.testing.assert_allclose(current.target_pose.as_10d(), following.start_pose.as_10d())


def test_relative_move_prompt_states_the_commanded_offset_exactly():
    model = mujoco.MjModel.from_xml_path("models/so101/scene.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cfg = OmegaConf.create(
        {
            "waypoint": {
                "relative_move": {
                    "dx_range_meters": [-0.08, 0.08],
                    "dy_range_meters": [-0.06, 0.06],
                    "dz_range_meters": [-0.08, 0.08],
                },
                "random_pose": {
                    "shoulder_distance_meters": [0.30, 0.45],
                    "height_meters": [0.1, 0.40],
                },
            }
        }
    )
    start_pose = Pose.from_euler([0.35, 0.0, 0.25], [0.0, 0.0, 0.0], 0.0, "XYZ", False)

    for _ in range(50):
        primitive = generate_relative_moves(model, data, cfg, start_pose)[0]
        offset = primitive.target_pose.position - start_pose.position
        offset_cm = np.rint(offset * 100).astype(int)

        np.testing.assert_allclose(offset, offset_cm / 100.0, atol=1e-6)
        assert primitive.prompt.count("cm along") == int(np.count_nonzero(offset_cm))
        for axis, centimeters in zip("xyz", offset_cm):
            if centimeters:
                assert f"{centimeters}cm along {axis}" in primitive.prompt
        if not offset_cm.any():
            assert primitive.prompt == "hold position"


@pytest.fixture
def zero_offset_gripper(monkeypatch):
    def geometry(model, data, gripper):
        return Pose.from_tcp_axes(np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), gripper)

    monkeypatch.setattr("robot_arm.waypoints.gripper_geometry_at_opening", geometry)


def test_waypoint_derives_azimuth_from_position_without_gripper_offset(zero_offset_gripper):
    model = mujoco.MjModel.from_xml_path("models/so101/scene.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    shoulder_pan_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_pan")
    shoulder_pan_position = data.xanchor[shoulder_pan_joint_id]
    pose = generate_oriented_waypoint(
        model,
        data,
        shoulder_pan_position + np.array([0.0, 0.1, 0.2]),
        pointing_axis_tilt_degrees=0.0,
        pointing_axis_rotation_degrees=0.0,
        gripper=0.5,
    )

    np.testing.assert_allclose(pose.closing_axis, [0.0, 2.0, -1.0] / np.sqrt(5.0), atol=1e-6)
    np.testing.assert_allclose(pose.secondary_axis, [-1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(pose.as_matrix()[:, 2], [0.0, 1.0, 2.0] / np.sqrt(5.0), atol=1e-6)
    assert pose.gripper == 0.5


def test_waypoint_rotates_axes_around_pointing_axis_without_gripper_offset(zero_offset_gripper):
    model = mujoco.MjModel.from_xml_path("models/so101/scene.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    shoulder_pan_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_pan")
    shoulder_pan_position = data.xanchor[shoulder_pan_joint_id]
    pose = generate_oriented_waypoint(
        model,
        data,
        shoulder_pan_position + np.array([0.1, 0.0, 0.2]),
        pointing_axis_tilt_degrees=0.0,
        pointing_axis_rotation_degrees=90.0,
        gripper=0.5,
    )

    np.testing.assert_allclose(pose.closing_axis, [0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(pose.secondary_axis, [-2.0, 0.0, 1.0] / np.sqrt(5.0), atol=1e-6)
    np.testing.assert_allclose(pose.as_matrix()[:, 2], [1.0, 0.0, 2.0] / np.sqrt(5.0), atol=1e-6)


def test_waypoint_tilt_preserves_position_without_gripper_offset(zero_offset_gripper):
    model = mujoco.MjModel.from_xml_path("models/so101/scene.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    shoulder_pan_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_pan")
    shoulder_pan_position = data.xanchor[shoulder_pan_joint_id]
    position = shoulder_pan_position + np.array([0.1, 0.0, 0.0])
    pose = generate_oriented_waypoint(
        model,
        data,
        position,
        pointing_axis_tilt_degrees=90.0,
        pointing_axis_rotation_degrees=0.0,
        gripper=0.5,
    )

    np.testing.assert_allclose(pose.position, position)
    np.testing.assert_allclose(pose.closing_axis, [-1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(pose.secondary_axis, [0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(pose.as_matrix()[:, 2], [0.0, 0.0, -1.0], atol=1e-6)


def test_base_rotation_uses_shoulder_pan_joint_anchor():
    model = mujoco.MjModel.from_xml_path("models/so101/scene.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    shoulder_pan_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_pan")
    shoulder_pan_position = data.xanchor[shoulder_pan_joint_id]

    forward_position = shoulder_pan_position + np.array([0.1, 0.0, 0.0])
    side_position = shoulder_pan_position + np.array([0.0, 0.1, 0.0])

    np.testing.assert_allclose(base_rotation_for_position(model, data, forward_position), 0.0)
    np.testing.assert_allclose(base_rotation_for_position(model, data, side_position), np.pi / 2)
