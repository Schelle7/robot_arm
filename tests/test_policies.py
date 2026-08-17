import mujoco
import numpy as np
from omegaconf import OmegaConf

from robot_arm.experimental_waypoints import base_rotation_for_position, generate_oriented_waypoint
from robot_arm.policies import ScriptedCartesianPolicy
from robot_arm.pose import Pose
from robot_arm.primitive_policy import ScriptedPrimitiveGeneratorPolicy
from robot_arm.primitives import ActionPrimitive


def make_policy():
    cfg = OmegaConf.create(
        {
            "waypoint": {
                "trajectory_length": 1,
                "position_speed_meters_per_second": 1.0,
                "rotation_speed_radians_per_second": 1.0,
                "gripper_speed_radians_per_second": 1.0,
            },
            "control": {
                "frequencies": {
                    "low_level": 1,
                }
            },
        }
    )
    return ScriptedCartesianPolicy(cfg)


def get_scripted_action(policy, current_pose: Pose, target_pose: Pose):
    return policy.get_action(
        current_pose=current_pose,
        image=np.zeros((1, 1, 3), dtype=np.uint8),
        vla_input_state=np.zeros(15, dtype=np.float32),
        primitive_prompt="follow waypoint",
        privileged_target_pose=target_pose,
    )


def test_waypoint_translation_is_limited_by_vector_length():
    policy = make_policy()
    current_pose = Pose.from_euler([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "XYZ", False)
    target_pose = Pose.from_euler([2.0, 2.0, 2.0], [0.0, 0.0, 0.0], 0.0, "XYZ", False)

    output = get_scripted_action(policy, current_pose, target_pose)

    np.testing.assert_allclose(np.linalg.norm(output.cartesian_action_path[0, :3]), 1.0)


def test_waypoint_translation_preserves_direction():
    policy = make_policy()
    current_pose = Pose.from_euler([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "XYZ", False)
    target_pose = Pose.from_euler([2.0, 1.0, 0.0], [0.0, 0.0, 0.0], 0.0, "XYZ", False)

    output = get_scripted_action(policy, current_pose, target_pose)

    np.testing.assert_allclose(output.cartesian_action_path[0, :3], np.array([2.0, 1.0, 0.0]) / np.sqrt(5.0))


def test_scripted_primitive_policy_builds_current_vla_context_and_advances_immediately():
    primitive_policy = ScriptedPrimitiveGeneratorPolicy(OmegaConf.create({}))
    start_pose = Pose.from_euler([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "XYZ", False)
    target_pose = Pose.from_euler([0.1, 0.0, 0.0], [0.0, 0.0, 0.0], 0.2, "XYZ", False)
    next_target_pose = Pose.from_euler([0.2, 0.0, 0.0], [0.0, 0.0, 0.0], 0.2, "XYZ", False)
    primitive_policy.primitives = [
        ActionPrimitive(
            start_pose=start_pose,
            target_pose=target_pose,
            prompt="move right",
            has_explicit_goal=True,
        ),
        ActionPrimitive(
            start_pose=target_pose,
            target_pose=next_target_pose,
            prompt="move right again",
            has_explicit_goal=True,
        ),
    ]

    primitive_index, primitive = primitive_policy.get_next_primitive(start_pose)
    vla_input_state = primitive_policy.build_vla_input_state(primitive, start_pose)

    np.testing.assert_allclose(vla_input_state[:7], start_pose.as_7d())
    np.testing.assert_allclose(vla_input_state[7:10], [0.1, 0.0, 0.0])
    assert vla_input_state[-1] == 1.0
    assert primitive.prompt == "move right"
    assert primitive_index == 0
    assert primitive_policy.next_primitive_index == 1

    actual_next_start_pose = Pose.from_euler([0.09, 0.0, 0.0], [0.0, 0.0, 0.0], 0.19, "XYZ", False)
    _, next_primitive = primitive_policy.get_next_primitive(actual_next_start_pose)
    next_vla_input_state = primitive_policy.build_vla_input_state(next_primitive, actual_next_start_pose)
    np.testing.assert_allclose(next_vla_input_state[:7], actual_next_start_pose.as_7d())


def test_scripted_primitive_policy_updates_remaining_target_offset():
    primitive_policy = ScriptedPrimitiveGeneratorPolicy(OmegaConf.create({}))
    start_pose = Pose.from_euler([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "XYZ", False)
    current_pose = Pose.from_euler([0.04, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "XYZ", False)
    target_pose = Pose.from_euler([0.1, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "XYZ", False)
    primitive = ActionPrimitive(
        start_pose=start_pose,
        target_pose=target_pose,
        prompt="move right",
        has_explicit_goal=True,
    )

    vla_input_state = primitive_policy.build_vla_input_state(primitive, current_pose)

    np.testing.assert_allclose(vla_input_state[:7], current_pose.as_7d())
    np.testing.assert_allclose(vla_input_state[7:10], [0.06, 0.0, 0.0])


def test_action_primitive_generation_uses_configured_ranges():
    model = mujoco.MjModel.from_xml_path("models/so101/scene.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cfg = OmegaConf.create(
        {
            "waypoint": {
                "num_waypoints": 3,
                "primitive_probabilities": {
                    "pick_and_place": 0.0,
                    "relative_move": 0.0,
                    "random_waypoint": 1.0,
                },
                "random_pose": {
                    "gripper_radians": [0.2, 0.7],
                    "shoulder_distance_meters": [0.35, 0.35],
                    "base_rotation_degrees": [0.0, 0.0],
                    "height_meters": [0.2, 0.2],
                    "pointing_axis_tilt_degrees": [0.0, 0.0],
                    "pointing_axis_rotation_degrees": [0.0, 0.0],
                },
                "pick_and_place": {
                    "grasp_z_offset_meters": 0.01,
                    "lift_z_offset_meters": 0.08,
                    "gripper_open_radians": 0.8,
                    "gripper_closed_radians": 0.0,
                },
                "relative_move": {
                    "dx_range_meters": [0.05, 0.05],
                    "dy_range_meters": [0.05, 0.05],
                    "dz_range_meters": [0.05, 0.05],
                    "min_displacement_threshold_meters": 0.02,
                    "fallback_displacement_meters": 0.05,
                },
            }
        }
    )
    primitive_policy = ScriptedPrimitiveGeneratorPolicy(cfg)

    primitive_policy.generate(
        model,
        data,
        Pose.from_euler([0.3, 0.0, 0.2], [0.0, 0.0, 0.0], 0.0, "XYZ", False),
    )

    assert len(primitive_policy.primitives) >= 1
    for primitive in primitive_policy.primitives:
        assert isinstance(primitive.prompt, str)
        assert len(primitive.prompt) > 0


def test_experimental_waypoint_derives_azimuth_from_position():
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

    np.testing.assert_allclose(pose.closing_axis, [0.0, 2.0, -1.0] / np.sqrt(5.0))
    np.testing.assert_allclose(pose.secondary_axis, [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(pose.as_matrix()[:, 2], [0.0, 1.0, 2.0] / np.sqrt(5.0))
    assert pose.gripper == 0.5


def test_experimental_waypoint_rotates_drawn_axes_around_pointing_axis():
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


def test_experimental_waypoint_tilt_rotates_pointing_axis_without_moving_position():
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
