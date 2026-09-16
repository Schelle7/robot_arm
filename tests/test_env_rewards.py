import numpy as np
from collections import deque
from types import SimpleNamespace

from robot_arm.envs.env import RobotEnv
from robot_arm.envs.grasp_estimator import GraspEstimator
from robot_arm.geometry.pose import Pose
from robot_arm.robot_schema import HISTORY_FEATURE_NAMES, MOTOR_ORDER


def make_env() -> RobotEnv:
    env = RobotEnv.__new__(RobotEnv)
    env.grasp_estimator = SimpleNamespace(reset=lambda: None)
    env.tracking_progress_enabled = True
    env.joint_limit_penalty_enabled = True
    env.termination_penalty_enabled = True
    env.sustained_duty_penalty_enabled = False
    env.idle_action_penalty_enabled = False
    env.action_change_penalty_enabled = False
    env.pose_delta_diagnostics_enabled = False
    env.position_distance_weight = 1.0
    env.rotation_primary_distance_weight = 1.0
    env.rotation_secondary_distance_weight = 1.0
    env.gripper_distance_weight = 1.0
    env.gripper_duty_weight = 1.0
    env.position_distance_scale = 0.1
    env.rotation_distance_scale = 0.2
    env.gripper_distance_scale = 0.5
    env.joint_limit_penalty_factor = 10.0
    env.action_change_penalty_factor = 1.0
    env.previous_action = np.zeros(6, dtype=np.float32)
    env.previous_position_distance = 0.0
    env.previous_primary_orientation_distance = 0.0
    env.previous_secondary_orientation_distance = 0.0
    env.previous_gripper_distance = 0.0
    return env


def make_pose(position, angles, gripper=0.0) -> Pose:
    return Pose.from_euler(position, angles, gripper, "XYZ", False)


def test_consecutive_velocity_uses_only_the_previous_sample():
    env = make_env()
    env._set_motion_reference(
        np.zeros(6, dtype=np.float32),
        make_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        1_000_000_000,
    )

    joint_velocity, tcp_velocity = env._consecutive_velocities(
        np.full(6, 0.1, dtype=np.float32),
        make_pose([0.01, 0.0, 0.0], [0.0, 0.0, 0.02]),
        1_050_000_000,
    )

    np.testing.assert_allclose(joint_velocity, np.full(6, 2.0))
    np.testing.assert_allclose(tcp_velocity, [0.2, 0.0, 0.0, 0.0, 0.0, 0.4], atol=1e-6)


def test_consecutive_velocity_advances_its_reference():
    env = make_env()
    first_pose = make_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    second_pose = make_pose([0.01, 0.0, 0.0], [0.0, 0.0, 0.0])
    env._set_motion_reference(np.zeros(6, dtype=np.float32), first_pose, 1_000_000_000)
    env._consecutive_velocities(
        np.full(6, 0.1, dtype=np.float32),
        second_pose,
        1_050_000_000,
    )
    joint_velocity, tcp_velocity = env._consecutive_velocities(
        np.full(6, 0.15, dtype=np.float32),
        make_pose([0.015, 0.0, 0.0], [0.0, 0.0, 0.0]),
        1_100_000_000,
    )

    np.testing.assert_allclose(joint_velocity, np.ones(6), atol=1e-6)
    np.testing.assert_allclose(tcp_velocity, [0.1, 0.0, 0.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_reset_policy_history_fills_one_window_with_zeros():
    env = make_env()
    env.policy_history_steps = 10
    env.policy_history = deque([np.ones(len(HISTORY_FEATURE_NAMES), dtype=np.float32)], maxlen=10)

    env._reset_policy_history()

    assert len(env.policy_history) == 10
    np.testing.assert_array_equal(np.stack(env.policy_history), np.zeros((10, len(HISTORY_FEATURE_NAMES))))


def test_box_distance_rejects_grasp_and_restarts_continuous_hold():
    env = make_env()
    env.backend = "sim"
    env.motor_order = MOTOR_ORDER
    env.max_box_distance_meters = 0.04
    env.grasp_estimator = GraspEstimator(SimpleNamespace(
        position_range_radians=(0.3, 0.7),
        min_closing_duty=0.1,
        max_velocity_radians_per_second=0.05,
        hold_seconds=0.2,
    ))
    env.policy_history = deque([np.zeros(len(HISTORY_FEATURE_NAMES))])
    box_pose = SimpleNamespace(position=np.array([0.04, 0.0, 0.0]))
    env.arm = SimpleNamespace(
        get_privileged_box_pose=lambda body_name: box_pose,
        sim_state=lambda: {},
    )
    pose = make_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.4)
    sensor_state = {
        "Present_Position": {"gripper": 0.4},
        "Present_Load": {"gripper": -0.35},
        "sample_time_ns": 0,
    }

    def observe(sample_time_ns):
        sensor_state["sample_time_ns"] = sample_time_ns
        return env._environment_state(sensor_state, np.zeros(6), np.zeros(6), np.zeros(6), pose)

    assert not observe(0).grasp_confirmed
    assert observe(200_000_000).grasp_confirmed
    box_pose.position[0] = 0.041
    assert not observe(250_000_000).grasp_confirmed
    box_pose.position[0] = 0.039
    assert not observe(300_000_000).grasp_confirmed
    assert not observe(499_000_000).grasp_confirmed
    assert observe(500_000_000).grasp_confirmed


def test_desired_pose_is_constructed_from_one_delta():
    env = make_env()
    cartesian_action_start_pose = make_pose([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], 0.2)
    action = np.array([0.1, -0.2, 0.3, 0.0, 0.0, np.pi / 2, 0.4], dtype=np.float32)

    desired_pose = env._compute_desired_pose(cartesian_action_start_pose, action)

    np.testing.assert_allclose(desired_pose.position, [1.1, 1.8, 3.3])
    np.testing.assert_allclose(desired_pose.gripper, 0.6)
    np.testing.assert_allclose(
        desired_pose.angular_distance(make_pose([0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2])),
        0.0,
        atol=1e-6,
    )


def test_pose_distances_use_component_specific_metrics():
    env = make_env()
    current_pose = make_pose([1.0, 1.0, 0.0], [0.0, 0.0, np.pi / 2], 0.2)
    desired_pose = make_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.7)

    (
        position_distance,
        primary_orientation_distance,
        secondary_orientation_distance,
        gripper_distance,
    ) = env._compute_pose_distances(current_pose, desired_pose)

    np.testing.assert_allclose(position_distance, np.sqrt(2.0))
    np.testing.assert_allclose(primary_orientation_distance, np.pi / 2)
    np.testing.assert_allclose(secondary_orientation_distance, np.pi / 2)
    np.testing.assert_allclose(gripper_distance, 0.5)


def test_compute_reward_filters_disabled_components_and_sums_breakdown():
    env = make_env()
    env.tracking_progress_enabled = False
    env.termination_penalty_enabled = False
    cartesian_action_start_pose = make_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    cartesian_action = np.zeros(7, dtype=np.float32)
    requested_action = {"motor": 1.0}
    safe_action = {"motor": 0.5}

    reward, breakdown = env.compute_reward(
        requested_action=requested_action,
        safe_action=safe_action,
        policy_action=np.zeros(6, dtype=np.float32),
        time_left=1.0,
        cartesian_action=cartesian_action,
        current_pose=cartesian_action_start_pose,
        cartesian_action_start_pose=cartesian_action_start_pose,
        cartesian_action_ends=False,
        gripper_duty=0.0,
        desired_gripper_duty=0.0,
        desired_gripper_duty_active=False,
    )

    assert set(breakdown) == {"joint_limit_penalty"}
    assert breakdown["joint_limit_penalty"] == -5.0
    assert reward == sum(breakdown.values())


def test_compute_reward_updates_tracking_state_after_calculation():
    env = make_env()
    cartesian_action_start_pose = make_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    current_pose = make_pose([0.5, 0.5, 0.0], [0.0, 0.0, 0.0])
    cartesian_action = np.zeros(7, dtype=np.float32)

    env.compute_reward(
        requested_action={},
        safe_action={},
        policy_action=np.zeros(6, dtype=np.float32),
        time_left=1.0,
        cartesian_action=cartesian_action,
        current_pose=current_pose,
        cartesian_action_start_pose=cartesian_action_start_pose,
        cartesian_action_ends=False,
        gripper_duty=0.0,
        desired_gripper_duty=0.0,
        desired_gripper_duty_active=False,
    )

    np.testing.assert_allclose(env.previous_position_distance, np.sqrt(0.5), atol=1e-6)
    assert env.previous_primary_orientation_distance == 0.0
    assert env.previous_secondary_orientation_distance == 0.0
    assert env.previous_gripper_distance == 0.0


def test_first_step_progress_reward_uses_cartesian_action_start_distance():
    env = make_env()
    cartesian_action_start_pose = make_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    cartesian_action = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    desired_pose = env._compute_desired_pose(cartesian_action_start_pose, cartesian_action)
    improved_pose = make_pose([0.05, 0.0, 0.0], [0.0, 0.0, 0.0])

    env.reset_cartesian_action_reward_tracking(cartesian_action_start_pose, cartesian_action)
    _, breakdown = env.compute_reward(
        requested_action={},
        safe_action={},
        policy_action=np.zeros(6, dtype=np.float32),
        time_left=1.0,
        cartesian_action=cartesian_action,
        current_pose=improved_pose,
        cartesian_action_start_pose=cartesian_action_start_pose,
        cartesian_action_ends=False,
        gripper_duty=0.0,
        desired_gripper_duty=0.0,
        desired_gripper_duty_active=False,
    )

    initial_distance = cartesian_action_start_pose.positional_distance(desired_pose)
    final_distance = improved_pose.positional_distance(desired_pose)
    np.testing.assert_allclose(
        breakdown["position_reward"],
        (initial_distance - final_distance) / env.position_distance_scale,
    )


def test_terminal_penalty_is_sum_of_normalized_weighted_distances():
    env = make_env()
    current_pose = make_pose([0.1, 0.0, 0.0], [0.0, 0.0, 0.2], 0.5)
    desired_pose = make_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0)

    penalty = env._compute_termination_penalty(True, current_pose, desired_pose, 0.0, False)

    # Position, primary orientation, secondary orientation and gripper each normalize to exactly 1.0.
    np.testing.assert_allclose(penalty, -4.0, rtol=1e-6)


def test_reward_is_zero_when_every_component_is_disabled():
    env = make_env()
    env.tracking_progress_enabled = False
    env.termination_penalty_enabled = False
    current_pose = make_pose([0.1, 0.0, 0.0], [0.0, 0.0, 0.0])

    reward, breakdown = env.compute_reward(
        requested_action={},
        safe_action={},
        policy_action=np.zeros(6, dtype=np.float32),
        time_left=1.0,
        cartesian_action=np.zeros(7, dtype=np.float32),
        current_pose=current_pose,
        cartesian_action_start_pose=make_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        cartesian_action_ends=False,
        gripper_duty=0.0,
        desired_gripper_duty=0.0,
        desired_gripper_duty_active=False,
    )

    assert reward == 0.0
    assert breakdown == {"joint_limit_penalty": 0.0}


def test_sustained_duty_penalty_charges_only_the_excess_per_joint():
    env = make_env()
    env.sustained_duty_allowance = 0.7
    env.sustained_duty_penalty_factor = 2.0
    history = np.zeros((10, len(HISTORY_FEATURE_NAMES)), dtype=np.float32)
    history[:, 6:12] = [0.9, 0.7, 0.2, 0.0, 0.0, 0.8]
    env.policy_history = deque(history, maxlen=10)

    # 0.2 over on one joint and 0.1 on another, with the rest at or below the allowance.
    np.testing.assert_allclose(env._compute_sustained_duty_penalty(), -0.6, rtol=1e-6)


def test_action_change_penalty_uses_actor_action_before_duty_scaling():
    env = make_env()
    env.tracking_progress_enabled = False
    env.joint_limit_penalty_enabled = False
    env.termination_penalty_enabled = False
    env.action_change_penalty_enabled = True
    env.previous_action = np.array([-1.0, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)
    policy_action = np.array([1.0, 0.0, -0.5, 0.0, 0.0, 0.0], dtype=np.float32)
    pose = make_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    reward, breakdown = env.compute_reward(
        {},
        {},
        policy_action,
        1.0,
        np.zeros(7, dtype=np.float32),
        pose,
        pose,
        False,
        0.0,
        0.0,
        False,
    )

    expected = -np.mean(np.square(policy_action - env.previous_action))
    assert breakdown == {"action_change_penalty": expected}
    assert reward == expected


def test_reset_clears_tracking_state():
    env = make_env()
    env._initial_environment_state = lambda: object()
    env.backend = "real"
    env.motor_order = MOTOR_ORDER
    env.policy_history_steps = 10
    env.policy_history = deque(maxlen=10)
    env.previous_position_distance = 2.0
    env.previous_primary_orientation_distance = 3.0
    env.previous_secondary_orientation_distance = 4.0
    env.previous_gripper_distance = 4.0
    env.previous_action = np.ones(6, dtype=np.float32)

    env.reset(enable_added_weight=False)

    assert env.previous_position_distance == 0.0
    assert env.previous_primary_orientation_distance == 0.0
    assert env.previous_secondary_orientation_distance == 0.0
    assert env.previous_gripper_distance == 0.0
    np.testing.assert_array_equal(env.previous_action, np.zeros(6, dtype=np.float32))
