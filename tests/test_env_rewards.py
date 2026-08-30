import numpy as np

from robot_arm.envs.env import RobotEnv
from robot_arm.pose import Pose


def make_env() -> RobotEnv:
    env = RobotEnv.__new__(RobotEnv)
    env.tracking_progress_enabled = True
    env.joint_limit_penalty_enabled = True
    env.termination_penalty_enabled = True
    env.sustained_duty_penalty_enabled = False
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
    env.previous_position_distance = 0.0
    env.previous_primary_orientation_distance = 0.0
    env.previous_secondary_orientation_distance = 0.0
    env.previous_gripper_distance = 0.0
    return env


def make_pose(position, angles, gripper=0.0) -> Pose:
    return Pose.from_euler(position, angles, gripper, "XYZ", False)


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
        requested_action,
        safe_action,
        cartesian_action,
        cartesian_action_start_pose,
        cartesian_action_start_pose,
        cartesian_action_terminated=False,
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
        {},
        {},
        cartesian_action,
        current_pose,
        cartesian_action_start_pose,
        cartesian_action_terminated=False,
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
        {},
        {},
        cartesian_action,
        improved_pose,
        cartesian_action_start_pose,
        cartesian_action_terminated=False,
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
        {},
        {},
        np.zeros(7, dtype=np.float32),
        current_pose,
        make_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        cartesian_action_terminated=False,
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
    env.duty_history = np.array([0.9, 0.7, 0.2, 0.0, 0.0, 0.8], dtype=np.float32)

    # 0.2 over on one joint and 0.1 on another, with the rest at or below the allowance.
    np.testing.assert_allclose(env._compute_sustained_duty_penalty(), -0.6, rtol=1e-6)


def test_reset_clears_tracking_state():
    env = make_env()
    pose = make_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    env.arm = type("ArmStub", (), {"read_state": lambda self: {}, "get_tcp_pose": lambda self, state: pose})()
    env._get_obs = lambda: {}
    env.backend = "real"
    env.staging_enabled = False
    env.previous_position_distance = 2.0
    env.previous_primary_orientation_distance = 3.0
    env.previous_secondary_orientation_distance = 4.0
    env.previous_gripper_distance = 4.0
    env.duty_history = np.ones(6, dtype=np.float32)

    env.reset()

    assert env.previous_position_distance == 0.0
    assert env.previous_primary_orientation_distance == 0.0
    assert env.previous_secondary_orientation_distance == 0.0
    assert env.previous_gripper_distance == 0.0
