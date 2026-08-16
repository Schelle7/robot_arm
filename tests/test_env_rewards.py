import numpy as np

from robot_arm.envs.env import RobotEnv
from robot_arm.pose import Pose


def make_env() -> RobotEnv:
    env = RobotEnv.__new__(RobotEnv)
    env.tracking_deviation_enabled = True
    env.tracking_progress_enabled = True
    env.safety_penalty_enabled = True
    env.termination_penalty_enabled = True
    env.pose_delta_diagnostics_enabled = False
    env.position_distance_weight = 1.0
    env.rotation_primary_distance_weight = 1.0
    env.rotation_secondary_distance_weight = 1.0
    env.gripper_distance_weight = 1.0
    env.position_distance_scale = 0.1
    env.rotation_distance_scale = 0.2
    env.gripper_distance_scale = 0.5
    env.violation_penalty_factor = 10.0
    env.previous_position_distance = 0.0
    env.previous_primary_orientation_distance = 0.0
    env.previous_secondary_orientation_distance = 0.0
    env.previous_gripper_distance = 0.0
    return env


def make_pose(position, angles, gripper=0.0) -> Pose:
    return Pose.from_euler(position, angles, gripper, "XYZ", False)


def test_desired_pose_is_constructed_from_one_delta():
    env = make_env()
    chunk_start_pose = make_pose([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], 0.2)
    action = np.array([[0.1, -0.2, 0.3, 0.0, 0.0, np.pi / 2, 0.4]], dtype=np.float32)

    desired_pose = env._compute_desired_pose(chunk_start_pose, action)

    np.testing.assert_allclose(desired_pose.position, [1.1, 1.8, 3.3])
    np.testing.assert_allclose(desired_pose.gripper, 0.6)
    assert desired_pose.angular_distance(make_pose([0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2])) == 0.0


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
    env.tracking_deviation_enabled = False
    env.tracking_progress_enabled = False
    env.termination_penalty_enabled = False
    chunk_start_pose = make_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    cartesian_action_path = np.zeros((1, 7), dtype=np.float32)
    requested_action = {"motor": 1.0}
    safe_action = {"motor": 0.5}

    reward, breakdown = env.compute_reward(
        requested_action,
        safe_action,
        cartesian_action_path,
        chunk_start_pose,
        chunk_start_pose,
        chunk_terminated=False,
    )

    assert set(breakdown) == {"safety_penalty"}
    assert breakdown["safety_penalty"] == -5.0
    assert reward == sum(breakdown.values())


def test_compute_reward_updates_tracking_state_after_calculation():
    env = make_env()
    chunk_start_pose = make_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    current_pose = make_pose([0.5, 0.5, 0.0], [0.0, 0.0, 0.0])
    cartesian_action_path = np.zeros((1, 7), dtype=np.float32)

    env.compute_reward(
        {},
        {},
        cartesian_action_path,
        current_pose,
        chunk_start_pose,
        chunk_terminated=False,
    )

    assert env.previous_position_distance == np.sqrt(0.5)
    assert env.previous_primary_orientation_distance == 0.0
    assert env.previous_secondary_orientation_distance == 0.0
    assert env.previous_gripper_distance == 0.0


def test_first_step_progress_reward_uses_chunk_start_distance():
    env = make_env()
    chunk_start_pose = make_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    cartesian_action_path = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    desired_pose = env._compute_desired_pose(chunk_start_pose, cartesian_action_path)
    improved_pose = make_pose([0.05, 0.0, 0.0], [0.0, 0.0, 0.0])

    env.reset_chunk_reward_tracking(chunk_start_pose, cartesian_action_path)
    _, breakdown = env.compute_reward(
        {},
        {},
        cartesian_action_path,
        improved_pose,
        chunk_start_pose,
        chunk_terminated=False,
    )

    initial_distance = chunk_start_pose.positional_distance(desired_pose)
    final_distance = improved_pose.positional_distance(desired_pose)
    np.testing.assert_allclose(
        breakdown["position_reward"],
        (initial_distance - final_distance) / env.position_distance_scale,
    )


def test_terminal_penalty_is_sum_of_normalized_weighted_distances():
    env = make_env()
    current_pose = make_pose([0.1, 0.0, 0.0], [0.0, 0.0, 0.2], 0.5)
    desired_pose = make_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0)

    penalty = env._compute_termination_penalty(True, current_pose, desired_pose)

    np.testing.assert_allclose(penalty, -3.0)


def test_deviation_penalties_can_be_disabled_independently():
    env = make_env()
    env.tracking_deviation_enabled = False
    env.tracking_progress_enabled = False
    env.termination_penalty_enabled = False
    current_pose = make_pose([0.1, 0.0, 0.0], [0.0, 0.0, 0.0])

    reward, breakdown = env.compute_reward(
        {},
        {},
        np.zeros((1, 7), dtype=np.float32),
        current_pose,
        make_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        chunk_terminated=False,
    )

    assert reward == 0.0
    assert breakdown == {}


def test_reset_clears_tracking_state():
    env = make_env()
    env.arm = type("ArmStub", (), {})()
    env._get_obs = lambda: {}
    env.previous_position_distance = 2.0
    env.previous_primary_orientation_distance = 3.0
    env.previous_secondary_orientation_distance = 4.0
    env.previous_gripper_distance = 4.0

    env.reset()

    assert env.previous_position_distance == 0.0
    assert env.previous_primary_orientation_distance == 0.0
    assert env.previous_secondary_orientation_distance == 0.0
    assert env.previous_gripper_distance == 0.0
