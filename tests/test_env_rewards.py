import numpy as np

from robot_arm.envs.env import RobotEnv
from robot_arm.pose import Pose


def make_env() -> RobotEnv:
    env = RobotEnv.__new__(RobotEnv)
    env.tracking_deviation_enabled = True
    env.tracking_progress_enabled = True
    env.safety_penalty_enabled = True
    env.termination_penalty_enabled = True
    env.violation_penalty_factor = 10.0
    env.position_distance_weights = np.ones(3, dtype=np.float32)
    env.rotation_distance_weights = np.ones(3, dtype=np.float32)
    env.gripper_distance_weights = np.ones(1, dtype=np.float32)
    env.previous_deviation = 0.0
    env.previous_progress = 0.0
    return env


def make_pose(position, angles, gripper=0.0) -> Pose:
    return Pose.from_euler(position, angles, gripper, "XYZ", False)


def test_projection_clamps_before_path_origin():
    env = make_env()
    path = np.array([[0.0, 0.0], [2.0, 0.0]])

    closest_point, segment_idx, t = env._get_closest_path_point(
        np.array([-1.0, 1.0]), path
    )

    np.testing.assert_allclose(closest_point, [0.0, 0.0])
    assert segment_idx == 0
    assert t == 0.0


def test_projection_clamps_after_path_endpoint():
    env = make_env()
    path = np.array([[0.0, 0.0], [2.0, 0.0]])

    closest_point, segment_idx, t = env._get_closest_path_point(
        np.array([3.0, 1.0]), path
    )

    np.testing.assert_allclose(closest_point, [2.0, 0.0])
    assert segment_idx == 0
    assert t == 1.0


def test_projection_selects_closest_segment_at_corner():
    env = make_env()
    path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])

    closest_point, segment_idx, t = env._get_closest_path_point(
        np.array([0.9, 0.8]), path
    )

    np.testing.assert_allclose(closest_point, [1.0, 0.8])
    assert segment_idx == 1
    assert t == 0.8


def test_projection_handles_duplicate_waypoint():
    env = make_env()
    path = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])

    closest_point, segment_idx, t = env._get_closest_path_point(
        np.array([0.5, 0.2]), path
    )

    np.testing.assert_allclose(closest_point, [0.5, 0.0])
    assert segment_idx == 1
    assert t == 0.5


def test_path_progress_at_endpoint_is_total_path_length():
    env = make_env()
    path = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])

    progress = env._compute_path_progress(path, segment_idx=1, t=1.0)

    assert progress == 7.0


def test_rotation_reward_uses_three_dimensional_rotation_vector():
    env = make_env()
    chunk_start_pose = make_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    current_pose = make_pose([0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2])
    high_level_action = np.zeros((1, 10), dtype=np.float32)
    high_level_action[0, 3:9] = current_pose.as_rot_6d() - chunk_start_pose.as_rot_6d()

    weighted_pose_delta, weighted_delta_path = (
        env._compute_weighted_pose_delta_and_path(
            current_pose, chunk_start_pose, high_level_action
        )
    )

    np.testing.assert_allclose(weighted_pose_delta[:3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        np.linalg.norm(weighted_pose_delta[3:6]), np.pi / 2, atol=1e-6
    )
    assert weighted_pose_delta.shape == (7,)
    assert weighted_delta_path.shape == (2, 7)
    np.testing.assert_allclose(weighted_delta_path[0], np.zeros(7))
    np.testing.assert_allclose(
        np.linalg.norm(weighted_delta_path[1, 3:6]), np.pi / 2, atol=1e-6
    )


def test_rotation_reward_handles_180_degree_rotation():
    env = make_env()
    chunk_start_pose = make_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    current_pose = make_pose([0.0, 0.0, 0.0], [np.pi, 0.0, 0.0])
    high_level_action = np.zeros((1, 10), dtype=np.float32)
    high_level_action[0, 3:9] = current_pose.as_rot_6d() - chunk_start_pose.as_rot_6d()

    weighted_pose_delta, weighted_delta_path = (
        env._compute_weighted_pose_delta_and_path(
            current_pose, chunk_start_pose, high_level_action
        )
    )

    np.testing.assert_allclose(
        np.linalg.norm(weighted_pose_delta[3:6]), np.pi, atol=1e-6
    )
    np.testing.assert_allclose(
        np.linalg.norm(weighted_delta_path[1, 3:6]), np.pi, atol=1e-6
    )


def test_compute_reward_filters_disabled_components_and_sums_breakdown():
    env = make_env()
    env.tracking_deviation_enabled = False
    env.tracking_progress_enabled = False
    env.termination_penalty_enabled = False
    chunk_start_pose = make_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    high_level_action = np.zeros((1, 10), dtype=np.float32)
    requested_action = {"motor": 1.0}
    safe_action = {"motor": 0.5}

    reward, breakdown = env.compute_reward(
        requested_action,
        safe_action,
        high_level_action,
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
    high_level_action = np.zeros((1, 10), dtype=np.float32)

    env.compute_reward(
        {},
        {},
        high_level_action,
        current_pose,
        chunk_start_pose,
        chunk_terminated=False,
    )

    assert env.previous_deviation == np.sqrt(0.5)
    assert env.previous_progress == 0.0


def test_reset_clears_tracking_state():
    env = make_env()
    env.arm = type("ArmStub", (), {})()
    env._get_obs = lambda: {}
    env.previous_deviation = 2.0
    env.previous_progress = 3.0

    env.reset()

    assert env.previous_deviation == 0.0
    assert env.previous_progress == 0.0


def test_tracking_rewards_return_current_state_without_mutating_it():
    env = make_env()
    path = np.array([[0.0, 0.0], [2.0, 0.0]])

    dev_reward, prog_reward, deviation, progress = env._compute_tracking_rewards(
        np.array([1.0, 0.0]), path
    )

    assert dev_reward == 0.0
    assert prog_reward == 1.0
    assert deviation == 0.0
    assert progress == 1.0
    assert env.previous_deviation == 0.0
    assert env.previous_progress == 0.0