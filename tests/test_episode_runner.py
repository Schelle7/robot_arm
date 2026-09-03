from types import SimpleNamespace

import numpy as np

from robot_arm.distributed import DummySpaceEnv
from robot_arm.episode_runner import EpisodeRunner
from robot_arm.envs.env import EnvironmentState
from robot_arm.pose import Pose


class LowLevelPolicyStub:
    def __init__(self):
        self.observations = []

    def predict(self, observation, deterministic):
        self.observations.append(observation)
        return np.zeros(6, dtype=np.float32), None


class EnvironmentStub:
    def __init__(self, reward_breakdown):
        self.reward_breakdown = reward_breakdown
        self.pose_delta_diagnostics = {
            "moved_delta_norm": 0.2,
            "desired_delta_norm": 0.5,
            "delta_error_norm": 0.3,
        }
        self.received_paths = []
        self.received_actions = []
        self.pose = Pose.from_euler([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "XYZ", False)

    def get_end_effector_pose(self):
        return self.pose

    def reset_cartesian_action_reward_tracking(self, cartesian_action_start_pose, cartesian_action):
        pass

    def step(self, action, joint_positions, cartesian_action, cartesian_action_start_pose, cartesian_action_terminated, desired_gripper_duty, desired_gripper_duty_active):
        self.received_actions.append(action)
        self.received_paths.append(cartesian_action)
        return (
            EnvironmentState(
                observation={
                    "joint_positions": np.ones(6, dtype=np.float32),
                    "joint_velocities": np.zeros(6, dtype=np.float32),
                    "gripper_duty": np.zeros(1, dtype=np.float32),
                    "tcp_velocity": np.zeros(6, dtype=np.float32),
                },
                sensor_state={},
                end_effector_pose=self.pose,
                sim_state=None,
            ),
            float(sum(self.reward_breakdown.values())),
            self.reward_breakdown,
        )


class MetricsQueueStub:
    def __init__(self):
        self.items = []

    def add(self, item):
        self.items.append(item)


def make_cfg(detailed_metrics=True):
    return SimpleNamespace(
        control=SimpleNamespace(
            frequencies=SimpleNamespace(cartesian=2, joint=4),
            max_seconds=1,
        ),
        runtime=SimpleNamespace(draw_waypoints=False, draw_tcp=False),
        training=SimpleNamespace(
            detailed_metrics=detailed_metrics,
            pose_delta_diagnostics_enabled=True,
            sync_weights_every_n_cartesian_actions=2,
            terminate_at_cartesian_action_end=True,
        ),
    )


def make_runner(environment, low_level_policy, metrics_queue, detailed_metrics=True):
    runner = EpisodeRunner.__new__(EpisodeRunner)
    runner.env = environment
    runner.joint_steps_per_cartesian_action = 2
    runner.low_level_policy = low_level_policy
    runner.training = True
    runner.recorder = None
    runner.replay_buffer = None
    runner.metrics_queue = metrics_queue
    runner.cfg = make_cfg(detailed_metrics)
    runner.episode_low_level_step = 0
    # Unit scales keep the normalized policy observation identical to the raw one.
    runner.joint_velocity_scale = 1.0
    runner.tcp_velocity_scale = 1.0
    runner.cartesian_action_scale = 1.0
    runner.duty_limits = np.ones(6, dtype=np.float32)
    runner.duty_compensator = SimpleNamespace(calculate=lambda positions, velocities: np.zeros(6, dtype=np.float32))
    return runner


def test_cartesian_action_measures_every_policy_observation_against_one_desired_pose():
    cartesian_action = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.4, 0.5], dtype=np.float32)
    environment = EnvironmentStub({"joint_limit_penalty": -1.0})
    low_level_policy = LowLevelPolicyStub()
    runner = make_runner(environment, low_level_policy, MetricsQueueStub())

    raw_obs = {
        "joint_positions": np.zeros(6, dtype=np.float32),
        "joint_velocities": np.zeros(6, dtype=np.float32),
        "gripper_duty": np.zeros(1, dtype=np.float32),
        "tcp_velocity": np.zeros(6, dtype=np.float32),
    }

    runner.execute_cartesian_action(
        EnvironmentState(
            observation=raw_obs,
            sensor_state={},
            end_effector_pose=environment.pose,
            sim_state=None,
        ),
        cartesian_action,
        0.0,
        False,
    )

    assert len(low_level_policy.observations) == runner.joint_steps_per_cartesian_action
    # The stub never moves, so the delta still to travel must stay the one the action asked for.
    for observation in low_level_policy.observations:
        np.testing.assert_allclose(observation["remaining_delta"], cartesian_action, atol=1e-6)
    for received_path in environment.received_paths:
        assert received_path is cartesian_action


def test_policy_observation_keys_match_declared_observation_space():
    cfg = SimpleNamespace(waypoint=SimpleNamespace(cartesian_action_dim=7))
    observation_space = DummySpaceEnv(cfg).observation_space
    cartesian_action = np.zeros(7, dtype=np.float32)
    environment = EnvironmentStub({"joint_limit_penalty": -1.0})
    low_level_policy = LowLevelPolicyStub()
    runner = make_runner(environment, low_level_policy, MetricsQueueStub())

    raw_obs = {
        "joint_positions": np.zeros(6, dtype=np.float32),
        "joint_velocities": np.zeros(6, dtype=np.float32),
        "gripper_duty": np.zeros(1, dtype=np.float32),
        "duty_history": np.zeros(6, dtype=np.float32),
        "tcp_velocity": np.zeros(6, dtype=np.float32),
    }
    runner.execute_cartesian_action(
        EnvironmentState(
            observation=raw_obs,
            sensor_state={},
            end_effector_pose=environment.pose,
            sim_state=None,
        ),
        cartesian_action,
        0.0,
        False,
    )

    assert set(low_level_policy.observations[0]) == set(observation_space)


def test_duty_compensation_excludes_gripper():
    environment = EnvironmentStub({"joint_limit_penalty": 0.0})
    runner = make_runner(environment, LowLevelPolicyStub(), MetricsQueueStub())
    runner.duty_compensator.calculate = lambda positions, velocities: np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.0], dtype=np.float32)
    raw_obs = {
        "joint_positions": np.zeros(6, dtype=np.float32),
        "joint_velocities": np.zeros(6, dtype=np.float32),
        "gripper_duty": np.zeros(1, dtype=np.float32),
        "duty_history": np.zeros(6, dtype=np.float32),
        "tcp_velocity": np.zeros(6, dtype=np.float32),
    }

    runner.execute_cartesian_action(
        EnvironmentState(raw_obs, {}, environment.pose, None),
        np.zeros(7, dtype=np.float32),
        0.0,
        False,
    )

    for action in environment.received_actions:
        np.testing.assert_allclose(action, [0.25, 0.25, 0.25, 0.25, 0.25, 0.0])


def test_detailed_metrics_only_include_returned_reward_components():
    reward_breakdown = {"joint_limit_penalty": -2.0}
    environment = EnvironmentStub(reward_breakdown)
    low_level_policy = LowLevelPolicyStub()
    metrics_queue = MetricsQueueStub()
    runner = make_runner(environment, low_level_policy, metrics_queue)

    raw_obs = {
        "joint_positions": np.zeros(6, dtype=np.float32),
        "joint_velocities": np.zeros(6, dtype=np.float32),
        "gripper_duty": np.zeros(1, dtype=np.float32),
        "duty_history": np.zeros(6, dtype=np.float32),
        "tcp_velocity": np.zeros(6, dtype=np.float32),
    }
    runner.execute_cartesian_action(
        EnvironmentState(
            observation=raw_obs,
            sensor_state={},
            end_effector_pose=environment.pose,
            sim_state=None,
        ),
        np.zeros(10, dtype=np.float32),
        0.0,
        False,
    )

    assert len(metrics_queue.items) == 1
    assert metrics_queue.items[0] == {
        "total_reward": -4.0,
        "joint_limit_penalty": [-2.0, -2.0],
        "moved_delta_norm": [0.2, 0.2],
        "desired_delta_norm": [0.5, 0.5],
        "delta_error_norm": [0.3, 0.3],
    }


def test_weights_sync_at_configured_cartesian_action_interval():
    runner = EpisodeRunner.__new__(EpisodeRunner)
    runner.cfg = make_cfg()
    runner.training_cartesian_action_count = 0
    sync_calls = []
    runner._sync_weights = lambda: sync_calls.append(runner.training_cartesian_action_count)

    runner._sync_weights_if_due()
    runner._sync_weights_if_due()
    runner._sync_weights_if_due()
    runner._sync_weights_if_due()

    assert sync_calls == [2, 4]
