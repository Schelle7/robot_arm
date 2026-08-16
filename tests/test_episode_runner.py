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
        self.pose = Pose.from_euler([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "XYZ", False)

    def get_privileged_end_effector_pose(self):
        return self.pose

    def step(self, action, cartesian_action_path, chunk_start_pose, chunk_terminated):
        self.received_paths.append(cartesian_action_path)
        return (
            EnvironmentState(
                observation={
                    "joint_positions": np.ones(6, dtype=np.float32),
                    "joint_velocities": np.zeros(6, dtype=np.float32),
                },
                sensor_state={},
                privileged_state={"end_effector_pose": self.pose},
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
            frequencies=SimpleNamespace(mid_level=2, low_level=4),
            max_seconds=1,
        ),
        runtime=SimpleNamespace(draw_waypoints=False, draw_tcp=False),
        training=SimpleNamespace(
            detailed_metrics=detailed_metrics,
            pose_delta_diagnostics_enabled=True,
            sync_weights_every_n_chunks=2,
        ),
    )


def make_runner(environment, low_level_policy, metrics_queue, detailed_metrics=True):
    runner = EpisodeRunner.__new__(EpisodeRunner)
    runner.env = environment
    runner.chunk_size = 2
    runner.low_level_policy = low_level_policy
    runner.training = True
    runner.recorder = None
    runner.replay_buffer = None
    runner.metrics_queue = metrics_queue
    runner.cfg = make_cfg(detailed_metrics)
    runner.episode_low_level_step = 0
    return runner


def test_run_chunk_reuses_exact_desired_path_for_every_policy_observation():
    cartesian_action_path = np.arange(14, dtype=np.float32).reshape(2, 7)
    environment = EnvironmentStub({"safety_penalty": -1.0})
    low_level_policy = LowLevelPolicyStub()
    runner = make_runner(environment, low_level_policy, MetricsQueueStub())

    raw_obs = {
        "joint_positions": np.zeros(6, dtype=np.float32),
        "joint_velocities": np.zeros(6, dtype=np.float32),
    }

    runner.run_chunk(
        EnvironmentState(
            observation=raw_obs,
            sensor_state={},
            privileged_state={"end_effector_pose": environment.pose},
        ),
        cartesian_action_path,
    )

    assert len(low_level_policy.observations) == runner.chunk_size
    for observation in low_level_policy.observations:
        assert observation["cartesian_action_path"] is cartesian_action_path
    for received_path in environment.received_paths:
        assert received_path is cartesian_action_path


def test_policy_observation_keys_match_declared_observation_space():
    cfg = SimpleNamespace(waypoint=SimpleNamespace(trajectory_length=2, trajectory_dim=7))
    observation_space = DummySpaceEnv(cfg).observation_space
    cartesian_action_path = np.zeros((2, 7), dtype=np.float32)
    environment = EnvironmentStub({"safety_penalty": -1.0})
    low_level_policy = LowLevelPolicyStub()
    runner = make_runner(environment, low_level_policy, MetricsQueueStub())

    raw_obs = {
        "joint_positions": np.zeros(6, dtype=np.float32),
        "joint_velocities": np.zeros(6, dtype=np.float32),
    }
    runner.run_chunk(
        EnvironmentState(
            observation=raw_obs,
            sensor_state={},
            privileged_state={"end_effector_pose": environment.pose},
        ),
        cartesian_action_path,
    )

    assert set(low_level_policy.observations[0]) == set(observation_space)


def test_detailed_metrics_only_include_returned_reward_components():
    reward_breakdown = {"safety_penalty": -2.0}
    environment = EnvironmentStub(reward_breakdown)
    low_level_policy = LowLevelPolicyStub()
    metrics_queue = MetricsQueueStub()
    runner = make_runner(environment, low_level_policy, metrics_queue)

    raw_obs = {
        "joint_positions": np.zeros(6, dtype=np.float32),
        "joint_velocities": np.zeros(6, dtype=np.float32),
    }
    runner.run_chunk(
        EnvironmentState(
            observation=raw_obs,
            sensor_state={},
            privileged_state={"end_effector_pose": environment.pose},
        ),
        np.zeros((2, 10), dtype=np.float32),
    )

    assert len(metrics_queue.items) == 1
    assert metrics_queue.items[0] == {
        "total_reward": -4.0,
        "safety_penalty": [-2.0, -2.0],
        "moved_delta_norm": [0.2, 0.2],
        "desired_delta_norm": [0.5, 0.5],
        "delta_error_norm": [0.3, 0.3],
    }


def test_weights_sync_at_configured_chunk_interval():
    runner = EpisodeRunner.__new__(EpisodeRunner)
    runner.cfg = make_cfg()
    runner.training_chunk_count = 0
    sync_calls = []
    runner._sync_weights = lambda: sync_calls.append(runner.training_chunk_count)

    runner._sync_weights_if_due()
    runner._sync_weights_if_due()
    runner._sync_weights_if_due()
    runner._sync_weights_if_due()

    assert sync_calls == [2, 4]
