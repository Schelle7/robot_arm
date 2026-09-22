from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from robot_arm.episode_runner import EpisodeRunner
from robot_arm.arms.communication import SafetyException
from robot_arm.control_types import CartesianAction, EnvironmentState
from robot_arm.geometry.pose import Pose
from robot_arm.policies.joint_observation import JointObservationBuilder
from robot_arm.recording.recorder import EpisodeRecorder
from robot_arm.robot_schema import HISTORY_FEATURE_NAMES, POLICY_OBSERVATION_NAMES, policy_observation_sizes

HISTORY_STEPS = 10


@pytest.mark.parametrize("from_sim_state", [False, True])
def test_safety_stop_is_saved_without_completed_transition(tmp_path, from_sim_state):
    cfg = SimpleNamespace(
        camera=SimpleNamespace(jpeg_quality=90),
        control=SimpleNamespace(frequencies=SimpleNamespace(joint=20, cartesian=5)),
        runtime=SimpleNamespace(record_sim_state=False, record_policy_debug=False, capture_camera=False),
    )
    runner = EpisodeRunner.__new__(EpisodeRunner)
    runner.recorder = EpisodeRecorder(str(tmp_path), cfg, "stopped")
    runner.primitive_policy = SimpleNamespace(task="pick_and_place")
    runner.env = SimpleNamespace(reset=Mock(), reset_from_sim_state=Mock())
    sample = {"Present_Temperature": {"wrist_roll": 70}, "sample_time_ns": 123}
    error = SafetyException("temperature limit", sample)
    runner._run_episode = Mock(side_effect=error)

    with pytest.raises(SafetyException) as caught:
        if from_sim_state:
            runner.run_episode_from_sim_state(False, np.zeros(6), np.zeros(6))
        else:
            runner.run_episode(False)

    assert caught.value is error
    sample["Present_Temperature"]["wrist_roll"] = 28
    with np.load(tmp_path / "stopped/episode.npz", allow_pickle=True) as data:
        event = data["safety_stops"][0]
        assert event["sensor_state"]["Present_Temperature"]["wrist_roll"] == 70
        assert event["sensor_state"]["sample_time_ns"] == 123
        assert event["reason"] == str(error)
        assert len(data["cartesian_action"]) == 0
        assert len(data["joint_positions"]) == 0
    assert runner.recorder.safety_stops[0]["sensor_state"]["Present_Temperature"]["wrist_roll"] == 70


def test_recording_keeps_teacher_labels_separate_from_executed_actions():
    runner = EpisodeRunner.__new__(EpisodeRunner)
    runner.cfg = SimpleNamespace(runtime=SimpleNamespace(record_sim_state=False))
    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.record_sim_state = False
    recorder.states = []
    recorder.transitions = []
    recorder.dense_trajectory_buffer = []
    recorder._make_state = lambda **kwargs: kwargs
    runner.recorder = recorder
    teacher = CartesianAction(np.full(7, 0.1, dtype=np.float32), {"teacher_completion_score": 0.75}, True, -0.35, True)
    teacher_inputs = []

    def get_teacher_action(**kwargs):
        teacher_inputs.append(kwargs)
        return teacher

    runner.teacher_policy = SimpleNamespace(get_action=get_teacher_action)
    state = SimpleNamespace(
        end_effector_pose=object(),
        observation={"gripper_duty": np.array([-0.35])},
        grasp_confirmed=True,
        sensor_state={},
    )
    primitive = SimpleNamespace(prompt="close gripper")
    executed = CartesianAction(np.full(7, -0.2), {}, False, 0.1, False)
    runner._record_transition(0, state, 0.0, {}, np.zeros(16), primitive, 0, executed)

    assert teacher_inputs[0]["state"] is state
    assert teacher_inputs[0]["primitive"] is primitive
    transition = recorder.transitions[0]
    np.testing.assert_array_equal(transition["cartesian_action"], executed.cartesian_action)
    assert transition["completes_active_primitive"] is False
    np.testing.assert_array_equal(transition["teacher_cartesian_action"], teacher.cartesian_action)
    assert transition["teacher_completes_active_primitive"] is True
    assert transition["teacher_completion_score"] == 0.75
    assert transition["desired_gripper_duty"] == 0.1
    assert transition["desired_gripper_duty_active"] is False
    assert transition["teacher_desired_gripper_duty"] == -0.35
    assert transition["teacher_desired_gripper_duty_active"] is True
    teacher.cartesian_action[:] = 0
    np.testing.assert_allclose(transition["teacher_cartesian_action"], 0.1)


def raw_observation(joint_position_value):
    return {
        "joint_positions": np.full(6, joint_position_value, dtype=np.float32),
        "joint_velocities": np.zeros(6, dtype=np.float32),
        "tcp_position": np.zeros(3, dtype=np.float32),
        "tcp_velocity": np.zeros(6, dtype=np.float32),
        "gripper_duty": np.zeros(1, dtype=np.float32),
        "policy_history": np.zeros((HISTORY_STEPS, len(HISTORY_FEATURE_NAMES)), dtype=np.float32),
    }


class JointPolicyStub:
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
        self.received_joint_steps = []
        self.pose = Pose.from_euler([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "XYZ", False)
        self.arm = SimpleNamespace(physics_metrics=lambda: {})
        self.policy_history_steps = HISTORY_STEPS

    def get_end_effector_pose(self):
        return self.pose

    def reset_cartesian_action_reward_tracking(self, cartesian_action_start_pose, cartesian_action):
        pass

    def step(
        self,
        action,
        state,
        policy_action,
        joint_step_idx,
        cartesian_action,
        cartesian_action_start_pose,
    ):
        self.received_actions.append(action)
        self.received_joint_steps.append(joint_step_idx)
        self.received_paths.append(cartesian_action.cartesian_action)
        return (
            EnvironmentState(
                observation=raw_observation(1.0),
                sensor_state={},
                end_effector_pose=self.pose,
                sim_state=None,
                grasp_confirmed=False,
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
        waypoint=SimpleNamespace(cartesian_action_dim=7),
    )


def make_runner(environment, joint_policy, metrics_queue, detailed_metrics=True):
    runner = EpisodeRunner.__new__(EpisodeRunner)
    runner.env = environment
    runner.joint_steps_per_cartesian_action = 2
    runner.joint_policy = joint_policy
    runner.training = True
    runner.recorder = None
    runner.replay_buffer = None
    runner.metrics_queue = metrics_queue
    runner.cfg = make_cfg(detailed_metrics)
    # Unit scales keep the normalized policy observation identical to the raw one.
    runner.joint_observation = JointObservationBuilder.__new__(JointObservationBuilder)
    runner.joint_observation.joint_velocity_scale = 1.0
    runner.joint_observation.tcp_velocity_scale = 1.0
    runner.joint_observation.cartesian_action_scale = 1.0
    runner.duty_limits = np.ones(6, dtype=np.float32)
    runner.joint_observation.policy_observation_sizes = policy_observation_sizes(7, HISTORY_STEPS)
    return runner


def test_cartesian_action_measures_every_policy_observation_against_one_desired_pose():
    cartesian_action = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.4, 0.5], dtype=np.float32)
    environment = EnvironmentStub({"joint_limit_penalty": -1.0})
    joint_policy = JointPolicyStub()
    runner = make_runner(environment, joint_policy, MetricsQueueStub())

    raw_obs = raw_observation(0.0)

    runner.execute_cartesian_action(
        EnvironmentState(
            observation=raw_obs,
            sensor_state={},
            end_effector_pose=environment.pose,
            sim_state=None,
            grasp_confirmed=False,
        ),
        CartesianAction(cartesian_action, {}, False, 0.0, False),
    )

    assert len(joint_policy.observations) == runner.joint_steps_per_cartesian_action
    assert environment.received_joint_steps == [1, 2]
    # The stub never moves, so the delta still to travel must stay the one the action asked for.
    for observation in joint_policy.observations:
        np.testing.assert_allclose(observation["goal"][:7], cartesian_action, atol=1e-6)
    for received_path in environment.received_paths:
        assert received_path is cartesian_action


def test_joint_recording_uses_consecutive_states():
    environment = EnvironmentStub({"joint_limit_penalty": 0.0})
    runner = make_runner(environment, JointPolicyStub(), MetricsQueueStub())
    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.dense_trajectory_buffer = []
    recorder.joint_steps_per_cartesian_action = runner.joint_steps_per_cartesian_action
    runner.recorder = recorder
    runner.cfg.runtime.record_policy_debug = True
    initial_pose = Pose.from_euler([0.1, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "XYZ", False)

    runner.execute_cartesian_action(
        EnvironmentState(raw_observation(0.0), {}, initial_pose, None, False),
        CartesianAction(np.zeros(7, dtype=np.float32), {}, False, 0.0, False),
    )

    first, second = recorder.dense_trajectory_buffer
    np.testing.assert_array_equal(first["end_effector_pose"], initial_pose.as_10d())
    np.testing.assert_array_equal(first["next_end_effector_pose"], environment.pose.as_10d())
    np.testing.assert_array_equal(second["end_effector_pose"], first["next_end_effector_pose"])


def test_policy_observation_keys_match_declared_observation_space():
    cartesian_action = np.zeros(7, dtype=np.float32)
    environment = EnvironmentStub({"joint_limit_penalty": -1.0})
    joint_policy = JointPolicyStub()
    runner = make_runner(environment, joint_policy, MetricsQueueStub())

    raw_obs = raw_observation(0.0)
    runner.execute_cartesian_action(
        EnvironmentState(
            observation=raw_obs,
            sensor_state={},
            end_effector_pose=environment.pose,
            sim_state=None,
            grasp_confirmed=False,
        ),
        CartesianAction(cartesian_action, {}, False, 0.0, False),
    )

    assert set(joint_policy.observations[0]) == set(POLICY_OBSERVATION_NAMES)


def test_latest_joint_positions_anchor_history_and_remain_in_current_state():
    runner = make_runner(EnvironmentStub({"joint_limit_penalty": 0.0}), JointPolicyStub(), MetricsQueueStub())
    observation = raw_observation(0.25)

    policy_observation = runner.joint_observation.build(
        EnvironmentState(observation, {}, runner.env.pose, None, False),
        CartesianAction(np.zeros(7, dtype=np.float32), {}, False, 0.0, False),
        runner.env.pose,
        0.0,
    )

    np.testing.assert_array_equal(policy_observation["history"][:6], observation["joint_positions"])
    np.testing.assert_array_equal(policy_observation["state"][:6], observation["joint_positions"])


def test_direct_duty_scales_policy_action_by_motor_limits():
    environment = EnvironmentStub({"joint_limit_penalty": 0.0})
    runner = make_runner(environment, JointPolicyStub(), MetricsQueueStub())
    runner.duty_limits = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.5], dtype=np.float32)
    runner.joint_policy.predict = lambda observation, deterministic: (np.array([0.25, -0.25, 1.0, -1.0, 0.0, -0.8], dtype=np.float32), None)
    raw_obs = raw_observation(0.0)

    runner.execute_cartesian_action(
        EnvironmentState(raw_obs, {}, environment.pose, None, False),
        CartesianAction(np.zeros(7, dtype=np.float32), {}, False, 0.0, False),
    )

    for action in environment.received_actions:
        np.testing.assert_allclose(action, [0.25, -0.25, 1.0, -1.0, 0.0, -0.4])


def test_detailed_metrics_only_include_returned_reward_components():
    reward_breakdown = {"joint_limit_penalty": -2.0}
    environment = EnvironmentStub(reward_breakdown)
    joint_policy = JointPolicyStub()
    metrics_queue = MetricsQueueStub()
    runner = make_runner(environment, joint_policy, metrics_queue)

    raw_obs = raw_observation(0.0)
    runner.execute_cartesian_action(
        EnvironmentState(
            observation=raw_obs,
            sensor_state={},
            end_effector_pose=environment.pose,
            sim_state=None,
            grasp_confirmed=False,
        ),
        CartesianAction(np.zeros(7, dtype=np.float32), {}, False, 0.0, False),
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


@pytest.mark.parametrize(
    "angle,holding,too_far,failed",
    [
        (0.299, True, False, True),
        (0.3, True, False, False),
        (0.3415, True, False, False),
        (0.3415, True, True, True),
        (0.0, False, True, False),
    ],
)
def test_grip_abort_checks_angle_and_box_distance_only_while_holding(angle, holding, too_far, failed):
    runner = EpisodeRunner.__new__(EpisodeRunner)
    runner.env = SimpleNamespace(box_too_far=lambda pose: too_far)
    runner.cfg = SimpleNamespace(
        waypoint=SimpleNamespace(
            pick_and_place=SimpleNamespace(gripper_abort_below_radians=0.3),
        )
    )
    state = SimpleNamespace(end_effector_pose=SimpleNamespace(gripper=angle))
    primitive = SimpleNamespace(desired_gripper_duty_active=holding)

    assert runner.grip_failed(state, primitive) == failed


@pytest.mark.parametrize(
    "flags,angles,too_far,expected_steps,aborted",
    [
        ([False, False], [0.4, 0.4], False, 1, False),
        ([False, False], [0.4, 0.299], False, 1, True),
        ([True, False], [0.4, 0.4], False, 1, False),
        ([True, False], [0.4, 0.4], True, 1, True),
    ],
)
def test_runner_preserves_policy_completion_and_aborts_failed_grips(flags, angles, too_far, expected_steps, aborted):
    runner = EpisodeRunner.__new__(EpisodeRunner)
    runner.progress = None
    runner.env = SimpleNamespace(box_too_far=lambda pose: too_far)
    runner.cfg = SimpleNamespace(
        waypoint=SimpleNamespace(pick_and_place=SimpleNamespace(gripper_abort_below_radians=0.3)),
        runtime=SimpleNamespace(capture_camera=False),
    )
    runner.max_cartesian_steps = 10
    runner.training = False
    runner.primitive_policy = SimpleNamespace(build_vla_input_state=lambda *args: np.zeros(16))
    runner.cartesian_policy = SimpleNamespace(
        get_action=lambda **kwargs: CartesianAction(np.zeros(7), {}, True, -0.35, True),
    )
    runner._draw_desired_path = lambda *args: None
    recorded_actions = []
    runner._record_transition = lambda *args: recorded_actions.append(args[-1])
    states = [
        SimpleNamespace(
            end_effector_pose=SimpleNamespace(gripper=angle),
            observation={"gripper_duty": np.array([-0.35])},
            grasp_confirmed=flag,
        )
        for angle, flag in zip(angles, flags)
    ]
    remaining_states = iter(states[1:])
    runner.execute_cartesian_action = lambda *args: (next(remaining_states), 0.0)
    primitive = SimpleNamespace(
        prompt="close gripper",
        desired_gripper_duty=-0.35,
        desired_gripper_duty_active=True,
    )

    final_state, completed_steps, truncated = runner.execute_primitive(states[0], primitive, 0, 0)

    assert completed_steps == expected_steps
    assert truncated == aborted
    assert final_state is states[-1]
    assert recorded_actions[-1].completes_active_primitive is True
    assert recorded_actions[-1].diagnostics == {}
