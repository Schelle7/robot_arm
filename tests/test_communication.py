import json
import time
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import mujoco
import numpy as np
from omegaconf import OmegaConf

from robot_arm.arms.communication import Communication
from robot_arm.arms.real_communication import RealCommunication
from robot_arm.arms.sim_communication import SimCommunication


@pytest.fixture
def cfg():
    config_dir = Path(__file__).resolve().parents[1] / "conf"
    return OmegaConf.create(
        {
            "model_path": str(config_dir.parent / "models/so101/scene.xml"),
            "safety": OmegaConf.load(config_dir / "safety/default.yaml"),
            "control": OmegaConf.load(config_dir / "control/default.yaml"),
            "servo": OmegaConf.load(config_dir / "servo/default.yaml"),
        }
    )


@pytest.fixture
def sim(cfg):
    model = mujoco.MjModel.from_xml_path(cfg.model_path)
    return SimCommunication(cfg, model, mujoco.MjData(model))


def test_sim_communication_reads_checks_and_sends_synchronously(sim):
    state = {"sample_time_ns": 123, "feedback_age_seconds": 0.0}
    calls = Mock()
    calls.read.return_value = state
    communication = sim
    communication.read_sensors = calls.read
    communication.send_duty = calls.send
    communication.check_safety = calls.check
    assert isinstance(communication, Communication)
    assert communication.get_state() is state
    communication.submit_duty({"gripper": 0.2}, 123)
    assert [call[0] for call in calls.mock_calls] == ["read", "check", "send"]
    calls.check.assert_called_once_with(state)
    calls.send.assert_called_once_with({"gripper": 0.2})
    communication.close()


def test_sim_communication_does_not_return_unsafe_state(sim):
    check = Mock(side_effect=RuntimeError("temperature limit"))
    communication = sim
    communication.check_safety = check
    with pytest.raises(RuntimeError, match="temperature limit"):
        communication.get_state()


def test_sim_sensors_report_physics_state_and_clipped_duty(sim, cfg):
    joint_id = sim.model.joint("gripper").id
    sim.data.qpos[sim.model.jnt_qposadr[joint_id]] = 0.4
    sim.data.qvel[sim.model.jnt_dofadr[joint_id]] = -0.2
    sim.data.time = 0.15
    sim.submit_duty({"gripper": 1.0}, 100_000_000)

    state = sim.get_state()

    assert state["Present_Position"]["gripper"] == pytest.approx(0.4)
    assert state["Present_Velocity"]["gripper"] == pytest.approx(-0.2)
    assert state["Present_Load"]["gripper"] == pytest.approx(cfg.servo.max_duty.gripper / cfg.servo.full_scale_duty)
    assert state["sample_time_ns"] == state["read_completed_ns"] == 150_000_000
    assert state["feedback_age_seconds"] == 0.0


def test_sim_held_duty_torque_decreases_as_motor_accelerates(sim, cfg):
    sim.submit_duty({"gripper": 1.0}, 0)
    sim.apply_servo_torques()
    actuator_id = sim.model.actuator("gripper").id
    initial_torque = sim.data.ctrl[actuator_id]
    assert initial_torque > 0

    joint_id = sim.model.joint("gripper").id
    duty_fraction = cfg.servo.max_duty.gripper / cfg.servo.full_scale_duty
    sim.data.qvel[sim.model.jnt_dofadr[joint_id]] = duty_fraction * cfg.servo.no_load_speed_radians_per_second
    sim.apply_servo_torques()

    assert sim.data.ctrl[actuator_id] == pytest.approx(0.0)
    assert sim.read_sensors()["Present_Load"]["gripper"] == pytest.approx(duty_fraction)


def test_sim_reset_clears_command_and_safety_history(sim):
    sim.submit_duty({"gripper": 0.5}, 0)
    sim.get_state()
    assert sim.smoothed_duties["gripper"] > 0

    sim.reset()

    np.testing.assert_array_equal(sim.commanded_duty, 0.0)
    assert sim.last_safety_read_ns is None
    assert all(value == 0 for value in sim.smoothed_duties.values())


@pytest.fixture
def worker(tmp_path, monkeypatch, cfg):
    cfg.hardware = {"port": "/unused", "calibration_id": "test", "read_timeout_seconds": 0.01, "max_feedback_age_seconds": 1.0, "read_history_size": 3}
    port = SimpleNamespace(setPacketTimeout=Mock(), setPacketTimeoutMillis=Mock())
    follower = SimpleNamespace(bus=SimpleNamespace(port_handler=port, calibration=True), connect=Mock())
    monkeypatch.setattr("robot_arm.arms.real_communication.SO101Follower", lambda cfg: follower)
    worker = RealCommunication(cfg, {})
    worker.check_safety = Mock()
    worker.setup = Mock()
    worker.shutdown = Mock()
    worker.read_sensors = Mock()
    worker.send_duty = Mock()
    worker.log_path = tmp_path / "communication.jsonl"
    return worker


def test_short_timeout_applies_only_after_setup(worker, monkeypatch):
    port = worker.bus.port_handler
    original_timeout = port.setPacketTimeout

    def check_setup_timeout(*args, **kwargs):
        assert port.setPacketTimeout is original_timeout

    worker.follower.connect.side_effect = check_setup_timeout
    monkeypatch.setattr("robot_arm.arms.real_communication.read_configuration", check_setup_timeout)
    worker._set_pwm_mode = Mock(side_effect=check_setup_timeout)
    RealCommunication.setup(worker)
    worker.follower.connect.assert_called_once_with(calibrate=True)
    worker._set_pwm_mode.assert_called_once_with()
    port.setPacketTimeout(126)
    port.setPacketTimeout(21)
    assert [call.args for call in port.setPacketTimeoutMillis.call_args_list] == [(10.0,), (10.0,)]


def test_setup_precedes_commands_and_sensor_reads(worker):
    calls = []
    worker.setup = Mock(side_effect=lambda: calls.append("setup"))
    worker.send_duty = Mock(side_effect=lambda duties: calls.append("send"))
    worker.pending_command = {"duties": {"gripper": 0.2}, "submitted_ns": 0, "sample_time_ns": 0}

    def read():
        calls.append("read")
        worker.stop_requested.set()
        raise ConnectionError("missing response")

    worker.read_sensors = read
    worker._run()
    assert calls == ["setup", "send", "read"]
    worker.setup.assert_called_once_with()
    worker.shutdown.assert_called_once_with()


def test_setup_failure_shuts_down_without_reading_or_sending(worker):
    error = RuntimeError("PWM mode rejected")
    worker.setup = Mock(side_effect=error)
    with pytest.raises(RuntimeError, match="PWM mode rejected"):
        worker._run()
    worker.read_sensors.assert_not_called()
    worker.send_duty.assert_not_called()
    worker.shutdown.assert_called_once_with()
    assert worker.error is error
    assert worker.ready.is_set()


def test_cached_state_is_available_while_read_blocks_and_commands_replace_each_other(worker):
    blocked = Event()
    release = Event()
    sent = Event()
    now = time.perf_counter_ns()
    state = {"read_completed_ns": now, "sample_time_ns": now, "Present_Load": {"gripper": 0.2}}
    calls = 0

    def read():
        nonlocal calls
        calls += 1
        if calls == 1:
            return state
        blocked.set()
        assert release.wait(2)
        raise ConnectionError("missing response")

    def send(duties):
        assert duties == {"gripper": 0.3}
        sent.set()
        worker.stop_requested.set()

    worker.read_sensors = read
    worker.send_duty = Mock(side_effect=send)
    try:
        worker.start(worker.log_path)
        assert blocked.wait(2)
        snapshot = worker.get_state()
        snapshot["Present_Load"]["gripper"] = 0.9
        assert worker.get_state()["Present_Load"]["gripper"] == 0.2
        assert snapshot["sample_time_ns"] == now
        worker.submit_duty({"gripper": 0.1}, now)
        worker.submit_duty({"gripper": 0.3}, now)
        release.set()
        assert sent.wait(2)
    finally:
        release.set()
        worker.close()

    worker.send_duty.assert_called_once()
    worker.shutdown.assert_called_once()
    assert [attempt["success"] for attempt in worker.read_attempts] == [True, False]
    worker.check_safety.assert_called_once_with(state)
    events = [json.loads(line) for line in worker.log_path.read_text().splitlines()]
    assert len([event for event in events if event["event"] == "command_sent"]) == 1
    assert len([event for event in events if event["event"] == "feedback_used"]) == 2


@pytest.mark.parametrize("late_success", [False, True])
def test_initial_feedback_timeout_stops_and_is_propagated(worker, monkeypatch, late_success):
    clock = [0]
    monkeypatch.setattr("robot_arm.arms.real_communication.time.perf_counter_ns", lambda: clock[0])

    def read():
        clock[0] += 500_000_000
        if late_success and clock[0] == 1_000_000_000:
            return {"read_completed_ns": clock[0], "sample_time_ns": clock[0]}
        raise ConnectionError("missing response")

    worker.read_sensors = read
    worker.started = True
    with pytest.raises(TimeoutError):
        worker._run()
    with pytest.raises(TimeoutError):
        worker.get_state()
    assert worker.latest_state is None
    assert worker.last_successful_read_ns is None
    assert worker.ready.is_set()
    worker.shutdown.assert_called_once()


def test_feedback_loss_stops_without_policy_calls_and_history_is_bounded(worker, monkeypatch):
    clock = [0]
    monkeypatch.setattr("robot_arm.arms.real_communication.time.perf_counter_ns", lambda: clock[0])

    def read():
        clock[0] += 250_000_000
        if clock[0] == 250_000_000:
            return {"read_completed_ns": clock[0], "sample_time_ns": clock[0]}
        raise ConnectionError("missing response")

    worker.read_sensors = read
    with pytest.raises(TimeoutError):
        worker._run()
    assert worker.last_successful_read_ns == 250_000_000
    assert len(worker.read_attempts) == 3
    assert worker.read_attempts[0]["completed_ns"] == 750_000_000
    worker.shutdown.assert_called_once()


def test_stop_requested_during_read_prevents_pending_command(worker):
    def request_stop():
        worker.pending_command = {"duties": {"gripper": 0.2}, "submitted_ns": 0, "sample_time_ns": 0}
        worker.stop_requested.set()
        raise ConnectionError("missing response")

    worker.read_sensors = request_stop
    worker._run()
    worker.send_duty.assert_not_called()
    worker.shutdown.assert_called_once()


def test_worker_failure_is_not_hidden_by_cached_feedback(worker):
    failure = ValueError("invalid feedback")
    worker.read_sensors = Mock(side_effect=failure)
    with pytest.raises(ValueError, match="invalid feedback"):
        worker._run()
    with pytest.raises(ValueError, match="invalid feedback"):
        worker.get_state()
    assert worker.ready.is_set()
    worker.shutdown.assert_called_once()


def test_safety_callback_failure_stops_before_publishing_state(worker):
    failure = RuntimeError("temperature limit")
    state = {"read_completed_ns": 123, "sample_time_ns": 123}
    worker.read_sensors = Mock(return_value=state)
    worker.check_safety = Mock(side_effect=failure)
    with pytest.raises(RuntimeError, match="temperature limit"):
        worker._run()
    worker.check_safety.assert_called_once_with(state)
    worker.shutdown.assert_called_once()
    assert worker.latest_state is None
    assert worker.error is failure
