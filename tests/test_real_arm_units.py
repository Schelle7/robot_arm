"""Check the hardware adapter against the installed LeRobot bus, without opening a port."""

import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from omegaconf import OmegaConf
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from robot_arm.arms.read_sensors import read_block
from robot_arm.arms.communication import SafetyException
from robot_arm.arms.real_arm import RealArm
from robot_arm.robot_schema import MOTOR_ORDER

ROOT = Path(__file__).resolve().parents[1]
CFG = OmegaConf.create(
    {
        "model_path": str(ROOT / "models/so101/scene.xml"),
        "control": {"frequencies": {"joint": 20}},
        "safety": OmegaConf.load(ROOT / "conf/safety/default.yaml"),
        "hardware": {"port": "/unused-offline-test", "calibration_id": "test", "read_timeout_seconds": 0.01, "max_feedback_age_seconds": 1.0, "read_history_size": 10000},
    }
)


@pytest.fixture
def offline_bus(monkeypatch):
    calibration = json.loads((ROOT / "conf/hardware_calibration.json").read_text())
    bus = FeetechMotorsBus(
        port="/unused-offline-test",
        motors={name: Motor(i + 1, "sts3215", MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.DEGREES) for i, name in enumerate(MOTOR_ORDER)},
        calibration={name: MotorCalibration(**value) for name, value in calibration.items()},
    )
    follower = SimpleNamespace(bus=bus, connect=Mock(), disconnect=Mock())
    monkeypatch.setattr("robot_arm.arms.real_communication.SO101Follower", lambda cfg: follower)
    return bus


@pytest.mark.parametrize("position", [0, 79, 3723, 4095, (1 << 15) | 123])
def test_block_read_matches_lerobot_signed_register_decoding(offline_bus, monkeypatch, position):
    bus = offline_bus
    values = {56: position, 58: (1 << 15) | 64, 60: (1 << 10) | 416, 62: 53, 63: 25, 69: 20}
    monkeypatch.setattr(bus, "_setup_sync_reader", lambda *args: None)
    bus.sync_reader = SimpleNamespace(
        txRxPacket=lambda: bus._comm_success,
        isAvailable=lambda *args: True,
        getData=lambda motor_id, address, width: values[address],
    )
    result = read_block(bus)
    for name in MOTOR_ORDER:
        assert result["Present_Position"][name] == bus._decode_sign("Present_Position", {1: position})[1]
        assert result["Present_Velocity"][name] == -64
        assert result["Present_Load"][name] == -416
        assert result["Present_Voltage"][name] == 53  # still raw at the reader boundary


def test_real_arm_converts_units_once_and_roundtrips_goals(offline_bus, monkeypatch):
    monkeypatch.setattr("robot_arm.arms.real_communication.read_configuration", lambda bus: {})
    monkeypatch.setattr(offline_bus, "read", lambda *args, **kwargs: 0)
    arm = RealArm(CFG)
    arm.communication.operating_modes = {name: offline_bus.read("Operating_Mode", name) for name in MOTOR_ORDER}
    ticks = {name: 2200 for name in MOTOR_ORDER}
    ticks["shoulder_pan"] = 3723  # observed in the failed run: signed decoding must leave this alone
    raw_state = {
        "Present_Position": ticks.copy(),
        "Present_Load": dict.fromkeys(MOTOR_ORDER, -416),
        "Present_Velocity": dict.fromkeys(MOTOR_ORDER, -64),
        "Present_Current": dict.fromkeys(MOTOR_ORDER, 20),
        "Present_Voltage": dict.fromkeys(MOTOR_ORDER, 53),
        "Present_Temperature": dict.fromkeys(MOTOR_ORDER, 25),
    }
    state = arm.communication._convert_sensor_units(raw_state)
    assert state["Present_Position"]["shoulder_pan"] == pytest.approx(np.deg2rad(144.615384615))
    assert state["Present_Load"]["shoulder_pan"] == pytest.approx(0.416)
    assert state["Present_Velocity"]["shoulder_pan"] == pytest.approx(-64 * 2 * np.pi / 4096)
    assert state["Present_Current"]["shoulder_pan"] == pytest.approx(0.13)
    assert state["Present_Voltage"]["shoulder_pan"] == pytest.approx(5.3)

    written = {}

    def capture_goal(register, positions, normalize):
        assert register == "Goal_Position" and normalize
        written.update(offline_bus._unnormalize({offline_bus.motors[name].id: value for name, value in positions.items()}))

    monkeypatch.setattr(offline_bus, "sync_write", capture_goal)
    arm.communication.write_goal(state["Present_Position"])
    for name, tick in ticks.items():
        assert abs(written[offline_bus.motors[name].id] - tick) <= 1


def test_real_arm_duty_sign_matches_model_joint_direction(offline_bus, monkeypatch):
    monkeypatch.setattr("robot_arm.arms.real_communication.read_configuration", lambda bus: {})
    monkeypatch.setattr(offline_bus, "read", lambda *args, **kwargs: 0)
    arm = RealArm(CFG)
    arm.communication.operating_modes = {name: offline_bus.read("Operating_Mode", name) for name in MOTOR_ORDER}
    written = {}

    def capture_duty(register, values, normalize):
        assert register == "Goal_Time" and normalize is False
        written.update(values)

    monkeypatch.setattr(offline_bus, "sync_write", capture_duty)
    arm.communication.send_duty({"shoulder_pan": 0.25, "shoulder_lift": -0.25, "elbow_flex": 0.0})
    assert written == {"shoulder_pan": 1024 + 250, "shoulder_lift": 250, "elbow_flex": 0}


@pytest.mark.parametrize("mode", [0, 2])
def test_same_hardware_pose_has_same_radians_in_position_and_pwm_modes(offline_bus, monkeypatch, mode):
    # Stationary hardware capture, 2026-09-06. Gripper differed by one tick
    # between physical reads; use the exact homing relation for this regression.
    pwm_ticks = dict(zip(MOTOR_ORDER, [3582, 1673, 80, 2543, 43, 3907]))
    position_ticks = dict(zip(MOTOR_ORDER, [1512, 2753, 2285, 1039, 2740, 1642]))
    monkeypatch.setattr("robot_arm.arms.real_communication.read_configuration", lambda bus: {})
    monkeypatch.setattr(offline_bus, "read", lambda *args, **kwargs: mode)
    arm = RealArm(CFG)
    arm.communication.operating_modes = {name: offline_bus.read("Operating_Mode", name) for name in MOTOR_ORDER}
    raw_state = {
        "Present_Position": (pwm_ticks if mode == 2 else position_ticks).copy(),
        **{register: dict.fromkeys(MOTOR_ORDER, 0) for register in ("Present_Load", "Present_Velocity", "Present_Current", "Present_Voltage", "Present_Temperature")},
    }
    positions = arm.communication._convert_sensor_units(raw_state)["Present_Position"]
    expected_degrees = offline_bus._normalize({offline_bus.motors[name].id: tick for name, tick in position_ticks.items()})
    for name in MOTOR_ORDER[:-1]:
        assert positions[name] == pytest.approx(np.deg2rad(expected_degrees[offline_bus.motors[name].id]))
    gripper_limits = arm.model.jnt_range[arm.joint_indices["gripper"]]
    assert positions["gripper"] == pytest.approx(gripper_limits[0] + (1642 - 1401) / (2725 - 1401) * np.diff(gripper_limits)[0])
    assert positions["elbow_flex"] > 0  # formerly -160.7 degrees
    assert positions["gripper"] < np.deg2rad(15)  # formerly clipped to +100 degrees


def test_pwm_encoder_wrap_is_continuous_in_calibrated_joint_frame(offline_bus, monkeypatch):
    monkeypatch.setattr("robot_arm.arms.real_communication.read_configuration", lambda bus: {})
    monkeypatch.setattr(offline_bus, "read", lambda *args, **kwargs: 2)
    arm = RealArm(CFG)
    arm.communication.operating_modes = {name: offline_bus.read("Operating_Mode", name) for name in MOTOR_ORDER}
    angles = []
    for tick in [4080, 79]:
        raw_state = {
            "Present_Position": {"shoulder_pan": tick},
            **{register: {} for register in ("Present_Load", "Present_Velocity", "Present_Current", "Present_Voltage", "Present_Temperature")},
        }
        angles.append(arm.communication._convert_sensor_units(raw_state)["Present_Position"]["shoulder_pan"])
    assert angles[1] - angles[0] == pytest.approx(95 * 2 * np.pi / 4095)


def test_setting_pwm_mode_updates_cached_feedback_frame(offline_bus, monkeypatch):
    monkeypatch.setattr("robot_arm.arms.real_communication.read_configuration", lambda bus: {})
    modes = dict.fromkeys(MOTOR_ORDER, 0)
    monkeypatch.setattr(offline_bus, "read", lambda register, name, **kwargs: modes[name])
    arm = RealArm(CFG)
    arm.communication.operating_modes = {name: offline_bus.read("Operating_Mode", name) for name in MOTOR_ORDER}
    monkeypatch.setattr(offline_bus, "disable_torque", lambda: None)
    monkeypatch.setattr(offline_bus, "enable_torque", lambda: None)
    monkeypatch.setattr(offline_bus, "write", lambda register, name, value: modes.update({name: value}))
    monkeypatch.setattr(offline_bus, "sync_write", lambda *args, **kwargs: None)
    arm.communication._set_pwm_mode()
    assert arm.communication.operating_modes == dict.fromkeys(MOTOR_ORDER, 2)


def test_communication_connects_and_clears_duty_before_enabling_torque(offline_bus, monkeypatch):
    communication = RealArm(CFG).communication
    operations = Mock()
    operations.attach_mock(communication.follower.connect, "connect")
    monkeypatch.setattr("robot_arm.arms.real_communication.read_configuration", operations.read_configuration)
    monkeypatch.setattr(offline_bus, "disable_torque", operations.disable_torque)
    monkeypatch.setattr(offline_bus, "enable_torque", operations.enable_torque)
    monkeypatch.setattr(offline_bus, "write", operations.write)
    monkeypatch.setattr(offline_bus, "read", Mock(return_value=2))
    monkeypatch.setattr(offline_bus, "sync_write", operations.sync_write)

    communication.setup()

    communication.follower.connect.assert_called_once_with(calibrate=True)
    assert communication.connected
    assert communication.configuration is operations.read_configuration.return_value
    operations.sync_write.assert_called_once_with("Goal_Time", dict.fromkeys(MOTOR_ORDER, 0), normalize=False)
    names = [call[0] for call in operations.mock_calls]
    assert names.index("connect") < names.index("disable_torque") < names.index("write") < names.index("sync_write") < names.index("enable_torque")


def test_communication_disables_torque_before_disconnecting(offline_bus, monkeypatch):
    communication = RealArm(CFG).communication
    communication.connected = True
    operations = Mock()
    operations.attach_mock(communication.follower.disconnect, "disconnect")
    monkeypatch.setattr(offline_bus.port_handler, "ser", operations.serial)
    monkeypatch.setattr("robot_arm.arms.real_communication.time.sleep", lambda seconds: None)

    communication.close()

    assert operations.serial.write.call_count == 3
    operations.serial.write.assert_called_with(b"\xff\xff\xfe\x04\x03\x28\x00\xd2")
    assert operations.serial.flush.call_count == 3
    assert [call[0] for call in operations.mock_calls][-1] == "disconnect"
    assert not communication.connected

    communication.close()
    communication.follower.disconnect.assert_called_once()


def test_cached_policy_reads_do_not_repeat_safety_updates(offline_bus, monkeypatch):
    monkeypatch.setattr("robot_arm.arms.real_communication.read_configuration", lambda bus: {})
    monkeypatch.setattr(offline_bus, "read", lambda *args, **kwargs: 0)
    arm = RealArm(CFG)
    arm.communication.operating_modes = {name: offline_bus.read("Operating_Mode", name) for name in MOTOR_ORDER}
    arm.communication.get_state = Mock(return_value={"sample_time_ns": 123})
    arm.communication._check_temperature = Mock()
    arm.communication._check_duty = Mock()
    arm.get_state()
    arm.get_state()
    arm.communication._check_temperature.assert_not_called()
    arm.communication._check_duty.assert_not_called()
    arm.communication.get_state.assert_called_with()


def test_fresh_feedback_checks_temperature_and_uses_elapsed_time_for_duty(offline_bus, monkeypatch):
    monkeypatch.setattr("robot_arm.arms.real_communication.read_configuration", lambda bus: {})
    monkeypatch.setattr(offline_bus, "read", lambda *args, **kwargs: 0)
    arm = RealArm(CFG)
    arm.communication.operating_modes = {name: offline_bus.read("Operating_Mode", name) for name in MOTOR_ORDER}
    arm.communication.last_safety_read_ns = 1_000_000_000
    state = {"read_completed_ns": 1_010_000_000, "Present_Load": {"gripper": 0.5}, "Present_Temperature": {"gripper": 20}}
    monkeypatch.setattr("robot_arm.arms.real_communication.read_block", lambda bus: state)
    monkeypatch.setattr(arm.communication, "_convert_sensor_units", lambda value: value)
    arm.communication.check_safety(arm.communication.read_sensors())
    assert arm.communication.smoothed_duties["gripper"] == pytest.approx(0.5 * (1 - math.exp(-0.01 / CFG.safety.duty_ema_seconds)))
    state["Present_Temperature"]["gripper"] = CFG.safety.max_temperature_celsius + 1
    state["read_completed_ns"] += round(CFG.safety.temperature_average_seconds * 1_000_000_000)
    arm.communication.close = Mock()
    with pytest.raises(SafetyException, match="temperature"):
        arm.communication.check_safety(arm.communication.read_sensors())
    arm.communication.close.assert_called_once()


def test_submission_uses_explicit_sample_even_after_another_state_read(offline_bus, monkeypatch):
    monkeypatch.setattr("robot_arm.arms.real_communication.read_configuration", lambda bus: {})
    monkeypatch.setattr(offline_bus, "read", lambda *args, **kwargs: 0)
    arm = RealArm(CFG)
    arm.communication.operating_modes = {name: offline_bus.read("Operating_Mode", name) for name in MOTOR_ORDER}
    arm.communication.get_state = Mock(side_effect=[{"sample_time_ns": 123}, {"sample_time_ns": 456}])
    arm.communication.submit_duty = Mock()
    policy_state = arm.get_state()
    arm.get_state()
    lower, upper = arm.joint_limits["gripper"]
    arm.submit_duty({"gripper": 0.2}, {"gripper": (lower + upper) / 2}, policy_state["sample_time_ns"])
    arm.communication.submit_duty.assert_called_once_with({"gripper": 0.2}, 123)
