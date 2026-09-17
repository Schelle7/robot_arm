"""Check the hardware adapter against the installed LeRobot bus, without opening a port."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from omegaconf import OmegaConf
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from robot_arm.arms.read_sensors import read_block
from robot_arm.arms.real_arm import RealArm
from robot_arm.robot_schema import MOTOR_ORDER

ROOT = Path(__file__).resolve().parents[1]
CFG = OmegaConf.create({
    "model_path": str(ROOT / "models/so101/scene.xml"),
    "control": {"frequencies": {"joint": 20}},
    "safety": OmegaConf.load(ROOT / "conf/safety/default.yaml"),
})


def test_feedback_retry_returns_fresh_read_and_warns(monkeypatch, caplog):
    arm = RealArm.__new__(RealArm)
    arm.bus = object()
    arm.disconnect = Mock()
    feedback = {"sample_time_ns": 123}
    read = Mock(side_effect=[ConnectionError("missing motor"), feedback])
    monkeypatch.setattr("robot_arm.arms.real_arm.read_block", read)

    assert arm._read_feedback() is feedback
    assert read.call_count == 2
    arm.disconnect.assert_not_called()
    assert "SENSOR READ FAILED: MISSING MOTOR. RETRYING ONCE." in caplog.text


def test_second_feedback_failure_disconnects_and_raises(monkeypatch):
    arm = RealArm.__new__(RealArm)
    arm.bus = object()
    arm.disconnect = Mock()
    second_error = ConnectionError("second failure")
    read = Mock(side_effect=[ConnectionError("first failure"), second_error])
    monkeypatch.setattr("robot_arm.arms.real_arm.read_block", read)

    with pytest.raises(ConnectionError) as raised:
        arm._read_feedback()

    assert raised.value is second_error
    assert read.call_count == 2
    arm.disconnect.assert_called_once_with()


def test_configured_read_timeout_restarts_timer():
    port = SimpleNamespace(tx_time_per_byte=0.01, getCurrentTime=Mock(side_effect=[100.0, 200.0]))
    arm = RealArm.__new__(RealArm)
    arm.bus = SimpleNamespace(port_handler=port)
    arm.configure_read_timeout(5.0)

    port.setPacketTimeout(126)
    assert port.packet_start_time == 100.0
    assert port.packet_timeout == pytest.approx(6.29)
    port.setPacketTimeout(21)
    assert port.packet_start_time == 200.0
    assert port.packet_timeout == pytest.approx(5.24)


@pytest.fixture
def offline_bus():
    calibration = json.loads((ROOT / "conf/hardware_calibration.json").read_text())
    return FeetechMotorsBus(
        port="/unused-offline-test",
        motors={name: Motor(i + 1, "sts3215", MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.DEGREES) for i, name in enumerate(MOTOR_ORDER)},
        calibration={name: MotorCalibration(**value) for name, value in calibration.items()},
    )


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
    monkeypatch.setattr("robot_arm.arms.real_arm.read_configuration", lambda bus: {})
    monkeypatch.setattr(offline_bus, "read", lambda *args, **kwargs: 0)
    arm = RealArm(offline_bus, CFG)
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
    monkeypatch.setattr("robot_arm.arms.real_arm.read_block", lambda bus: raw_state)
    state = arm.read_state()
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
    arm.write_goal(state["Present_Position"])
    for name, tick in ticks.items():
        assert abs(written[offline_bus.motors[name].id] - tick) <= 1


def test_real_arm_duty_sign_matches_model_joint_direction(offline_bus, monkeypatch):
    monkeypatch.setattr("robot_arm.arms.real_arm.read_configuration", lambda bus: {})
    monkeypatch.setattr(offline_bus, "read", lambda *args, **kwargs: 0)
    arm = RealArm(offline_bus, CFG)
    written = {}

    def capture_duty(register, values, normalize):
        assert register == "Goal_Time" and normalize is False
        written.update(values)

    monkeypatch.setattr(offline_bus, "sync_write", capture_duty)
    arm.write_duty(
        {"shoulder_pan": 0.25, "shoulder_lift": -0.25, "elbow_flex": 0.0},
        {motor: (lower + upper) / 2 for motor, (lower, upper) in arm.joint_limits.items()},
    )
    assert written == {"shoulder_pan": 1024 + 250, "shoulder_lift": 250, "elbow_flex": 0}


@pytest.mark.parametrize("mode", [0, 2])
def test_same_hardware_pose_has_same_radians_in_position_and_pwm_modes(offline_bus, monkeypatch, mode):
    # Stationary hardware capture, 2026-09-06. Gripper differed by one tick
    # between physical reads; use the exact homing relation for this regression.
    pwm_ticks = dict(zip(MOTOR_ORDER, [3582, 1673, 80, 2543, 43, 3907]))
    position_ticks = dict(zip(MOTOR_ORDER, [1512, 2753, 2285, 1039, 2740, 1642]))
    monkeypatch.setattr("robot_arm.arms.real_arm.read_configuration", lambda bus: {})
    monkeypatch.setattr(offline_bus, "read", lambda *args, **kwargs: mode)
    arm = RealArm(offline_bus, CFG)
    monkeypatch.setattr("robot_arm.arms.real_arm.read_block", lambda bus: {
        "Present_Position": (pwm_ticks if mode == 2 else position_ticks).copy(),
        **{register: dict.fromkeys(MOTOR_ORDER, 0) for register in ("Present_Load", "Present_Velocity", "Present_Current", "Present_Voltage", "Present_Temperature")},
    })
    positions = arm.read_state()["Present_Position"]
    expected_degrees = offline_bus._normalize({offline_bus.motors[name].id: tick for name, tick in position_ticks.items()})
    for name in MOTOR_ORDER[:-1]:
        assert positions[name] == pytest.approx(np.deg2rad(expected_degrees[offline_bus.motors[name].id]))
    gripper_limits = arm.model.jnt_range[arm.joint_indices["gripper"]]
    assert positions["gripper"] == pytest.approx(gripper_limits[0] + (1642 - 1401) / (2725 - 1401) * np.diff(gripper_limits)[0])
    assert positions["elbow_flex"] > 0  # formerly -160.7 degrees
    assert positions["gripper"] < np.deg2rad(15)  # formerly clipped to +100 degrees


def test_pwm_encoder_wrap_is_continuous_in_calibrated_joint_frame(offline_bus, monkeypatch):
    monkeypatch.setattr("robot_arm.arms.real_arm.read_configuration", lambda bus: {})
    monkeypatch.setattr(offline_bus, "read", lambda *args, **kwargs: 2)
    arm = RealArm(offline_bus, CFG)
    angles = []
    for tick in [4080, 79]:
        monkeypatch.setattr("robot_arm.arms.real_arm.read_block", lambda bus: {
            "Present_Position": {"shoulder_pan": tick},
            **{register: {} for register in ("Present_Load", "Present_Velocity", "Present_Current", "Present_Voltage", "Present_Temperature")},
        })
        angles.append(arm.read_state()["Present_Position"]["shoulder_pan"])
    assert angles[1] - angles[0] == pytest.approx(95 * 2 * np.pi / 4095)


def test_setting_pwm_mode_updates_cached_feedback_frame(offline_bus, monkeypatch):
    monkeypatch.setattr("robot_arm.arms.real_arm.read_configuration", lambda bus: {})
    modes = dict.fromkeys(MOTOR_ORDER, 0)
    monkeypatch.setattr(offline_bus, "read", lambda register, name, **kwargs: modes[name])
    arm = RealArm(offline_bus, CFG)
    monkeypatch.setattr(offline_bus, "disable_torque", lambda: None)
    monkeypatch.setattr(offline_bus, "enable_torque", lambda: None)
    monkeypatch.setattr(offline_bus, "write", lambda register, name, value: modes.update({name: value}))
    monkeypatch.setattr(offline_bus, "sync_write", lambda *args, **kwargs: None)
    arm.set_pwm_mode()
    assert arm.operating_modes == dict.fromkeys(MOTOR_ORDER, 2)
