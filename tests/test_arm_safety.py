import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from omegaconf import OmegaConf

from robot_arm.arms.arm import Arm
from robot_arm.arms.communication import SafetyException
from robot_arm.arms.sim_communication import SimCommunication
from robot_arm.arms.sim_arm import SimArm
from robot_arm.robot_schema import MOTOR_ORDER


class TestArm(Arm):
    __test__ = False

    def __init__(self, cfg):
        super().__init__(cfg)
        self.communication = SimCommunication(cfg, self.model, self.data)
        self.model = SimpleNamespace(jnt_range=np.array([[-1.0, 1.0]]))
        self.joint_indices = {"shoulder_pan": 0}
        self.state = {"Present_Load": {"shoulder_pan": 0.0}, "Present_Temperature": {"shoulder_pan": 20.0}}
        self.reads = 0
        self.written = {}
        self.communication.read_sensors = self._read_sensors
        self.communication.send_duty = self._send_duty
        self.communication.close = Mock()

    def _read_sensors(self):
        self.reads += 1
        self.state["read_completed_ns"] = round(self.reads * self.control_step_seconds * 1_000_000_000)
        return self.state

    def _send_duty(self, duties):
        self.written = duties

    def disconnect(self):
        self.communication.close()

    def get_tcp(self):
        raise NotImplementedError

    def read_cameras(self):
        raise NotImplementedError

    def advance_control_step(self):
        raise NotImplementedError


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


@pytest.mark.parametrize(
    "position, duty, expected",
    [
        (-1.0, -0.5, 0.0),
        (1.0, 0.5, 0.0),
        (-1.0, 0.5, 0.5),
        (1.0, -0.5, -0.5),
        (0.0, 0.5, 0.5),
    ],
)
def test_joint_limits_filter_only_outward_duties(cfg, position, duty, expected):
    arm = TestArm(cfg)
    applied = arm.submit_duty({"shoulder_pan": duty}, {"shoulder_pan": position}, 123)
    assert applied == arm.written == {"shoulder_pan": expected}
    assert arm.reads == 0


def test_each_read_updates_duty_average_once(cfg):
    arm = TestArm(cfg)
    arm.state["Present_Load"]["shoulder_pan"] = -0.5
    arm.get_state()
    arm.get_state()
    assert arm.reads == 2
    assert arm.communication.smoothed_duties["shoulder_pan"] == pytest.approx(0.5 * (1 - math.exp(-2 / cfg.control.frequencies.joint / cfg.safety.duty_ema_seconds)))


def test_temperature_violation_disconnects_and_raises(cfg):
    arm = TestArm(cfg)
    arm.state["Present_Temperature"]["shoulder_pan"] = cfg.safety.max_temperature_celsius + 1
    with pytest.raises(SafetyException, match="temperature") as caught:
        arm.get_state()
    assert caught.value.sensor_state is arm.state
    arm.communication.close.assert_called_once()


def test_rechecking_sample_does_not_count_it_again(cfg):
    arm = TestArm(cfg)
    arm.state["Present_Load"]["shoulder_pan"] = 0.5
    state = arm.get_state()
    smoothed = arm.communication.smoothed_duties.copy()
    arm.communication.check_safety(state)
    assert arm.communication.smoothed_duties == smoothed
    assert len(arm.communication.temperature_samples["shoulder_pan"]) == 1
    arm.communication.close.assert_not_called()


def test_temperature_mean_uses_only_samples_within_window(cfg):
    cfg.safety.max_temperature_celsius = 45
    arm = TestArm(cfg)
    state = arm.state
    for timestamp, temperature in [(0, 20), (100_000_000, 20), (150_000_000, 70)]:
        state["read_completed_ns"] = timestamp
        state["Present_Temperature"]["shoulder_pan"] = temperature
        arm.communication.check_safety(state)
    arm.communication.close.assert_not_called()
    assert state["Present_Temperature"]["shoulder_pan"] == 70

    state["read_completed_ns"] = 300_000_000
    state["Present_Temperature"]["shoulder_pan"] = 30
    with pytest.raises(SafetyException, match="mean temperature 50.00C"):
        arm.communication.check_safety(state)
    arm.communication.close.assert_called_once()


def test_sustained_duty_violation_disconnects_and_raises(cfg):
    cfg.safety.max_smoothed_duty = 0.01
    arm = TestArm(cfg)
    arm.state["Present_Load"]["shoulder_pan"] = 1.0
    with pytest.raises(SafetyException, match="sustained duty"):
        arm.get_state()
    arm.communication.close.assert_called_once()


@pytest.mark.parametrize("method", ["get_state", "submit_duty"])
def test_public_safety_methods_cannot_be_overridden(method):
    with pytest.raises(TypeError, match="cannot override final method"):
        type("UnsafeArm", (TestArm,), {method: lambda self: None})


def test_missing_raw_io_methods_prevent_instantiation():
    with pytest.raises(TypeError, match="abstract"):
        Arm(object())


def test_sim_state_restoration_resets_duty_average(cfg):
    arm = SimArm.__new__(SimArm)
    Arm.__init__(arm, cfg)
    arm.communication = SimCommunication(cfg, arm.model, arm.data)
    arm.communication.smoothed_duties["shoulder_pan"] = 0.8
    arm.communication.commanded_duty = np.full(arm.model.nu, 0.5)
    qpos = arm.data.qpos.copy()
    qpos[arm.model.jnt_qposadr[arm.joint_indices["shoulder_pan"]]] = 0.2
    qvel = np.full(arm.model.nv, 0.3)
    arm.restore_sim_state(qpos, qvel)
    assert arm.communication.smoothed_duties == dict.fromkeys(MOTOR_ORDER, 0.0)
    np.testing.assert_array_equal(arm.data.qpos, qpos)
    np.testing.assert_array_equal(arm.data.qvel, qvel)
    np.testing.assert_array_equal(arm.communication.commanded_duty, 0.0)
