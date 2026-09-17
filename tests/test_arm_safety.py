import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from robot_arm.arms.arm import Arm, SafetyException
from robot_arm.arms.sim_arm import SimArm
from robot_arm.robot_schema import MOTOR_ORDER


class TestArm(Arm):
    __test__ = False

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = SimpleNamespace(jnt_range=np.array([[-1.0, 1.0]]))
        self.joint_indices = {"shoulder_pan": 0}
        self.state = {"Present_Load": {"shoulder_pan": 0.0}, "Present_Temperature": {"shoulder_pan": 20.0}}
        self.reads = 0
        self.stopped = False
        self.written = {}

    def _read_state(self):
        self.reads += 1
        return self.state

    def _write_duty(self, duties):
        self.written = duties

    def disconnect(self):
        self.stopped = True

    def get_tcp(self):
        raise NotImplementedError

    def read_cameras(self):
        raise NotImplementedError

    def advance_control_step(self):
        raise NotImplementedError


@pytest.fixture
def cfg():
    config_dir = Path(__file__).resolve().parents[1] / "conf"
    return OmegaConf.create({
        "safety": OmegaConf.load(config_dir / "safety/default.yaml"),
        "control": OmegaConf.load(config_dir / "control/default.yaml"),
    })


@pytest.mark.parametrize("position, duty, expected", [
    (-1.0, -0.5, 0.0), (1.0, 0.5, 0.0),
    (-1.0, 0.5, 0.5), (1.0, -0.5, -0.5), (0.0, 0.5, 0.5),
])
def test_joint_limits_filter_only_outward_duties(cfg, position, duty, expected):
    arm = TestArm(cfg)
    applied = arm.write_duty({"shoulder_pan": duty}, {"shoulder_pan": position})
    assert applied == arm.written == {"shoulder_pan": expected}
    assert arm.reads == 0


def test_each_read_updates_duty_average_once(cfg):
    arm = TestArm(cfg)
    arm.state["Present_Load"]["shoulder_pan"] = -0.5
    arm.read_state()
    arm.read_state()
    assert arm.reads == 2
    assert arm.smoothed_duties["shoulder_pan"] == pytest.approx(0.5 * (1 - math.exp(-2 / cfg.control.frequencies.joint / cfg.safety.duty_ema_seconds)))


def test_temperature_violation_disconnects_and_raises(cfg):
    arm = TestArm(cfg)
    arm.state["Present_Temperature"]["shoulder_pan"] = cfg.safety.max_temperature_celsius + 1
    with pytest.raises(SafetyException, match="temperature"):
        arm.read_state()
    assert arm.stopped


def test_sustained_duty_violation_disconnects_and_raises(cfg):
    cfg.safety.max_smoothed_duty = 0.01
    arm = TestArm(cfg)
    arm.state["Present_Load"]["shoulder_pan"] = 1.0
    with pytest.raises(SafetyException, match="sustained duty"):
        arm.read_state()
    assert arm.stopped


@pytest.mark.parametrize("method", ["read_state", "write_duty"])
def test_public_safety_methods_cannot_be_overridden(method):
    with pytest.raises(TypeError, match="instead of overriding"):
        type("UnsafeArm", (TestArm,), {method: lambda self: None})


def test_missing_raw_io_methods_prevent_instantiation():
    with pytest.raises(TypeError, match="abstract"):
        Arm(object())


def test_sim_state_restoration_resets_duty_average(cfg, monkeypatch):
    arm = TestArm(cfg)
    arm.smoothed_duties["shoulder_pan"] = 0.8
    arm.data = type("SimData", (), {"qpos": [0.0], "qvel": [0.0]})()
    arm.commanded_duty = np.array([0.5])
    monkeypatch.setattr("robot_arm.arms.sim_arm.mujoco.mj_resetData", lambda model, data: None)
    monkeypatch.setattr("robot_arm.arms.sim_arm.mujoco.mj_forward", lambda model, data: None)
    SimArm.restore_sim_state(arm, [0.2], [0.3])
    assert arm.smoothed_duties == dict.fromkeys(MOTOR_ORDER, 0.0)
    assert arm.data.qpos == [0.2]
    assert arm.data.qvel == [0.3]
    assert arm.commanded_duty[0] == 0.0
