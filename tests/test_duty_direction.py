from types import SimpleNamespace

import pytest

from analysis import duty_direction as diagnostic


class FakeArm:
    def __init__(self, direction=1, torque=0, fail_read=False):
        self.direction = direction
        self.torque = torque
        self.fail_read = fail_read
        self.duty = 0
        self.position = 0
        self.commands = []
        self.model = SimpleNamespace(jnt_range=[(-2.74, 2.74)])
        self.joint_indices = {"wrist_roll": 0}
        self.operating_modes = {"wrist_roll": 2}
        self.bus = SimpleNamespace(motors={"wrist_roll": object()}, read=self.read, enable_torque=self.enable, disable_torque=self.disable)

    def read(self, register, name, **kwargs):
        if register == "Goal_Time":
            return round(self.duty * 1000)
        assert register == "Torque_Enable"
        return self.torque

    def enable(self, names):
        assert names == ["wrist_roll"] and self.duty == 0
        self.torque = 1

    def disable(self, names, **kwargs):
        assert names == ["wrist_roll"]
        self.torque = 0

    def write_duty(self, duties):
        self.duty = duties["wrist_roll"]
        self.commands.append(self.duty)

    def read_state(self):
        if self.torque and self.fail_read:
            raise ConnectionError("Sensor read failed")
        if self.torque and self.duty:
            self.position += self.direction * 0.01
        return {"Present_Position": {"wrist_roll": self.position}, "Present_Load": {"wrist_roll": self.duty}, "Present_Voltage": {"wrist_roll": 5.2}}


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(diagnostic.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(diagnostic.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))


@pytest.mark.parametrize("direction, expected", [(1, "positive duty increased angle"), (-1, "positive duty DECREASED angle"), (0, "inconclusive: too little movement")])
def test_pulse_measures_direction_and_stops(direction, expected):
    arm = FakeArm(direction)
    result = {}
    diagnostic.pulse_wrist_roll(arm, result)
    assert result["direction"] == expected
    assert result["torque_off_verified"]
    assert arm.commands == [0, 0.08, 0]
    assert result["pulse_elapsed_seconds"] == pytest.approx(0.15)


def test_read_failure_still_clears_duty_and_disables_torque():
    arm = FakeArm(fail_read=True)
    with pytest.raises(ConnectionError):
        diagnostic.pulse_wrist_roll(arm, {})
    assert arm.duty == 0 and arm.torque == 0


@pytest.mark.parametrize("condition", ["torque", "mode", "limit"])
def test_invalid_start_never_sends_pulse(condition):
    arm = FakeArm(torque=1 if condition == "torque" else 0)
    if condition == "mode":
        arm.operating_modes["wrist_roll"] = 0
    if condition == "limit":
        arm.position = 2.7
    with pytest.raises(RuntimeError):
        diagnostic.pulse_wrist_roll(arm, {})
    assert arm.commands == []


def test_excess_travel_stops_early():
    arm = FakeArm(direction=10)
    result = {}
    diagnostic.pulse_wrist_roll(arm, result)
    assert "stopped_early" in result
    assert result["pulse_elapsed_seconds"] < 0.15
    assert arm.duty == 0 and arm.torque == 0


def test_failed_zero_duty_cleanup_still_disables_torque():
    arm = FakeArm()
    original = arm.write_duty

    def write(duties):
        if duties["wrist_roll"] == 0 and arm.torque:
            raise ConnectionError("Zero duty failed")
        original(duties)

    arm.write_duty = write
    with pytest.raises(ConnectionError, match="Zero duty failed"):
        diagnostic.pulse_wrist_roll(arm, {})
    assert arm.torque == 0
