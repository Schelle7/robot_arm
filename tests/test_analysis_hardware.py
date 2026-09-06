import json
from types import SimpleNamespace

import pytest

from analysis import hardware


class ReadOnlyBus:
    def __init__(self, missing=()):
        self.missing = missing
        self.motors = {name: SimpleNamespace(id=i, norm_mode=SimpleNamespace(name="DEGREES")) for i, name in enumerate(("pan", "elbow"), 1)}
        self.calibration = {name: SimpleNamespace(homing_offset=100, range_min=100, range_max=100) for name in self.motors}
        self.connected = False

    def connect(self, *, handshake):
        assert handshake is False
        self.connected = True

    def disconnect(self, *, disable_torque):
        assert disable_torque is False
        self.connected = False

    def read(self, register, motor, *, normalize, num_retry):
        assert normalize is False and num_retry == 2
        if motor in self.missing:
            raise ConnectionError(f"No status packet from {motor}")
        return 100

    def _normalize(self, values):
        return {i: value / 10 for i, value in values.items()}


def test_grouped_read_failure_preserves_individual_results(monkeypatch):
    def failed_block(bus):
        raise ConnectionError("No grouped status packet")

    monkeypatch.setattr(hardware, "read_block", failed_block)
    result = hardware.inspect_bus(ReadOnlyBus())
    assert result["read_errors"] == {}
    assert result["registers_decoded_not_normalized"]["Operating_Mode"] == {"pan": 100, "elbow": 100}
    assert result["same_position_sample_lerobot_normalized"] == {"pan": 10, "elbow": 10}
    assert result["block_read_later_sample"] is None
    assert result["block_read_error"] == "No grouped status packet"


@pytest.mark.parametrize("missing", [("elbow",), ("pan", "elbow")])
def test_unresponsive_motors_are_reported_without_false_calibration_mismatches(monkeypatch, missing):
    monkeypatch.setattr(hardware, "read_block", lambda bus: {"sample": "ok"})
    result = hardware.inspect_bus(ReadOnlyBus(missing))
    assert set(result["read_errors"]["Operating_Mode"]) == set(missing)
    for name in missing:
        assert name not in result["same_position_sample_lerobot_normalized"]
        assert result["calibration_registers_match_file"][name] == dict.fromkeys(("homing_offset", "range_min", "range_max"))
    if len(missing) == 2:
        assert result["block_read_error"].startswith("Skipped")
    else:
        assert all(result["calibration_registers_match_file"]["pan"].values())


def test_cli_saves_failed_reads_and_closes_port_without_writes(monkeypatch, tmp_path):
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text("{}")
    output = tmp_path / "hardware.json"
    bus = ReadOnlyBus(missing=("pan", "elbow"))
    monkeypatch.setattr(hardware, "FeetechMotorsBus", lambda **kwargs: bus)
    monkeypatch.setattr("sys.argv", ["hardware", "--port", "/unused", "--calibration", str(calibration_path), "--out", str(output)])
    hardware.main()
    assert not bus.connected
    result = json.loads(output.read_text())
    assert len(result["read_errors"]) == 11
    assert result["registers_decoded_not_normalized"]["Present_Position"] == {}


class ModeBus:
    motors = {"pan": object()}

    def __init__(self, torque=0, mode=2, fail_position=False):
        self.torque = torque
        self.mode = mode
        self.fail_position = fail_position
        self.writes = []

    def read(self, register, name, *, normalize):
        assert name == "pan" and normalize is False
        if register == "Present_Position":
            if self.mode == 0 and self.fail_position:
                raise ConnectionError("Position read failed")
            return 3582 if self.mode == 2 else 1512
        return {"Torque_Enable": self.torque, "Operating_Mode": self.mode, "Homing_Offset": -2026}[register]

    def write(self, register, name, value, *, normalize):
        assert register == "Operating_Mode" and name == "pan" and normalize is False
        assert self.torque == 0
        self.writes.append(value)
        self.mode = value


@pytest.mark.parametrize("original_mode", [0, 2])
def test_compare_modes_measures_feedback_and_restores_original_mode(monkeypatch, original_mode):
    monkeypatch.setattr(hardware.time, "sleep", lambda seconds: None)
    bus = ModeBus(mode=original_mode)
    result = hardware.compare_position_modes(bus)["pan"]
    assert result["position_mode_tick"] == result["predicted_position_tick_if_pwm_ignores_homing"] == 1512
    assert result["pwm_mode_tick"] == 3582
    assert result["restored_tick"] == result["original_tick"]
    assert bus.mode == original_mode
    assert bus.writes == [2 if original_mode == 0 else 0, original_mode]


def test_compare_modes_refuses_enabled_torque():
    bus = ModeBus(torque=1)
    with pytest.raises(RuntimeError, match="torque already OFF"):
        hardware.compare_position_modes(bus)
    assert bus.writes == []


def test_compare_modes_restores_mode_when_read_fails(monkeypatch):
    monkeypatch.setattr(hardware.time, "sleep", lambda seconds: None)
    bus = ModeBus(fail_position=True)
    with pytest.raises(ConnectionError, match="Position read failed"):
        hardware.compare_position_modes(bus)
    assert bus.mode == 2
    assert bus.writes == [0, 2]
