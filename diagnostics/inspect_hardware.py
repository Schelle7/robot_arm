"""Read calibration and position registers without configuring or commanding the arm.

Unlike SOFollower.connect(), this uses the bus directly, skips its handshake,
and leaves torque, operating mode and EEPROM untouched by default.
--compare-modes temporarily changes operating mode with torque already off,
then restores the original mode. It never enables torque or sends motion goals.
"""

import argparse
import json
import time
from pathlib import Path

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from robot_arm.arms.read_sensors import read_block


def compare_position_modes(bus) -> dict:
    """Compare feedback frames at a stationary pose, restoring each motor immediately."""
    torque = {name: bus.read("Torque_Enable", name, normalize=False) for name in bus.motors}
    if any(value != 0 for value in torque.values()):
        raise RuntimeError(f"Mode comparison requires torque already OFF on every motor; read {torque}. No modes changed.")
    modes = {name: bus.read("Operating_Mode", name, normalize=False) for name in bus.motors}
    if any(mode not in (0, 2) for mode in modes.values()):
        raise RuntimeError(f"Only position/PWM modes are supported for comparison; read {modes}. No modes changed.")

    results = {}
    for name, original_mode in modes.items():
        if bus.read("Torque_Enable", name, normalize=False) != 0:
            raise RuntimeError(f"Torque became enabled on {name}; refusing to change its mode.")
        original_tick = bus.read("Present_Position", name, normalize=False)
        other_mode = 0 if original_mode == 2 else 2
        try:
            bus.write("Operating_Mode", name, other_mode, normalize=False)
            if bus.read("Operating_Mode", name, normalize=False) != other_mode:
                raise RuntimeError(f"{name} did not accept mode {other_mode}.")
            time.sleep(0.05)  # allow firmware feedback to refresh after the mode change
            other_tick = bus.read("Present_Position", name, normalize=False)
        finally:
            # Attempt restoration even if the first write timed out: it may have succeeded.
            bus.write("Operating_Mode", name, original_mode, normalize=False)
            if bus.read("Operating_Mode", name, normalize=False) != original_mode:
                raise RuntimeError(f"Failed to restore {name} to mode {original_mode}.")
        time.sleep(0.05)
        restored_tick = bus.read("Present_Position", name, normalize=False)
        if bus.read("Torque_Enable", name, normalize=False) != 0:
            raise RuntimeError(f"Torque unexpectedly enabled on {name} after comparison.")
        mode_ticks = {original_mode: original_tick, other_mode: other_tick}
        homing = bus.read("Homing_Offset", name, normalize=False)
        results[name] = {
            "original_mode": original_mode,
            "original_tick": original_tick,
            "restored_tick": restored_tick,
            "position_mode_tick": mode_ticks[0],
            "pwm_mode_tick": mode_ticks[2],
            "homing_offset": homing,
            "predicted_position_tick_if_pwm_ignores_homing": (mode_ticks[2] - homing) % 4096,
            "original_mode_restored": True,
            "torque_remained_off_at_checks": True,
        }
    return results


def inspect_bus(bus) -> dict:
    registers = (
        "Operating_Mode",
        "Torque_Enable",
        "Phase",
        "Homing_Offset",
        "Min_Position_Limit",
        "Max_Position_Limit",
        "Min_Voltage_Limit",
        "Present_Position",
        "Present_Velocity",
        "Present_Load",
        "Present_Voltage",
    )
    values = {register: {} for register in registers}
    errors = {}
    # Individual reads isolate an unresponsive motor and avoid requiring grouped
    # reads for configuration registers. Never configure the motors to make a read work.
    for register in registers:
        for name in bus.motors:
            try:
                values[register][name] = bus.read(register, name, normalize=False, num_retry=2)
            except ConnectionError as exc:
                errors.setdefault(register, {})[name] = str(exc)
    # Normalize the SAME sample, so motion between bus reads cannot masquerade as a conversion bug.
    normalized = bus._normalize({bus.motors[name].id: tick for name, tick in values["Present_Position"].items()})
    block = None
    block_error = None
    if values["Present_Position"]:
        try:
            block = read_block(bus)
        except ConnectionError as exc:
            block_error = str(exc)
    else:
        block_error = "Skipped: no individual position reads succeeded."

    def matches(register, name, expected):
        actual = values[register].get(name)
        return None if actual is None else actual == expected

    return {
        "read_errors": errors,
        "registers_decoded_not_normalized": values,
        "same_position_sample_lerobot_normalized": {name: normalized[motor.id] for name, motor in bus.motors.items() if motor.id in normalized},
        "normalization_modes": {name: motor.norm_mode.name for name, motor in bus.motors.items()},
        "block_read_later_sample": block,
        "block_read_error": block_error,
        "calibration_registers_match_file": {
            name: {
                "homing_offset": matches("Homing_Offset", name, cal.homing_offset),
                "range_min": matches("Min_Position_Limit", name, cal.range_min),
                "range_max": matches("Max_Position_Limit", name, cal.range_max),
            }
            for name, cal in bus.calibration.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--compare-modes",
        action="store_true",
        help="With torque already off, temporarily switch each motor between position and PWM modes and restore it. Writes Operating_Mode only.",
    )
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text())
    bus = FeetechMotorsBus(
        port=args.port,
        motors={name: Motor(cal["id"], "sts3215", MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.DEGREES) for name, cal in calibration.items()},
        calibration={name: MotorCalibration(**cal) for name, cal in calibration.items()},
    )
    bus.connect(handshake=False)
    try:
        result = {"calibration_file": str(args.calibration.resolve()), "calibration": calibration, **inspect_bus(bus)}
        if args.compare_modes:
            try:
                result["mode_comparison"] = compare_position_modes(bus)
            except (ConnectionError, RuntimeError) as exc:
                result["mode_comparison_error"] = str(exc)
    finally:
        bus.disconnect(disable_torque=False)
    output = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(output)
        failed_reads = sum(len(motors) for motors in result["read_errors"].values())
        print(f"Saved {args.out} (failed individual reads: {failed_reads}; block read: {'failed/skipped' if result['block_read_error'] else 'OK'}).")
        if "mode_comparison_error" in result:
            print(f"Mode comparison failed: {result['mode_comparison_error']}")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
