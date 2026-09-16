"""Briefly move wrist_roll to check PWM direction using the corrected RealArm adapter."""

import argparse
import json
import math
import time
from pathlib import Path

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from robot_arm.arms.real_arm import RealArm

JOINT = "wrist_roll"
DUTY = 0.08
DURATION = 0.15
MAX_TRAVEL = math.radians(5)
ROOT = Path(__file__).resolve().parents[1]


def pulse_wrist_roll(arm, result):
    bus = arm.bus
    torque = {name: bus.read("Torque_Enable", name, normalize=False) for name in bus.motors}
    if any(value != 0 for value in torque.values()):
        raise RuntimeError("All motors must already have torque OFF. No pulse sent.")
    if arm.operating_modes[JOINT] != 2:
        raise RuntimeError("wrist_roll must already be in PWM mode (2). No pulse sent.")
    initial = arm.read_state()
    start = initial["Present_Position"][JOINT]
    lower, upper = arm.model.jnt_range[arm.joint_indices[JOINT]]
    if not lower + 2 * MAX_TRAVEL < start < upper - 2 * MAX_TRAVEL:
        raise RuntimeError("wrist_roll must be at least 10 degrees inside its model limits. No pulse sent.")
    result.update(joint=JOINT, requested_duty=DUTY, requested_seconds=DURATION,
                  initial_positions_degrees={name: math.degrees(q) for name, q in initial["Present_Position"].items()}, samples=[])
    print(f"wrist_roll starts at {math.degrees(start):.2f} degrees. Applying +8% duty for 150 ms.", flush=True)
    started = time.perf_counter()
    try:
        # Clear any old PWM command BEFORE enabling this motor alone.
        arm.write_duty({JOINT: 0.0})
        if bus.read("Goal_Time", JOINT, normalize=False) != 0:
            raise RuntimeError("Old PWM command did not clear. Refusing to enable torque.")
        bus.enable_torque([JOINT])
        started = time.perf_counter()
        arm.write_duty({JOINT: DUTY})
        while time.perf_counter() - started < DURATION:
            time.sleep(min(0.02, max(0, DURATION - (time.perf_counter() - started))))
            state = arm.read_state()
            position = state["Present_Position"][JOINT]
            result["samples"].append({
                "seconds": time.perf_counter() - started,
                "position_degrees": math.degrees(position),
                "load": state["Present_Load"][JOINT],
                "voltage": state["Present_Voltage"][JOINT],
            })
            if abs(position - start) >= MAX_TRAVEL or not lower < position < upper:
                result["stopped_early"] = "Measured travel reached the 5 degree cutoff or a joint limit."
                break
    finally:
        # Always attempt torque-off even if writing zero duty fails.
        try:
            arm.write_duty({JOINT: 0.0})
        finally:
            bus.disable_torque([JOINT], num_retry=2)
        result["pulse_elapsed_seconds"] = time.perf_counter() - started
        result["torque_off_verified"] = bus.read("Torque_Enable", JOINT, normalize=False) == 0
        if not result["torque_off_verified"]:
            raise RuntimeError("wrist_roll torque-off did not read back as zero.")
    delta = result["samples"][-1]["position_degrees"] - math.degrees(start) if result["samples"] else 0.0
    result["delta_degrees"] = delta
    result["direction"] = "inconclusive: too little movement" if abs(delta) < 0.25 else ("positive duty increased angle" if delta > 0 else "positive duty DECREASED angle")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--calibration", type=Path, default=ROOT / "conf/hardware_calibration.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/duty_direction.json")
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text())
    bus = FeetechMotorsBus(
        port=args.port,
        motors={name: Motor(cal["id"], "sts3215", MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.DEGREES) for name, cal in calibration.items()},
        calibration={name: MotorCalibration(**cal) for name, cal in calibration.items()},
    )
    result = {}
    # Check the output path before touching hardware; retain partial data on errors.
    with args.out.open("w") as output:
        bus.connect(handshake=False)
        try:
            arm = RealArm(bus, str(ROOT / "models/so101/scene.xml"), 0.02)
            pulse_wrist_roll(arm, result)
        except (Exception, KeyboardInterrupt) as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                bus.disconnect(disable_torque=False)
            finally:
                json.dump(result, output, indent=2)
                output.write("\n")
    print(f"{result['direction']}: {result['delta_degrees']:+.3f} degrees. Saved {args.out}.")


if __name__ == "__main__":
    main()
