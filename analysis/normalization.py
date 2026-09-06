"""Offline calibration, policy-input and command-direction audit of recorded rollouts.

Run with ``python -m analysis.normalization 0 1 --calibration conf/hardware_calibration.json``.
Reconstructed ticks assume the recording used RealArm's DEGREES body conversion.
They are not a substitute for raw register values or a calibration snapshot.
The rollout audit describes the legacy path BEFORE the PWM homing-offset fix.
"""

import argparse
import contextlib
import json
from pathlib import Path

import mujoco
import numpy as np
from omegaconf import OmegaConf

from analysis.rollouts import Rollout, dense_steps, find_rollouts, load_rollout
from robot_arm.numpy_policy import load_numpy_policy
from robot_arm.robot_schema import MOTOR_ORDER


def reconstruct_body_ticks(positions: np.ndarray, calibration: dict) -> np.ndarray:
    midpoint = np.array([(calibration[name]["range_min"] + calibration[name]["range_max"]) / 2 for name in MOTOR_ORDER[:-1]])
    return np.asarray(positions)[..., :5] * 4095 / (2 * np.pi) + midpoint


def print_mode_comparison(capture: dict, model_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    print("\nSTATIONARY MODE COMPARISON: degrees in the model frame")
    print(f"{'joint':16} {'old PWM':>12} {'corrected':>12} {'position mode':>14} {'tick residual':>14}")
    for name in MOTOR_ORDER:
        row = capture["mode_comparison"][name]
        cal = capture["calibration"][name]
        corrected_tick = (row["pwm_mode_tick"] - cal["homing_offset"]) % 4096

        def degrees(tick):
            if name != "gripper":
                return (tick - (cal["range_min"] + cal["range_max"]) / 2) * 360 / 4095
            limits = np.rad2deg(model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
            fraction = np.clip((tick - cal["range_min"]) / (cal["range_max"] - cal["range_min"]), 0, 1)
            return limits[0] + fraction * (limits[1] - limits[0])

        print(f"{name:16} {degrees(row['pwm_mode_tick']):12.3f} {degrees(corrected_tick):12.3f} {degrees(row['position_mode_tick']):14.3f} {corrected_tick - row['position_mode_tick']:14}")


def print_normalization_audit(rollout: Rollout, calibration: dict) -> None:
    print(f"\nRUN {rollout.run_dir}")
    if rollout.data is None:
        print("No recording.")
        return
    if rollout.config.backend != "real":
        print("Skipping hardware calibration audit for simulated recording.")
        return
    steps = dense_steps(rollout.data)
    if not steps:
        print("No dense policy observations.")
        return
    model = mujoco.MjModel.from_xml_path(str(rollout.model_path))
    limits = np.array([model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in MOTOR_ORDER])
    q = np.array([s["obs"]["joint_positions"] for s in steps])
    next_q = np.array([s["next_obs"]["joint_positions"] for s in steps])
    ticks = reconstruct_body_ticks(q, calibration)
    next_ticks = reconstruct_body_ticks(next_q, calibration)
    print("Assumption: supplied calibration was active; body observations used LeRobot DEGREES.")
    print("Gripper ticks cannot be recovered at its clipped endpoints.")
    print("\nMeasured calibration endpoints vs model endpoints, degrees; no endpoint rescaling in DEGREES mode")
    print(f"{'joint':16} {'cal low':>10} {'cal high':>10} {'model low':>10} {'model high':>10} {'start':>10} {'outside':>10} {'start tick':>12}")
    for j, name in enumerate(MOTOR_ORDER):
        cal = calibration[name]
        cal_half_span = (cal["range_max"] - cal["range_min"]) * 180 / 4095
        lower, upper = np.rad2deg(limits[j])
        start = np.rad2deg(q[0, j])
        outside = max(lower - start, start - upper, 0)
        tick = f"{ticks[0, j]:.3f}" if j < 5 else "clipped/map"
        cal_low, cal_high = (-cal_half_span, cal_half_span) if j < 5 else (lower, upper)
        print(f"{name:16} {cal_low:10.3f} {cal_high:10.3f} {lower:10.3f} {upper:10.3f} {start:10.3f} {outside:10.3f} {tick:>12}")
    print(f"Maximum reconstructed body tick rounding residual: {np.max(np.abs(ticks - np.rint(ticks))):.6f}")
    print(f"Reconstructed body tick range: {ticks.min():.3f} .. {ticks.max():.3f}")

    print("\nEvery transition, every joint: reconstructed tick, requested duty, inferred guarded duty, angle change")
    print("Guarded duty is reconstructed from the recorded request and run model, not recorded bus output.")
    print("Wrapped angle differences are diagnostic only; they do not establish absolute joint alignment.")
    print(f"{'step':>5} {'joint':16} {'tick':>9} {'next tick':>10} {'request':>9} {'guarded':>9} {'dq deg':>10} {'wrap dq':>10}")
    requested = np.array([s["compensated_duty"] for s in steps])
    guarded = np.where(((q <= limits[:, 0]) & (requested < 0)) | ((q >= limits[:, 1]) & (requested > 0)), 0, requested)
    delta = next_q - q
    wrapped_delta = delta.copy()
    # The installed LeRobot conversion uses 4095, but the encoder wraps every 4096 ticks.
    encoder_period = 2 * np.pi * 4096 / 4095
    wrapped_delta[:, :5] = (delta[:, :5] + encoder_period / 2) % encoder_period - encoder_period / 2
    for i in range(len(steps)):
        for j, name in enumerate(MOTOR_ORDER):
            tick = f"{ticks[i, j]:.1f}" if j < 5 else "unknown"
            next_tick = f"{next_ticks[i, j]:.1f}" if j < 5 else "unknown"
            print(f"{i:5} {name:16} {tick:>9} {next_tick:>10} {requested[i,j]:9.4f} {guarded[i,j]:9.4f} {np.rad2deg(delta[i,j]):10.3f} {np.rad2deg(wrapped_delta[i,j]):10.3f}")
    print("\nDirection evidence (|guarded duty| >= .1 and movement >= 1 degree; inertia/gravity can confound)")
    for j, name in enumerate(MOTOR_ORDER):
        mask = (np.abs(guarded[:, j]) >= 0.1) & (np.abs(wrapped_delta[:, j]) >= np.deg2rad(1))
        opposite = np.sum(guarded[mask, j] * wrapped_delta[mask, j] < 0)
        jumps = np.flatnonzero(np.abs(delta[:, j]) > np.pi).tolist()
        print(f"{name:16}: opposite direction {opposite}/{mask.sum()}, >180 degree discontinuities at transitions {jumps}")

    print("\nPolicy normalization checks")
    cfg = rollout.config
    joint_velocity_scale = float(cfg.control.joint_velocity_scale_radians_per_second)
    print(f"Joint velocity policy divisor: {joint_velocity_scale}; positions stay in radians.")
    print(f"Maximum |physical finite-difference velocity|: {max(np.max(np.abs(s['obs']['joint_velocities'])) for s in steps) * joint_velocity_scale:.3f} rad/s")
    checkpoint = Path(cfg.policy_name)
    config_path = checkpoint.parent.parent / ".hydra/config.yaml"
    if config_path.exists():
        trained = OmegaConf.load(config_path)
        fields = (
            "control.frequencies",
            "control.joint_velocity_scale_radians_per_second",
            "control.state_history_seconds",
            "control.duty_history_seconds",
            "waypoint.position_speed_meters_per_second",
            "waypoint.rotation_speed_radians_per_second",
            "waypoint.gripper_speed_radians_per_second",
            "reward.action_change_penalty",
            "reward.action_change_penalty_factor",
            "servo",
        )
        for field in fields:
            a, b = OmegaConf.select(cfg, field), OmegaConf.select(trained, field)
            print(f"{field}: {'MATCH' if a == b else f'MISMATCH rollout={a} training={b}'}")
    if checkpoint.with_suffix(".actor.npz").exists():
        policy = load_numpy_policy(str(checkpoint))
        errors = [np.max(np.abs(policy.predict(s["obs"], deterministic=True)[0] - s["action"])) for s in steps]
        print(f"Replaying ALL {len(steps)} recorded policy inputs: max absolute action error {max(errors):.9g}")
    else:
        print("Actor export unavailable; policy replay skipped.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", default=["0", "1"], help="Run directories or indices into the newest 10 runs.")
    parser.add_argument("--calibration", type=Path, required=True, help="Calibration JSON assumed active during these runs.")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--mode-capture", type=Path, help="Also compare measured position/PWM feedback against the homing correction, offline.")
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text())
    runs = find_rollouts()
    with contextlib.ExitStack() as stack:
        if args.out:
            stack.enter_context(contextlib.redirect_stdout(stack.enter_context(args.out.open("w"))))
        print(f"Calibration source: {args.calibration.resolve()}")
        print("Legacy rollout audit: assumes PWM positions were normalized without first applying homing offsets.")
        if args.mode_capture:
            print_mode_comparison(json.loads(args.mode_capture.read_text()), load_rollout(runs[0]).model_path)
        for run in args.runs:
            print_normalization_audit(load_rollout(runs[int(run)] if run.isdigit() else Path(run)), calibration)


if __name__ == "__main__":
    main()
