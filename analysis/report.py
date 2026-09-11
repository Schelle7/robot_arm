import numpy as np
from omegaconf import OmegaConf

from analysis.rollouts import Rollout, dense_steps, stack_dense
from robot_arm.robot_schema import CARTESIAN_ACTION_NAMES, MOTOR_ORDER

COLUMN = 15
INDEX = 6


def print_rollout(rollout: Rollout) -> None:
    """
    The whole run, every step of it. Nothing is summarised away: a rollout that went wrong is
    diagnosed from the trace, and an aggregate hides the three steps that mattered.
    """
    _print_run(rollout)
    _print_config(rollout)
    _print_servo_configuration(rollout)
    _print_git(rollout)

    if rollout.data is None:
        print("\nNo episode.npz under this run, it never reached save().")
    else:
        _print_state_traces(rollout)
        _print_cartesian_traces(rollout)
        _print_dense_traces(rollout)
        _print_waypoints(rollout)

    _print_log(rollout)


def _print_run(rollout: Rollout) -> None:
    _banner(f"{rollout.job}  {rollout.started}")
    print(f"run_dir        {rollout.run_dir}")
    print(f"episode        {rollout.episode_path}")


def _print_config(rollout: Rollout) -> None:
    _banner("merged config")
    print(OmegaConf.to_yaml(rollout.config))


def _print_servo_configuration(rollout: Rollout) -> None:
    if rollout.servo_configuration is None:
        return

    _banner("servo configuration read off the arm")
    _print_columns(MOTOR_ORDER, label_width=24)
    for register, values in rollout.servo_configuration.items():
        print(f"{register:<24}" + "".join(f"{values[motor]:>{COLUMN}}" for motor in MOTOR_ORDER))


def _print_git(rollout: Rollout) -> None:
    if rollout.git_status is None:
        return

    _banner("git status at launch")
    print(rollout.git_status)


def _print_state_traces(rollout: Rollout) -> None:
    data = rollout.data
    seconds = _seconds(data)

    _banner(f"per-state traces, {len(data['step'])} states at the cartesian rate")

    print(f"\nstep / primitive_index / image_path")
    print(f"{'i':>{INDEX}}{'t s':>10}{'step':>8}{'primitive':>12}   image_path")
    for index, second in enumerate(seconds):
        print(f"{index:>{INDEX}}{second:>10.3f}{data['step'][index]:>8}{data['primitive_index'][index]:>12}   {data['image_path'][index]}")

    _trace("joint_positions, radians", MOTOR_ORDER, data["joint_positions"], seconds)
    _trace("joint_velocities, radians per second", MOTOR_ORDER, data["joint_velocities"], seconds)
    _trace("sensor_load, fraction of full duty", MOTOR_ORDER, data["sensor_load"], seconds)
    _trace("sensor_voltage, volts", MOTOR_ORDER, data["sensor_voltage"], seconds)
    _trace("sensor_temperature, celsius", MOTOR_ORDER, data["sensor_temperature"], seconds)
    _trace("end_effector_pose, 10d [xyz, rot6d, gripper]", _numbered("p", 10), data["end_effector_pose"], seconds)

    interval_ms = np.concatenate([[np.nan], np.diff(data["sensor_sample_time_ns"].astype(np.int64)) / 1e6])
    read_ms = (data["sensor_read_completed_ns"] - data["sensor_read_started_ns"]) / 1e6
    _trace("bus timing", ("interval_ms", "read_ms"), np.stack([interval_ms, read_ms], axis=1), seconds)

    if "qpos" in data:
        _trace("qpos", _numbered("q", data["qpos"].shape[1]), data["qpos"], seconds)
        _trace("qvel", _numbered("v", data["qvel"].shape[1]), data["qvel"], seconds)


def _print_cartesian_traces(rollout: Rollout) -> None:
    data = rollout.data
    seconds = _seconds(data)[: len(data["reward"])]

    _banner(f"per-transition traces, {len(data['reward'])} cartesian actions")

    actions = np.array([np.asarray(action, dtype=np.float64) for action in data["cartesian_action"]])
    _trace("cartesian_action", CARTESIAN_ACTION_NAMES[: actions.shape[1]], actions, seconds)
    _trace("vla_input_state", _numbered("s", data["vla_input_state"].shape[1]), data["vla_input_state"], seconds)
    _trace("reward and primitive completion", ("reward", "completes"), np.stack([data["reward"], data["completes_active_primitive"].astype(float)], axis=1), seconds)

    print("\nprimitive_prompt")
    for index, prompt in enumerate(data["primitive_prompt"]):
        print(f"{index:>{INDEX}}   {prompt}")

    print("\ncartesian_action_diagnostics")
    for index, diagnostics in enumerate(data["cartesian_action_diagnostics"]):
        print(f"{index:>{INDEX}}   " + "  ".join(f"{key}={_format(value)}" for key, value in diagnostics.items()))


def _print_dense_traces(rollout: Rollout) -> None:
    data = rollout.data
    steps = dense_steps(data)
    if not steps:
        return

    _banner(f"per-dense-step traces, {len(steps)} low level joint steps")

    _trace("action, policy output before duty scaling", MOTOR_ORDER, stack_dense(steps, "action"))
    _trace("requested_duty, before safety processing", MOTOR_ORDER, stack_dense(steps, "requested_duty"))
    _trace("end_effector_pose", _numbered("p", 10), stack_dense(steps, "end_effector_pose"))
    _trace("next_end_effector_pose", _numbered("p", 10), stack_dense(steps, "next_end_effector_pose"))

    breakdown_keys = sorted({key for step in steps for key in step["reward_breakdown"]})
    rows = np.array([[step["reward"]] + [step["reward_breakdown"].get(key, np.nan) for key in breakdown_keys] for step in steps])
    _trace("reward and its breakdown", ("reward", *breakdown_keys), rows)

    terminated = np.flatnonzero([step["terminated"] for step in steps])
    print(f"\nterminated at dense steps: {terminated.tolist()}")

    for name in sorted(steps[0]["obs"]):
        _trace(f"obs.{name}", _numbered(name[:6], np.atleast_1d(steps[0]["obs"][name]).size), np.array([np.atleast_1d(step["obs"][name]) for step in steps]))


def _print_waypoints(rollout: Rollout) -> None:
    data = rollout.data
    if "waypoints" not in data:
        return

    _banner("waypoints")
    for index, waypoint in enumerate(data["waypoints"]):
        print(f"{index:>{INDEX}}   {np.array2string(np.asarray(waypoint), precision=5, max_line_width=200)}")


def _print_log(rollout: Rollout) -> None:
    if rollout.log is None:
        return

    _banner(f"log, {len(rollout.log.splitlines())} lines")
    print(rollout.log)


def _trace(title: str, column_names, rows, seconds=None) -> None:
    rows = np.asarray(rows, dtype=np.float64)
    print(f"\n{title}   {len(rows)} rows")

    header = f"{'i':>{INDEX}}" + (f"{'t s':>10}" if seconds is not None else "")
    print(header + "".join(f"{str(name):>{COLUMN}}" for name in column_names))

    for index, row in enumerate(rows):
        line = f"{index:>{INDEX}}" + (f"{seconds[index]:>10.3f}" if seconds is not None else "")
        print(line + "".join(f"{value:>{COLUMN}.5f}" for value in np.atleast_1d(row)))


def _print_columns(names, label_width: int) -> None:
    print(f"{'':<{label_width}}" + "".join(f"{name:>{COLUMN}}" for name in names))


def _seconds(data) -> np.ndarray:
    sample_ns = data["sensor_sample_time_ns"].astype(np.int64)
    return (sample_ns - sample_ns[0]) / 1e9


def _numbered(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}{index}" for index in range(count))


def _format(value) -> str:
    if isinstance(value, float):
        return f"{value:.5f}"
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=5, max_line_width=200)
    return str(value)


def _banner(title: str) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
