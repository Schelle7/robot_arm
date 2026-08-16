import shutil
import numpy as np

from robot_arm.backends.sim_arm import get_tcp_geometry
from robot_arm.robot_schema import CARTESIAN_ACTION_NAMES, MOTOR_ORDER


VALUE_WIDTH = 12
VALUE_PRECISION = 4
VALUE_SEPARATOR = "  "
ACTION_WIDTH = 9
ACTION_SEPARATOR = " "
REPLAY_CONTROLS = """\
Controls:
    Right Arrow : Next frame
    Left Arrow  : Previous frame
    Space       : Auto-play toggle
    Esc         : Quit
"""


def format_values(values):
    return VALUE_SEPARATOR.join(
        f"{float(value):+{VALUE_WIDTH}.{VALUE_PRECISION}f}"
        for value in np.asarray(values).reshape(-1)
    )


def format_action_values(values):
    return ACTION_SEPARATOR.join(
        f"{float(value):+{ACTION_WIDTH}.{VALUE_PRECISION}f}"
        for value in np.asarray(values).reshape(-1)
    )


def format_desired_pose(cartesian_action):
    return (
        f"desired_pose=[{format_action_values(cartesian_action)}]"
        if cartesian_action is not None
        else "desired_pose: not available for this step"
    )


def format_pose_delta(delta):
    return (
        f"position=[{format_values(delta[:3])}] "
        f"primary_orientation={delta[3]:+.4f}rad "
        f"secondary_orientation={delta[4]:+.4f}rad "
        f"gripper={delta[5]:+.4f}"
    )


def format_low_level_observation(observation):
    if not observation:
        return "low_level_observation: not available for this step"

    values = "  ".join(f"{key}=[{format_values(value)}]" for key, value in observation.items())
    return f"low_level_observation: {values}"


def format_primitive_status(action_diagnostics, completes_active_primitive):
    if not action_diagnostics:
        return "primitive: no outgoing transition"

    return (
        f"completes_active_primitive={completes_active_primitive} "
        f"position={action_diagnostics['position_distance']:.4f}m/"
        f"{action_diagnostics['position_threshold']:.4f}m "
        f"primary_orientation={action_diagnostics['primary_orientation_distance']:.4f}/"
        f"{action_diagnostics['orientation_threshold']:.4f}rad "
        f"secondary_orientation={action_diagnostics['secondary_orientation_distance']:.4f}/"
        f"{action_diagnostics['orientation_threshold']:.4f}rad "
        f"gripper={action_diagnostics['gripper_distance']:.4f}/"
        f"{action_diagnostics['gripper_threshold']:.4f}"
    )


def format_transition_line(label, value):
    if value is None:
        return f"{label}: not available for this step"
    return f"{label}: {format_pose_delta(value)}"


def format_reward_line(dense_trajectory):
    if not dense_trajectory:
        return "last_low_level_reward: not available for this step"

    reward = dense_trajectory[-1]
    reward_parts = "  ".join(
        f"{key.removesuffix('_reward').removesuffix('_penalty')}={value:+.4f}"
        for key, value in reward["reward_breakdown"].items()
    )
    return f"last_low_level_reward={reward['reward']:+.4f}  {reward_parts}"


def format_terminal_warnings(display_lines, terminal_width, terminal_height):
    required_width = max(map(len, display_lines))
    required_height = len(display_lines)
    warnings = []
    if terminal_width == -1:
        warnings.append("WARNING: terminal width could not be determined.")
    elif terminal_width < required_width:
        warnings.append(f"WARNING: terminal width {terminal_width} is below the required {required_width} columns.")
    if terminal_height == -1:
        warnings.append("WARNING: terminal height could not be determined.")
    elif terminal_height < required_height:
        warnings.append(f"WARNING: terminal height {terminal_height} is below the required {required_height} lines.")
    return warnings


def build_replay_display(
    model,
    mdata,
    joint_positions,
    joint_velocities,
    cartesian_action,
    dense_trajectory,
    observed_pose_delta,
    pose_tracking_error,
    action_diagnostics,
    completes_active_primitive,
    frame_index,
    recorded_cfg,
):
    live_pose, _, _ = get_tcp_geometry(model, mdata)
    joint_labels = VALUE_SEPARATOR.join(f"{name:>{VALUE_WIDTH}}" for name in MOTOR_ORDER)
    cartesian_labels = ACTION_SEPARATOR.join(f"{name:>{ACTION_WIDTH}}" for name in CARTESIAN_ACTION_NAMES)
    cartesian_action_text = None if cartesian_action is None else f"[{format_action_values(cartesian_action)}]"
    dense_sample = dense_trajectory[-1] if dense_trajectory else {}
    low_level_action = dense_sample["action"] if dense_sample else None
    low_level_action_text = None if low_level_action is None else f"[{format_values(low_level_action)}]"
    low_level_observation = dense_sample["obs"] if dense_sample else {}
    episode_time = frame_index / recorded_cfg.control.frequencies.mid_level

    lines = [
        "",
        f"episode_time={episode_time:0.2f} s  frame={frame_index}",
        format_primitive_status(action_diagnostics, completes_active_primitive),
        format_low_level_observation(low_level_observation),
        format_desired_pose(cartesian_action),
        "",
        f"TCP position=[{format_values(live_pose.position)}]",
        f"TCP orientation=[{format_values(live_pose.as_euler('XYZ', False))}]",
        f"TCP gripper={live_pose.gripper:+.4f}",
        f"                 {joint_labels}",
        f"joint_positions=  {format_values(joint_positions)}",
        f"joint_velocities= {format_values(joint_velocities)}",
        f"low_level_action= {low_level_action_text}",
        f"                 {cartesian_labels}",
        f"cartesian_action= {cartesian_action_text}",
        format_transition_line("observed_pose_delta", observed_pose_delta),
        format_transition_line("pose_tracking_error", pose_tracking_error),
        format_reward_line(dense_trajectory),
    ]
    display_lines = REPLAY_CONTROLS.splitlines() + lines
    terminal_width, terminal_height = shutil.get_terminal_size(fallback=(-1, -1))
    warnings = format_terminal_warnings(display_lines, terminal_width, terminal_height)
    return display_lines, warnings
