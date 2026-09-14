import numpy as np

from robot_arm.geometry.gripper_geometry import get_tcp_geometry
from robot_arm.robot_schema import CARTESIAN_ACTION_NAMES, MOTOR_ORDER

def format_value(value):
    return f"{float(value):+.4f}"


def format_vector(values):
    return [format_value(value) for value in np.asarray(values).reshape(-1)]


def vector_row(label, values, value_count):
    if values is None:
        return [label] + ["N/A"] * value_count
    return [label] + format_vector(values)


def section(title, columns, rows):
    return {"title": title, "columns": columns, "rows": rows}


def primitive_diagnostic_rows(action_diagnostics):
    diagnostic_pairs = (
        ("Position (cm)", "position_distance", "position_threshold", 100),
        ("Primary orientation (degrees)", "primary_orientation_distance", "primary_orientation_threshold", 180 / np.pi),
        ("Secondary orientation (degrees)", "secondary_orientation_distance", "secondary_orientation_threshold", 180 / np.pi),
        ("Gripper position (degrees)", "gripper_distance", "gripper_threshold", 180 / np.pi),
        ("Gripper duty", "gripper_duty_distance", "duty_threshold", 1),
    )
    rows = [
        [
            label,
            format_value(action_diagnostics[difference_key] * scale) if difference_key in action_diagnostics else "Not available for this rollout",
            format_value(action_diagnostics[threshold_key] * scale) if threshold_key in action_diagnostics else "Not available for this rollout",
        ]
        for label, difference_key, threshold_key, scale in diagnostic_pairs
    ]
    if "completion_score" in action_diagnostics:
        rows.append(["VLA completion score", format_value(action_diagnostics["completion_score"]), "0.5"])
    if "teacher_completes_active_primitive" in action_diagnostics:
        rows.append(["Teacher considers primitive complete", str(action_diagnostics["teacher_completes_active_primitive"]), "N/A"])
    if "teacher_completion_score" in action_diagnostics:
        rows.append(["Teacher completion score", format_value(action_diagnostics["teacher_completion_score"]), "0.5"])
    return rows


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
    low_level_step,
    recorded_cfg,
):
    live_pose, _, _ = get_tcp_geometry(model, mdata)
    dense_sample = dense_trajectory[low_level_step] if len(dense_trajectory) else {}
    low_level_action = dense_sample["action"] if dense_sample else None
    requested_duty = dense_sample["requested_duty"] if dense_sample else None
    low_level_observation = dense_sample["obs"] if dense_sample else {}
    episode_time = frame_index / recorded_cfg.control.frequencies.cartesian
    primitive_rows = [["Completes active primitive", str(bool(completes_active_primitive)), "N/A"]]
    primitive_rows.extend(primitive_diagnostic_rows(action_diagnostics))

    observation_rows = [
        [key, "  ".join(format_vector(value))]
        for key, value in low_level_observation.items()
        if key != "history"
    ]
    if not observation_rows:
        observation_rows = [["Status", "Not available for this step"]]

    if dense_sample:
        reward_rows = [["Total", format_value(dense_sample["reward"])]]
        reward_rows.extend(
            [key.removesuffix("_reward").removesuffix("_penalty").replace("_", " ").title(), format_value(value)]
            for key, value in dense_sample["reward_breakdown"].items()
        )
    else:
        reward_rows = [["Total", "Not available for this step"]]

    sections = [
        section(
            "Overview",
            ["Metric", "Value"],
            [
                ["Replay type", str(recorded_cfg.backend)],
                ["Policy", str(recorded_cfg.policy_name)],
                ["Episode time", f"{episode_time:.2f} s"],
            ],
        ),
        section(
            "TCP pose",
            ["Signal", *CARTESIAN_ACTION_NAMES],
            [["Pose", *format_vector(live_pose.as_7d())]],
        ),
        section(
            "Joint state",
            ["Signal", *MOTOR_ORDER],
            [
                vector_row("Position", joint_positions, len(MOTOR_ORDER)),
                vector_row("Velocity", joint_velocities, len(MOTOR_ORDER)),
                vector_row("Policy action", low_level_action, len(MOTOR_ORDER)),
                vector_row("Requested duty", requested_duty, len(MOTOR_ORDER)),
            ],
        ),
        section(
            "Cartesian action",
            ["Signal", *CARTESIAN_ACTION_NAMES],
            [vector_row("Commanded pose delta", cartesian_action, len(CARTESIAN_ACTION_NAMES))],
        ),
        section(
            "Transition",
            ["Signal", "X", "Y", "Z", "Primary orientation", "Secondary orientation", "Gripper"],
            [
                vector_row("Observed delta", observed_pose_delta, 6),
                vector_row("Tracking error", pose_tracking_error, 6),
            ],
        ),
        section(
            "Primitive",
            ["Metric", "Value / difference", "Threshold"],
            primitive_rows,
        ),
        section("Low-level observation", ["Input", "Values"], observation_rows),
        section("Last low-level reward", ["Component", "Value"], reward_rows),
    ]
    return sections, []
