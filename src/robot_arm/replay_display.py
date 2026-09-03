import numpy as np

from robot_arm.backends.sim_arm import get_tcp_geometry
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
    dense_sample = dense_trajectory[-1] if len(dense_trajectory) else {}
    low_level_action = dense_sample["action"] if dense_sample else None
    duty_compensation = dense_sample["duty_compensation"] if dense_sample else None
    compensated_duty_action = dense_sample["compensated_duty"] if dense_sample else None
    low_level_observation = dense_sample["obs"] if dense_sample else {}
    episode_time = frame_index / recorded_cfg.control.frequencies.cartesian
    primitive_rows = [["Completes active primitive", str(bool(completes_active_primitive))]]
    if action_diagnostics:
        primitive_rows.extend([key.replace("_", " ").title(), format_value(value)] for key, value in action_diagnostics.items())
    else:
        primitive_rows.append(["Status", "No outgoing transition"])

    observation_rows = [[key, "  ".join(format_vector(value))] for key, value in low_level_observation.items()]
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
                ["Policy", str(recorded_cfg.policy_name)],
                ["Episode time", f"{episode_time:.2f} s"],
                ["Frame", str(frame_index)],
                ["TCP gripper", format_value(live_pose.gripper)],
            ],
        ),
        section(
            "TCP pose",
            ["Signal", "X", "Y", "Z"],
            [
                vector_row("Position", live_pose.position, 3),
                vector_row("Orientation XYZ", live_pose.as_euler("XYZ", False), 3),
            ],
        ),
        section(
            "Joint state",
            ["Signal", *MOTOR_ORDER],
            [
                vector_row("Position", joint_positions, len(MOTOR_ORDER)),
                vector_row("Velocity", joint_velocities, len(MOTOR_ORDER)),
                vector_row("Policy action", low_level_action, len(MOTOR_ORDER)),
                vector_row("Duty compensation", duty_compensation, len(MOTOR_ORDER)),
                vector_row("Compensated duty", compensated_duty_action, len(MOTOR_ORDER)),
            ],
        ),
        section(
            "Cartesian action",
            ["Signal", *CARTESIAN_ACTION_NAMES],
            [vector_row("Desired pose", cartesian_action, len(CARTESIAN_ACTION_NAMES))],
        ),
        section(
            "Transition",
            ["Signal", "X", "Y", "Z", "Primary orientation", "Secondary orientation", "Gripper"],
            [
                vector_row("Observed delta", observed_pose_delta, 6),
                vector_row("Tracking error", pose_tracking_error, 6),
            ],
        ),
        section("Primitive", ["Metric", "Value"], primitive_rows),
        section("Low-level observation", ["Input", "Values"], observation_rows),
        section("Last low-level reward", ["Component", "Value"], reward_rows),
    ]
    return sections, []
