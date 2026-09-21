MOTOR_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

CAMERA_NAMES = ("external_camera", "wrist_camera")

POLICY_OBSERVATION_NAMES = ("history", "state", "goal")

HISTORY_CONTEXT_FEATURE_NAMES = tuple(f"current_joint_position_{motor}" for motor in MOTOR_ORDER)

HISTORY_POLICY_ACTION_SLICE = slice(0, len(MOTOR_ORDER))
HISTORY_APPLIED_DUTY_SLICE = slice(len(MOTOR_ORDER), 2 * len(MOTOR_ORDER))
HISTORY_JOINT_VELOCITY_SLICE = slice(2 * len(MOTOR_ORDER), 3 * len(MOTOR_ORDER))
HISTORY_TCP_VELOCITY_SLICE = slice(3 * len(MOTOR_ORDER), 4 * len(MOTOR_ORDER))

HISTORY_FEATURE_NAMES = (
    *[f"policy_action_{motor}" for motor in MOTOR_ORDER],
    *[f"applied_duty_{motor}" for motor in MOTOR_ORDER],
    *[f"joint_velocity_{motor}" for motor in MOTOR_ORDER],
    "tcp_velocity_x",
    "tcp_velocity_y",
    "tcp_velocity_z",
    "tcp_velocity_rx",
    "tcp_velocity_ry",
    "tcp_velocity_rz",
)

STATE_FEATURE_NAMES = (
    *[f"joint_position_{motor}" for motor in MOTOR_ORDER],
    *[f"joint_velocity_{motor}" for motor in MOTOR_ORDER],
    "tcp_position_x",
    "tcp_position_y",
    "tcp_position_z",
    "tcp_velocity_x",
    "tcp_velocity_y",
    "tcp_velocity_z",
    "tcp_velocity_rx",
    "tcp_velocity_ry",
    "tcp_velocity_rz",
    "gripper_duty",
)

STATE_JOINT_POSITION_SLICE = slice(0, len(MOTOR_ORDER))
STATE_JOINT_VELOCITY_SLICE = slice(len(MOTOR_ORDER), 2 * len(MOTOR_ORDER))
STATE_TCP_POSITION_SLICE = slice(2 * len(MOTOR_ORDER), 2 * len(MOTOR_ORDER) + 3)
STATE_TCP_VELOCITY_SLICE = slice(2 * len(MOTOR_ORDER) + 3, 3 * len(MOTOR_ORDER) + 3)
STATE_GRIPPER_DUTY_SLICE = slice(3 * len(MOTOR_ORDER) + 3, 3 * len(MOTOR_ORDER) + 4)


def policy_observation_sizes(cartesian_action_dim: int, history_steps: int) -> dict[str, int]:
    return {
        "history": len(HISTORY_CONTEXT_FEATURE_NAMES) + len(HISTORY_FEATURE_NAMES) * history_steps,
        "state": len(STATE_FEATURE_NAMES),
        "goal": cartesian_action_dim + 4,
    }


BOX_BODY_NAMES = ("box_0",)

TILE_BODY_NAME = "tile"

# Drawn without replacement across the boxes and the tile, so every object in a scene reads differently.
OBJECT_COLORS = ("red", "green", "blue", "yellow", "purple")

CARTESIAN_ACTION_NAMES = (
    "x",
    "y",
    "z",
    "rx",
    "ry",
    "rz",
    "gripper",
)

PRIMITIVE_COMPLETION = "observation.environment_state"  # hacky way to use smolvla

DESIRED_GRIPPER_DUTY_NAMES = (
    "desired_gripper_duty",
    "desired_gripper_duty_active",
)

VLA_ACTION_NAMES = (*CARTESIAN_ACTION_NAMES, PRIMITIVE_COMPLETION, *DESIRED_GRIPPER_DUTY_NAMES)

CURRENT_POSE_NAMES = (
    "current_x",
    "current_y",
    "current_z",
    "current_roll",
    "current_pitch",
    "current_yaw",
    "current_gripper",
)

TARGET_OFFSET_NAMES = (
    "target_offset_x",
    "target_offset_y",
    "target_offset_z",
    "target_offset_roll",
    "target_offset_pitch",
    "target_offset_yaw",
    "target_offset_gripper",
    "includes_target_offset",
)

DUTY_NAMES = ("gripper_duty",)
