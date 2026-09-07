MOTOR_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

CAMERA_NAMES = ("external_camera", "wrist_camera")

POLICY_OBSERVATION_NAMES = (
    "joint_positions",
    "joint_velocities",
    "previous_action",
    "remaining_delta",
    "time_left",
    "tcp_velocity",
    "duty_history",
    "gripper_duty",
    "desired_gripper_duty",
    "desired_gripper_duty_active",
    "gripper_duty_difference",
)


def policy_observation_sizes(cartesian_action_dim: int) -> dict[str, int]:
    return {
        "joint_positions": 6,
        "joint_velocities": 6,
        "previous_action": 6,
        "remaining_delta": cartesian_action_dim,
        "time_left": 1,
        "tcp_velocity": 6,
        "duty_history": 6,
        "gripper_duty": 1,
        "desired_gripper_duty": 1,
        "desired_gripper_duty_active": 1,
        "gripper_duty_difference": 1,
    }


def policy_observation_dim(cartesian_action_dim: int) -> int:
    sizes = policy_observation_sizes(cartesian_action_dim)
    assert tuple(sizes) == POLICY_OBSERVATION_NAMES
    return sum(sizes.values())

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
