MOTOR_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

BOX_BODY_NAMES = ("box_0", "box_1")

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
