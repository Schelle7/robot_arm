from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from robot_arm.geometry.pose import Pose


@dataclass
class EnvironmentState:
    observation: Dict[str, np.ndarray]
    sensor_state: Dict[str, Any]
    end_effector_pose: Pose
    sim_state: Dict[str, np.ndarray] | None
    grasp_confirmed: bool


@dataclass
class CartesianAction:
    cartesian_action: np.ndarray
    diagnostics: Dict[str, Any]
    completes_active_primitive: bool
    desired_gripper_duty: float
    desired_gripper_duty_active: bool


@dataclass
class ActionPrimitive:
    start_pose: Pose
    target_pose: Pose
    prompt: str
    include_target_offset: bool
    desired_gripper_duty: float
    desired_gripper_duty_active: bool
