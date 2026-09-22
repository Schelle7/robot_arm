import logging
import time
from contextlib import ExitStack
from typing import Dict

import mujoco
import numpy as np

from robot_arm.arms.arm import Arm
from robot_arm.arms.camera_capture import CameraCapture
from robot_arm.arms.real_communication import RealCommunication
from robot_arm.geometry.gripper_geometry import get_tcp_geometry
from robot_arm.geometry.pose import Pose
from robot_arm.robot_schema import MOTOR_ORDER

log = logging.getLogger(__name__)


class RealArm(Arm):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.last_control_step_end = time.perf_counter()
        self.control_step_overruns = 0
        self.qpos_indices = self.model.jnt_qposadr[np.array([self.joint_indices[name] for name in MOTOR_ORDER])]
        self.communication = RealCommunication(cfg, self.joint_limits)
        self.camera_cleanup = ExitStack()

    def connect_cameras(self, cfg):
        self.cameras = CameraCapture(cfg)
        self.camera_cleanup.callback(self.cameras.close)

    def _forward_kinematics_pose(self, present_positions: dict) -> Pose:
        self.data.qpos[self.qpos_indices] = np.array([present_positions[name] for name in MOTOR_ORDER])
        mujoco.mj_kinematics(self.model, self.data)
        pose, _, _ = get_tcp_geometry(self.model, self.data)
        return pose

    def get_end_effector_pose_7d_forward_kinematics(self, present_positions: dict) -> np.ndarray:
        pose = self._forward_kinematics_pose(present_positions)
        return np.concatenate([pose.position, pose.as_euler("XYZ", False), [pose.gripper]]).astype(np.float32)

    def get_tcp_pose(self, state: Dict[str, Dict[str, float]]) -> Pose:
        return self._forward_kinematics_pose(state["Present_Position"])

    def get_tcp(self) -> np.ndarray:
        raise NotImplementedError("Real arm does not have access to pinch point")

    def read_cameras(self) -> dict[str, np.ndarray]:
        return self.cameras.read()

    def advance_control_step(self) -> None:
        # The servos run their own loop, so a control period is wall-clock time rather than steps.
        # Paced from the end of the previous period so the caller's own work counts against the
        # budget instead of being added on top of it.
        remaining = (self.last_control_step_end + self.control_step_seconds) - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        else:
            self.control_step_overruns += 1
            if self.control_step_overruns % 100 + 1 == 1:
                log.warning(
                    "Control step overran its %.1f ms budget by %.1f ms. The loop is running slower than the configured rate; further overruns are counted, not logged.",
                    self.control_step_seconds * 1e3,
                    -remaining * 1e3,
                )
        self.last_control_step_end = time.perf_counter()

    def disconnect(self):
        try:
            self.communication.close()
        finally:
            self.camera_cleanup.close()
