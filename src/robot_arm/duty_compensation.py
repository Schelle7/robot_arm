import mujoco
import numpy as np

from robot_arm.backends.servo import torque_to_duty
from robot_arm.robot_schema import MOTOR_ORDER


class DutyCompensator:
    def __init__(self, model: mujoco.MjModel, stall_torque_newton_meters: float):
        self.model = model
        self.data = mujoco.MjData(model)
        joint_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in MOTOR_ORDER])
        self.qpos_indices = model.jnt_qposadr[joint_ids]
        self.qvel_indices = model.jnt_dofadr[joint_ids]
        self.stall_torque_newton_meters = stall_torque_newton_meters

    def calculate(self, joint_positions: np.ndarray, joint_velocities: np.ndarray) -> np.ndarray:
        self.data.qpos[self.qpos_indices] = joint_positions
        self.data.qvel[self.qvel_indices] = joint_velocities
        mujoco.mj_forward(self.model, self.data)
        compensation = torque_to_duty(
            self.data.qfrc_bias[self.qvel_indices],
            self.stall_torque_newton_meters,
        ).astype(np.float32)
        compensation[MOTOR_ORDER.index("gripper")] = 0.0
        return compensation