from typing import Dict

import mujoco
import numpy as np

from robot_arm.arms.communication import Communication
from robot_arm.arms.servo import duty_from_action, duty_to_torque
from robot_arm.robot_schema import MOTOR_ORDER


class SimCommunication(Communication):
    def __init__(self, cfg, model, data):
        super().__init__(cfg)
        self.model = model
        self.data = data
        self.servo = cfg.servo
        self.joint_indices = {name: model.joint(name).id for name in MOTOR_ORDER}
        # Build explicit mappings for actuator and joint indices
        self.actuator_indices = {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(self.model.nu)}

        # Indexed in actuator order so the servo law runs on all six joints as one vector operation.
        self.actuator_order = sorted(self.actuator_indices, key=self.actuator_indices.get)
        self.actuator_dof_indices = np.array([self.model.jnt_dofadr[self.joint_indices[name]] for name in self.actuator_order])
        self.max_duty = np.array([float(cfg.servo.max_duty[name]) for name in self.actuator_order])
        self.commanded_duty = np.zeros(self.model.nu)

    def reset(self):
        self.last_safety_read_ns = None
        self.smoothed_duties = dict.fromkeys(MOTOR_ORDER, 0.0)
        for samples in self.temperature_samples.values():
            samples.clear()
        self.temperature_totals = dict.fromkeys(MOTOR_ORDER, 0.0)
        self.commanded_duty[:] = 0.0

    def read_sensors(self) -> Dict[str, Dict[str, float]]:
        # Map MuJoCo qpos, qvel and the commanded duty to our expected dictionary format
        sample_time_ns = round(self.data.time * 1_000_000_000)
        state = {
            "Present_Position": {},
            "Present_Velocity": {},
            "Present_Load": {},  # Returning actuator control effort as load
            "Present_Voltage": {},  # Dummy data
            "Present_Temperature": {},  # Dummy data
            "read_started_ns": sample_time_ns,
            "read_completed_ns": sample_time_ns,
            "sample_time_ns": sample_time_ns,
            "feedback_age_seconds": 0.0,
        }

        for name, actuator_idx in self.actuator_indices.items():
            qpos_idx = self.model.jnt_qposadr[self.joint_indices[name]]
            qvel_idx = self.model.jnt_dofadr[self.joint_indices[name]]

            state["Present_Position"][name] = float(self.data.qpos[qpos_idx])
            state["Present_Velocity"][name] = float(self.data.qvel[qvel_idx])
            # The duty the servo law commanded, which is what Present_Load reports on the real arm:
            # a fraction of full output, not a torque.
            state["Present_Load"][name] = float(self.commanded_duty[actuator_idx] / self.servo.full_scale_duty)

            state["Present_Voltage"][name] = 12.0
            state["Present_Temperature"][name] = 40.0

        return state

    def send_duty(self, duties: Dict[str, float]) -> None:
        requested = np.zeros(self.model.nu)
        for name, duty_fraction in duties.items():
            requested[self.actuator_indices[name]] = duty_fraction
        self.commanded_duty = duty_from_action(
            requested,
            self.servo.full_scale_duty,
            self.servo.min_startup_duty,
            self.max_duty,
        )

    def apply_servo_torques(self) -> None:
        """
        The duty is held for the whole control period the way the servo's open loop PWM mode holds
        it, so only the back-EMF term varies as the joint picks up speed.
        """
        self.data.ctrl[:] = duty_to_torque(
            self.commanded_duty,
            self.data.qvel[self.actuator_dof_indices],
            self.servo.stall_torque_newton_meters,
            self.servo.no_load_speed_radians_per_second,
        )

    def get_state(self):
        state = self.read_sensors()
        self.check_safety(state)
        return state

    def submit_duty(self, duties, sample_time_ns):
        self.send_duty(duties)

    def close(self):
        pass
