from typing import Dict

import mujoco
import numpy as np

from robot_arm.arms.communication import Communication
from robot_arm.arms.servo import duty_from_action, duty_to_torque, supply_voltage_and_current
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
        self.reset()

    def reset(self):
        self.last_safety_read_ns = None
        self.smoothed_duties = dict.fromkeys(MOTOR_ORDER, 0.0)
        for samples in self.temperature_samples.values():
            samples.clear()
        self.temperature_totals = dict.fromkeys(MOTOR_ORDER, 0.0)
        self.commanded_duty[:] = 0.0
        self.braking_multiplier = np.random.uniform(*self.servo.physics_randomization.braking_multiplier, size=self.model.nu)
        assert np.all(self.braking_multiplier >= 0)
        self.motor_strength_multiplier = np.random.uniform(*self.servo.physics_randomization.motor_strength_multiplier, size=self.model.nu)
        assert np.all(self.motor_strength_multiplier > 0)
        ranges = self.servo.physics_randomization.supply
        self.source_voltage = np.random.uniform(*ranges.source_voltage_volts)
        self.voltage_drop = np.random.uniform(*ranges.voltage_drop_volts_per_amp)
        current_ranges = np.array([ranges.current_scale_amps_per_volt[name] for name in self.actuator_order])
        emf_ranges = np.array([ranges.back_emf_volts_per_radian_per_second[name] for name in self.actuator_order])
        self.current_scale = np.random.uniform(current_ranges[:, 0], current_ranges[:, 1])
        self.back_emf = np.random.uniform(emf_ranges[:, 0], emf_ranges[:, 1])
        assert self.source_voltage > 0 and self.servo.nominal_voltage_volts > 0
        assert self.voltage_drop >= 0 and np.all(self.current_scale >= 0) and np.all(self.back_emf >= 0)
        # Ensures the coupled supply equation is monotonic and has a unique clamped solution.
        assert self.voltage_drop * np.sum(self.current_scale * self.max_duty / self.servo.full_scale_duty) < 1

    def _supply_state(self):
        return supply_voltage_and_current(
            self.commanded_duty / self.servo.full_scale_duty,
            self.data.qvel[self.actuator_dof_indices],
            self.source_voltage,
            self.voltage_drop,
            self.current_scale,
            self.back_emf,
        )

    def read_sensors(self) -> Dict[str, Dict[str, float]]:
        # Map MuJoCo qpos, qvel and the commanded duty to our expected dictionary format
        sample_time_ns = round(self.data.time * 1_000_000_000)
        voltage, currents = self._supply_state()
        state = {
            "Present_Position": {},
            "Present_Velocity": {},
            "Present_Load": {},  # Returning actuator control effort as load
            "Present_Current": {},
            "Present_Voltage": {},
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

            state["Present_Voltage"][name] = voltage
            state["Present_Current"][name] = float(currents[actuator_idx])
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
        Recompute supply sag as motor speeds change even while the command is held constant.
        """
        voltage, _ = self._supply_state()
        self.data.ctrl[:] = self.motor_strength_multiplier * duty_to_torque(
            self.commanded_duty,
            self.data.qvel[self.actuator_dof_indices],
            self.servo.stall_torque_newton_meters,
            self.servo.no_load_speed_radians_per_second,
            voltage / self.servo.nominal_voltage_volts,
            self.braking_multiplier,
        )

    def get_state(self):
        state = self.read_sensors()
        self.check_safety(state)
        return state

    def submit_duty(self, duties, sample_time_ns):
        self.send_duty(duties)

    def close(self):
        pass
