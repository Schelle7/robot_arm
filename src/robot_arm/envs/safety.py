import math
import numpy as np
import csv
from pathlib import Path
from typing import Dict
from robot_arm.backends.arm import Arm


class SafetyException(Exception):
    """Raised when a dynamic hardware constraint (e.g., duty, temp) is violated."""
    pass


class SafeArmWrapper(Arm):
    """
    Wraps an Arm interface to enforce safety bounds on commanded actions
    and hardware readings.
    If an action violates bounds, it is clipped before being sent to hardware.
    If the hardware reports a dangerous state, an emergency stop is triggered.
    """

    def __init__(
        self,
        backend_arm: Arm,
        max_temperature: float,
        duty_ema_seconds: float,
        max_smoothed_duty: float,
        read_hz: float,
    ):
        self.backend_arm = backend_arm
        self.joint_limits: Dict[str, tuple[float, float]] = {
            name: (float(backend_arm.model.jnt_range[joint_id][0]), float(backend_arm.model.jnt_range[joint_id][1])) for name, joint_id in backend_arm.joint_indices.items()
        }

        self.max_temperature = max_temperature
        self.max_smoothed_duty = max_smoothed_duty
        self.control_step_seconds = 1.0 / read_hz
        # Derived from the read rate so the averaging window stays a fixed duration. A bare alpha
        # would mean 2 s at 5 Hz and 0.1 s at 100 Hz, while what overheats a servo is seconds of duty.
        self.duty_ema_alpha = 1.0 - math.exp(-1.0 / (read_hz * duty_ema_seconds))

        # Exponential Moving Average for duty tracking
        # Pre-initialize based on the backend's standard motor list
        self.smoothed_duties: Dict[str, float] = {}
        for motor in self.backend_arm.read_state()["Present_Load"]:
            self.smoothed_duties[motor] = 0.0

    def get_tcp(self) -> np.ndarray:
        return self.backend_arm.get_tcp()

    def get_tcp_pose(self, state: Dict[str, Dict[str, float]]):
        return self.backend_arm.get_tcp_pose(state)

    def get_tcp_axes(self):
        return self.backend_arm.get_tcp_axes()

    def read_state(self) -> Dict[str, Dict[str, float]]:
        state = self.backend_arm.read_state()

        for motor in state["Present_Load"]:
            self._check_temperature(motor, state)
            self._update_and_check_duty_ema(motor, state)

        return state

    def disconnect(self):
        self.backend_arm.disconnect()

    def advance_control_step(self) -> None:
        self.backend_arm.advance_control_step()

    def restore_sim_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        self.smoothed_duties = {motor: 0.0 for motor in self.smoothed_duties}
        self.backend_arm.restore_sim_state(qpos, qvel)

    def write_duty(self, duties: Dict[str, float], positions: Dict[str, float]) -> Dict[str, float]:
        """
        A duty specifies force, not a place to stop, so the joint limits can only be enforced by
        refusing the duties that drive further past one.
        """
        safe_duties = {}
        for motor, duty in duties.items():
            lower, upper = self.joint_limits[motor]
            position = positions[motor]
            duty = float(duty)
            if (position <= lower and duty < 0.0) or (position >= upper and duty > 0.0):
                duty = 0.0
            safe_duties[motor] = duty

        self.backend_arm.write_duty(safe_duties)

        return safe_duties

    def _check_temperature(self, motor: str, state: Dict[str, Dict[str, float]]):
        temp = state["Present_Temperature"][motor]
        if temp > self.max_temperature:
            self._trigger_emergency_stop(f"Motor {motor} temperature {temp}C exceeds limit {self.max_temperature}C")

    def _update_and_check_duty_ema(self, motor: str, state: Dict[str, Dict[str, float]]):
        # Assumes duties are normalized floats (-1.0 to 1.0). If they are raw ticks, they need to be pre-scaled.
        current_duty = abs(state["Present_Load"][motor])

        prev = self.smoothed_duties[motor]
        new_smoothed = self.duty_ema_alpha * current_duty + (1 - self.duty_ema_alpha) * prev
        self.smoothed_duties[motor] = new_smoothed

        if new_smoothed > self.max_smoothed_duty:
            self._trigger_emergency_stop(f"Motor {motor} sustained duty {new_smoothed:.2f} exceeds limit {self.max_smoothed_duty:.2f}")

    def _trigger_emergency_stop(self, reason: str):
        """
        Sends an immediate disconnect/torque-off signal to the hardware, then crashes python.
        """
        self.backend_arm.disconnect()
        raise SafetyException(f"EMERGENCY STOP TRIGGERED: {reason}")

    # Proxy all other attribute accesses to the inner backend
    def __getattr__(self, name):
        return getattr(self.backend_arm, name)
