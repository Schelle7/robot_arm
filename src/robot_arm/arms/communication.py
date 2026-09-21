import math
from abc import ABC, abstractmethod
from typing import Dict

from robot_arm.robot_schema import MOTOR_ORDER


class SafetyException(Exception):
    pass


class Communication(ABC):
    def __init__(self, cfg):
        self.max_temperature = float(cfg.safety.max_temperature_celsius)
        self.max_smoothed_duty = float(cfg.safety.max_smoothed_duty)
        self.control_step_seconds = 1.0 / cfg.control.frequencies.joint
        self.duty_ema_seconds = float(cfg.safety.duty_ema_seconds)
        self.last_safety_read_ns = None
        self.smoothed_duties = dict.fromkeys(MOTOR_ORDER, 0.0)

    def check_safety(self, state) -> None:
        for motor in state["Present_Load"]:
            self._check_temperature(motor, state)
        completed_ns = state["read_completed_ns"]
        if completed_ns == self.last_safety_read_ns:
            return
        elapsed_seconds = self.control_step_seconds if self.last_safety_read_ns is None else (completed_ns - self.last_safety_read_ns) / 1_000_000_000
        self.duty_ema_alpha = 1.0 - math.exp(-elapsed_seconds / self.duty_ema_seconds)
        for motor in state["Present_Load"]:
            self._check_duty(motor, state)
        self.last_safety_read_ns = completed_ns

    def _check_temperature(self, motor: str, state: Dict[str, Dict[str, float]]) -> None:
        temperature = state["Present_Temperature"][motor]
        if temperature > self.max_temperature:
            self._trigger_emergency_stop(f"Motor {motor} temperature {temperature}C exceeds limit {self.max_temperature}C")

    def _check_duty(self, motor: str, state: Dict[str, Dict[str, float]]) -> None:
        smoothed = self.duty_ema_alpha * abs(state["Present_Load"][motor]) + (1 - self.duty_ema_alpha) * self.smoothed_duties[motor]
        self.smoothed_duties[motor] = smoothed
        if smoothed > self.max_smoothed_duty:
            self._trigger_emergency_stop(f"Motor {motor} sustained duty {smoothed:.2f} exceeds limit {self.max_smoothed_duty:.2f}")

    def _trigger_emergency_stop(self, reason: str) -> None:
        self.close()
        raise SafetyException(f"EMERGENCY STOP TRIGGERED: {reason}")

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def submit_duty(self, duties, sample_time_ns):
        pass

    @abstractmethod
    def close(self):
        pass
