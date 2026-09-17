import abc
import math
from typing import Dict, final
import numpy as np
from robot_arm.geometry.pose import Pose
from robot_arm.robot_schema import MOTOR_ORDER


class SafetyException(Exception):
    pass


class Arm(abc.ABC):
    """
    Unified interface for hardware and simulation backends.
    """

    def __init_subclass__(cls):
        super().__init_subclass__()
        for name in ("read_state", "write_duty"):
            if name in cls.__dict__:
                raise TypeError(f"{cls.__name__} must implement _{name} instead of overriding {name}")

    def __init__(self, cfg):
        self.max_temperature = float(cfg.safety.max_temperature_celsius)
        self.max_smoothed_duty = float(cfg.safety.max_smoothed_duty)
        self.control_step_seconds = 1.0 / cfg.control.frequencies.joint
        self.duty_ema_alpha = 1.0 - math.exp(-self.control_step_seconds / float(cfg.safety.duty_ema_seconds))
        self.smoothed_duties = dict.fromkeys(MOTOR_ORDER, 0.0)

    @property
    def joint_limits(self) -> Dict[str, tuple[float, float]]:
        return {name: tuple(map(float, self.model.jnt_range[index])) for name, index in self.joint_indices.items()}

    @final
    def read_state(self) -> Dict[str, Dict[str, float]]:
        state = self._read_state()
        for motor in state["Present_Load"]:
            self._check_temperature(motor, state)
            self._check_duty(motor, state)
        return state

    def _check_temperature(self, motor: str, state: Dict[str, Dict[str, float]]) -> None:
        temperature = state["Present_Temperature"][motor]
        if temperature > self.max_temperature:
            self._trigger_emergency_stop(f"Motor {motor} temperature {temperature}C exceeds limit {self.max_temperature}C")

    def _check_duty(self, motor: str, state: Dict[str, Dict[str, float]]) -> None:
        smoothed = self.duty_ema_alpha * abs(state["Present_Load"][motor]) + (1 - self.duty_ema_alpha) * self.smoothed_duties[motor]
        self.smoothed_duties[motor] = smoothed
        if smoothed > self.max_smoothed_duty:
            self._trigger_emergency_stop(f"Motor {motor} sustained duty {smoothed:.2f} exceeds limit {self.max_smoothed_duty:.2f}")

    @abc.abstractmethod
    def _read_state(self) -> Dict[str, Dict[str, float]]:
        pass

    def _trigger_emergency_stop(self, reason: str) -> None:
        self.disconnect()
        raise SafetyException(f"EMERGENCY STOP TRIGGERED: {reason}")

    @abc.abstractmethod
    def get_tcp(self) -> np.ndarray:
        """
        Returns the Tool Center Point (TCP) pose as a 7D array:
        [x, y, z, roll, pitch, yaw, aperture]
        Raises NotImplementedError if the backend cannot compute this.
        """
        pass

    def get_tcp_pose(self, state: Dict[str, Dict[str, float]]) -> Pose:
        """
        Returns the TCP pose for an already-read state. The state is passed in so that
        backends deriving the pose from joint readings do not add a bus round trip.
        """
        raise NotImplementedError("Arm backend does not expose a TCP pose.")

    def get_tcp_axes(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Arm backend does not expose TCP axes.")

    def physics_metrics(self) -> Dict[str, float]:
        """
        Physics parameters that vary between episodes, for logging. Empty where they cannot,
        which is every backend driving real hardware.
        """
        return {}

    @abc.abstractmethod
    def read_cameras(self) -> dict[str, np.ndarray]:
        pass

    @abc.abstractmethod
    def disconnect(self):
        """
        Emergency power cutoff or safe shutdown routine.
        """
        pass

    @final
    def write_duty(self, duties: Dict[str, float], positions: Dict[str, float]) -> Dict[str, float]:
        # Suppress only outward duties at a limit, so the arm can still move back.
        safe_duties = {}
        for motor, duty in duties.items():
            lower, upper = self.joint_limits[motor]
            position = positions[motor]
            duty = float(duty)
            if (position <= lower and duty < 0.0) or (position >= upper and duty > 0.0):
                duty = 0.0
            safe_duties[motor] = duty
        self._write_duty(safe_duties)
        return safe_duties

    @abc.abstractmethod
    def _write_duty(self, duties: Dict[str, float]) -> None:
        pass

    @abc.abstractmethod
    def advance_control_step(self) -> None:
        """
        Lets exactly one joint control period elapse against the duty last written. Simulation
        advances physics, hardware waits, so callers do not need to know which backend they hold.
        """
        pass
