import numpy as np
from typing import Dict
from robot_arm.backends.arm import Arm


class SafetyException(Exception):
    """Raised when a dynamic hardware constraint (e.g., load, temp) is violated."""

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
        min_pos: float,
        max_pos: float,
        max_temperature: float,
        load_ema_alpha: float,
        max_smoothed_load: float,
    ):
        self.backend_arm = backend_arm
        self.min_pos = min_pos
        self.max_pos = max_pos

        self.max_temperature = max_temperature
        self.load_ema_alpha = load_ema_alpha
        self.max_smoothed_load = max_smoothed_load

        # Exponential Moving Average for load tracking
        # Pre-initialize based on the backend's standard motor list
        self.smoothed_loads: Dict[str, float] = {}
        for motor in self.backend_arm.read_state()["Present_Load"]:
            self.smoothed_loads[motor] = 0.0

    def get_tcp(self) -> np.ndarray:
        return self.backend_arm.get_tcp()

    def read_state(self) -> Dict[str, Dict[str, float]]:
        state = self.backend_arm.read_state()

        for motor in state["Present_Load"]:
            self._check_temperature(motor, state)
            self._update_and_check_load_ema(motor, state)

        return state

    def disconnect(self):
        self.backend_arm.disconnect()

    def write_goal(self, positions: Dict[str, float]) -> Dict[str, float]:
        safe_positions = {}
        for motor, pos in positions.items():
            # Clip position to absolute safety bounds
            safe_pos = max(self.min_pos, min(self.max_pos, float(pos)))
            safe_positions[motor] = safe_pos

        # Forward safely clamped commands down the chain
        self.backend_arm.write_goal(safe_positions)

        return safe_positions

    def _check_temperature(self, motor: str, state: Dict[str, Dict[str, float]]):
        temp = state["Present_Temperature"][motor]
        if temp > self.max_temperature:
            self._trigger_emergency_stop(
                f"Motor {motor} temperature {temp}C exceeds limit {self.max_temperature}C"
            )

    def _update_and_check_load_ema(
        self, motor: str, state: Dict[str, Dict[str, float]]
    ):
        # Assumes loads are normalized floats (-1.0 to 1.0). If they are raw ticks, they need to be pre-scaled.
        current_load = abs(state["Present_Load"][motor])

        prev = self.smoothed_loads[motor]
        new_smoothed = (
            self.load_ema_alpha * current_load + (1 - self.load_ema_alpha) * prev
        )
        self.smoothed_loads[motor] = new_smoothed

        if new_smoothed > self.max_smoothed_load:
            self._trigger_emergency_stop(
                f"Motor {motor} sustained load {new_smoothed:.2f} exceeds limit {self.max_smoothed_load:.2f}"
            )

    def _trigger_emergency_stop(self, reason: str):
        """
        Sends an immediate disconnect/torque-off signal to the hardware, then crashes python.
        """
        self.backend_arm.disconnect()
        raise SafetyException(f"EMERGENCY STOP TRIGGERED: {reason}")

    # Proxy all other attribute accesses to the inner backend
    def __getattr__(self, name):
        return getattr(self.backend_arm, name)
