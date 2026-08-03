import numpy as np
import time
import csv
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

    def get_tcp_pose(self):
        return self.backend_arm.get_tcp_pose()

    def get_tcp_axes(self):
        return self.backend_arm.get_tcp_axes()

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

    def move_to_staging_pose(
        self,
        initial_joint_range_percent: tuple[float, float],
        speed_radians_per_second: float,
        tolerance_radians: float,
        max_steps: int,
        pause_seconds: float,
        log_path: str,
    ) -> None:
        min_percent, max_percent = initial_joint_range_percent
        staging_positions = {
            name: float(
                self.backend_arm.model.jnt_range[joint_id][0]
                + (np.random.uniform(min_percent, max_percent) / 100.0)
                * (
                    self.backend_arm.model.jnt_range[joint_id][1]
                    - self.backend_arm.model.jnt_range[joint_id][0]
                )
            )
            for name, joint_id in self.backend_arm.joint_indices.items()
        }

        print(f"Staging target positions: {staging_positions}")
        fieldnames = ["timestamp", "step", "status"]
        for name in staging_positions:
            for field in (
                "target",
                "present_position",
                "commanded_position",
                "present_velocity",
                "present_load",
                "present_voltage",
                "present_temperature",
            ):
                fieldnames.append(f"{name}_{field}")

        with open(log_path, "w", newline="") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=fieldnames)
            writer.writeheader()

            for step in range(max_steps):
                state = self.read_state()
                current_state = state["Present_Position"]
                errors = {
                    name: staging_positions[name] - current_state[name]
                    for name in staging_positions
                }

                row = {
                    "timestamp": state["python_recording_time"],
                    "step": step,
                    "status": "tracking",
                }
                for name in staging_positions:
                    row[f"{name}_target"] = staging_positions[name]
                    row[f"{name}_present_position"] = current_state[name]
                    row[f"{name}_present_velocity"] = state["Present_Velocity"][name]
                    row[f"{name}_present_load"] = state["Present_Load"][name]
                    row[f"{name}_present_voltage"] = state["Present_Voltage"][name]
                    row[f"{name}_present_temperature"] = state["Present_Temperature"][name]

                if max(abs(error) for error in errors.values()) <= tolerance_radians:
                    row["status"] = "complete"
                    writer.writerow(row)
                    print(f"Staging log written to: {log_path}")
                    return

                next_positions = {
                    name: current_state[name]
                    + np.clip(
                        error,
                        -speed_radians_per_second * pause_seconds,
                        speed_radians_per_second * pause_seconds,
                    )
                    for name, error in errors.items()
                }
                safe_positions = self.write_goal(next_positions)
                for name in staging_positions:
                    row[f"{name}_commanded_position"] = safe_positions[name]
                writer.writerow(row)
                log_file.flush()
                time.sleep(pause_seconds)

        raise RuntimeError("Real arm did not reach the staging pose.")

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
