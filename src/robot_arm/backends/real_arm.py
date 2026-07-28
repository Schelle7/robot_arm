import math
import time
import numpy as np
from typing import Dict

from robot_arm.backends.arm import Arm
from robot_arm.backends.read_sensors import read_block


class RealArm(Arm):
    """
    Hardware adapter for the SO-101 using the LeRobot bus.
    Translates hardware integer ticks to standard SI radians for position.
    """

    def __init__(self, bus):
        if not bus.calibration:
            raise RuntimeError(
                "Bus has no calibration registered. Cannot convert units."
            )

        self.bus = bus
        self.max_res = 4095  # STS3215 specific
        self.deg_to_rad = math.pi / 180.0

        # Precompute the tick-to-radian conversion offsets per motor
        self.tick_midpoints = {}
        self.drive_modes = {}

        for name, calib in self.bus.calibration.items():
            self.tick_midpoints[name] = (calib.range_min + calib.range_max) / 2.0
            self.drive_modes[name] = calib.drive_mode

    def _tick_to_rad(self, name: str, tick: int) -> float:
        """Applies exact LeRobot formula: ticks -> degrees -> radians."""
        mid = self.tick_midpoints[name]
        degrees = (tick - mid) * 360.0 / self.max_res

        if self.drive_modes[name] and self.bus.apply_drive_mode:
            degrees = -degrees

        return degrees * self.deg_to_rad

    def _rad_to_tick(self, name: str, rad: float) -> int:
        """Inverses the formula: radians -> degrees -> ticks."""
        degrees = rad / self.deg_to_rad

        if self.drive_modes[name] and self.bus.apply_drive_mode:
            degrees = -degrees

        mid = self.tick_midpoints[name]
        tick = int((degrees * self.max_res / 360.0) + mid)

        # Clamp to calibrated bounds
        calib = self.bus.calibration[name]
        return max(calib.range_min, min(calib.range_max, tick))

    def read_state(self) -> Dict[str, Dict[str, float]]:
        raw_state = read_block(self.bus)
        raw_state["python_recording_time"] = time.time()

        # Convert positions from raw ticks to radians
        for name, tick in raw_state["Present_Position"].items():
            raw_state["Present_Position"][name] = self._tick_to_rad(name, tick)

        # Note: Velocity and Load remain in raw units (-1000 to 1000) for now.
        # If policy needs them in rad/s, we need their specific max scale factors.
        return raw_state

    def get_tcp(self) -> np.ndarray:
        raise NotImplementedError("Real arm does not have access to pinch point")

    def write_goal(self, positions: Dict[str, float]) -> None:
        raw_positions = {}
        for name, rad in positions.items():
            raw_positions[name] = self._rad_to_tick(name, rad)

        # sync_write handles the broadcast automatically if the bus is capable
        self.bus.sync_write("Goal_Position", raw_positions)

    def disconnect(self):
        # Implementation assumes the real arm bus has a disconnect or torque-off feature
        self.bus.disconnect()
