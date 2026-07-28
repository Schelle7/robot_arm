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
        self.max_res = 4096  # STS3215 specific (12-bit encoder)
        self.deg_to_rad = math.pi / 180.0
        self.velocity_scale = (
            2.0 * math.pi / 4096.0
        )  # units are ticks/sec, so 1 tick/sec is (2*pi/4096) rad/s

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

        # Convert load from raw ticks (-1000 to 1000) to normalized float (-1.0 to 1.0)
        for name, load_tick in raw_state["Present_Load"].items():
            raw_state["Present_Load"][name] = load_tick / 1000.0

        for name, vel_tick in raw_state["Present_Velocity"].items():
            raw_state["Present_Velocity"][name] = vel_tick * self.velocity_scale

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
        # Immediate hardware emergency stop broadcast packet for Feetech servos
        # Packet breakdown:
        # \xFF\xFF : Standard 2-byte header
        # \xFE     : Broadcast ID (targets all servos simultaneously)
        # \x04     : Length of remaining bytes
        # \x03     : Instruction (WRITE Data)
        # \x29     : Register Address 41 (Torque Enable)
        # \x00     : Data value 0 (Disable)
        # \xD1     : Checksum (~(0xFE + 0x04 + 0x03 + 0x29 + 0x00) & 0xFF)
        estop_packet = b"\xff\xff\xfe\x04\x03\x29\x00\xd1"

        try:
            # 1. Blindly spam the fast broadcast command first to instantly drop torque
            serial_port = self.bus.port
            for _ in range(3):
                serial_port.write(estop_packet)
                serial_port.flush()
                time.sleep(0.002)

            # 2. Follow up with the official API cleanup ensuring internal state matches
            # The follower object holds the disconnect method in our specific initialization chain
            self.bus.robot.disconnect()
        except Exception as e:
            # If serial completely died, re-raise so outer layers know the port crashed
            raise RuntimeError(f"Failed to execute emergency stop on hardware: {e}")
