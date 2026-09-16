import logging
import math
import time
import numpy as np
from typing import Dict
import mujoco
from lerobot.motors.feetech import OperatingMode

from robot_arm.arms.arm import Arm
from robot_arm.arms.read_sensors import read_block, read_configuration
from robot_arm.arms.servo import CURRENT_AMPS_PER_TICK, FULL_SCALE_DUTY
from robot_arm.geometry.gripper_geometry import get_tcp_geometry
from robot_arm.geometry.pose import Pose
from robot_arm.robot_schema import MOTOR_ORDER

log = logging.getLogger(__name__)


class RealArm(Arm):
    """
    Hardware adapter for the SO-101 using the LeRobot bus.
    Translates hardware integer ticks to standard SI radians for position.
    Uses a headless MuJoCo model to compute Forward Kinematics (FK) for the TCP pose.
    """

    def __init__(self, bus, model_path: str, control_step_seconds: float):
        if not bus.calibration:
            raise RuntimeError("Bus has no calibration registered. Cannot convert units.")

        self.bus = bus
        self.control_step_seconds = control_step_seconds
        self.last_control_step_end = time.perf_counter()
        self.control_step_overruns = 0
        self.max_res = 4096  # STS3215 specific (12-bit encoder)
        self.deg_to_rad = math.pi / 180.0
        self.velocity_scale = 2.0 * math.pi / 4096.0  # units are ticks/sec, so 1 tick/sec is (2*pi/4096) rad/s
        # Position mode applies Homing_Offset in firmware; PWM feedback does not.
        # Cache modes rather than adding six serial transactions to every control step.
        self.operating_modes = {name: self.bus.read("Operating_Mode", name, normalize=False) for name in MOTOR_ORDER}

        # Initialize Headless MuJoCo for Forward Kinematics (FK)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.joint_indices = {
            # Map motor names directly to their mujoco joint IDs to update qpos efficiently
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in MOTOR_ORDER
        }
        self.qpos_indices = self.model.jnt_qposadr[np.array([self.joint_indices[name] for name in MOTOR_ORDER])]

        # Read once: these decide what a commanded position delta actually does, and lerobot rewrites
        # several of them on every connect, so a run is not interpretable without them.
        self.configuration = read_configuration(self.bus)

    def _tick_to_rad(self, name: str, tick: int) -> float:
        """
        'tick' is already zero-centered and direction-corrected by LeRobot.
        Strict scalar conversion to radians.
        """
        return tick * (2.0 * math.pi / self.max_res)

    def _rad_to_tick(self, name: str, rad: float) -> int:
        """
        Scale radians back to zero-centered ticks.
        LeRobot applies homing offsets, drive modes, and limit clipping internally.
        """
        return int(rad * self.max_res / (2.0 * math.pi))

    def configure_read_timeout(self, margin_ms: float) -> None:
        assert margin_ms > 0.0, "Read timeout margin must be positive"
        port = self.bus.port_handler

        def set_packet_timeout(packet_length):
            port.packet_start_time = port.getCurrentTime()
            port.packet_timeout = port.tx_time_per_byte * (packet_length + 3.0) + margin_ms

        port.setPacketTimeout = set_packet_timeout

    def _read_feedback(self):
        try:
            return read_block(self.bus)
        except ConnectionError as error:
            log.warning("\033[38;5;208mSENSOR READ FAILED: %s. RETRYING ONCE.\033[0m", str(error).upper())
        try:
            return read_block(self.bus)
        except ConnectionError:
            self.disconnect()
            raise

    def read_state(self) -> Dict[str, Dict[str, float]]:
        raw_state = self._read_feedback()
        raw_state["python_recording_time"] = time.time()

        position_ticks = {}
        for name, tick in raw_state["Present_Position"].items():
            if self.operating_modes[name] == OperatingMode.PWM.value:
                # Measured at the same stationary pose in both modes: PWM returns
                # the unhomed encoder count. Restore the position-mode frame BEFORE
                # LeRobot centers/scales it (and before it clips the gripper).
                tick = (tick - self.bus.calibration[name].homing_offset) % self.max_res
            position_ticks[self.bus.motors[name].id] = tick
        calibrated_positions = self.bus._normalize(position_ticks)
        for name, tick in raw_state["Present_Position"].items():
            calibrated_value = calibrated_positions[self.bus.motors[name].id]
            if name == "gripper":
                model_range = self.model.jnt_range[self.joint_indices[name]]
                raw_state["Present_Position"][name] = float(model_range[0] + (calibrated_value / 100.0) * (model_range[1] - model_range[0]))
            else:
                raw_state["Present_Position"][name] = calibrated_value * self.deg_to_rad

        # Feetech's PWM sign is opposite the model joint-angle direction measured on all six motors.
        for name, load_tick in raw_state["Present_Load"].items():
            raw_state["Present_Load"][name] = -load_tick / 1000.0

        for name, vel_tick in raw_state["Present_Velocity"].items():
            raw_state["Present_Velocity"][name] = vel_tick * self.velocity_scale

        for name, current_tick in raw_state["Present_Current"].items():
            raw_state["Present_Current"][name] = current_tick * CURRENT_AMPS_PER_TICK

        for name, voltage_tick in raw_state["Present_Voltage"].items():
            raw_state["Present_Voltage"][name] = voltage_tick * 0.1

        return raw_state

    def _forward_kinematics_pose(self, present_positions: dict) -> Pose:
        self.data.qpos[self.qpos_indices] = np.array([present_positions[name] for name in MOTOR_ORDER])
        mujoco.mj_kinematics(self.model, self.data)
        pose, _, _ = get_tcp_geometry(self.model, self.data)
        return pose

    def get_end_effector_pose_7d_forward_kinematics(self, present_positions: dict) -> np.ndarray:
        pose = self._forward_kinematics_pose(present_positions)
        return np.concatenate([pose.position, pose.as_euler("XYZ", False), [pose.gripper]]).astype(np.float32)

    def get_tcp_pose(self, state: Dict[str, Dict[str, float]]) -> Pose:
        return self._forward_kinematics_pose(state["Present_Position"])

    def get_tcp(self) -> np.ndarray:
        raise NotImplementedError("Real arm does not have access to pinch point")

    def read_cameras(self) -> dict[str, np.ndarray]:
        raise NotImplementedError("Real camera capture is not implemented.")

    def set_pwm_mode(self) -> None:
        """
        Operating_Mode sits in EEPROM, which the servo only accepts while torque is disabled, so the
        write has to be bracketed. The read back is not paranoia: a rejected EEPROM write is silent,
        and the servo would stay in position mode while duties are interpreted as goal positions.
        """
        self.bus.disable_torque()
        for motor in MOTOR_ORDER:
            self.bus.write("Operating_Mode", motor, OperatingMode.PWM.value)

        modes = {motor: self.bus.read("Operating_Mode", motor) for motor in MOTOR_ORDER}
        self.operating_modes = modes
        rejected = {motor: mode for motor, mode in modes.items() if mode != OperatingMode.PWM.value}
        if rejected:
            raise RuntimeError(f"Servos did not accept PWM mode, Operating_Mode reads back as {rejected}.")

        # Goal_Time still holds whatever position mode left there, which becomes a duty the moment
        # torque comes back.
        self.write_duty({motor: 0.0 for motor in MOTOR_ORDER})
        self.bus.enable_torque()

    def write_duty(self, duties: Dict[str, float]) -> None:
        """
        In PWM mode Goal_Time carries the duty instead of a travel time, sign-magnitude encoded with
        bit 10, the same direction bit the servo uses for Present_Load. LeRobot's encoding table does
        not list Goal_Time, so the bus writes the value through untouched. The hardware sign is
        inverted here to make positive interface duty increase the model joint angle.

        The interface passes a fraction of full output, which is 0 to 1000 in the servo's units.
        """
        encoded = {}
        for name, duty in duties.items():
            duty = float(duty)
            if abs(duty) > 1.0:
                raise ValueError(f"Duty {duty} for {name} is not a fraction of full output.")
            magnitude = round(abs(duty) * FULL_SCALE_DUTY)
            encoded[name] = magnitude + (1 << 10) if duty > 0.0 else magnitude

        self.bus.sync_write("Goal_Time", encoded, normalize=False)

    def write_goal(self, positions: Dict[str, float]) -> None:
        calibrated_positions = {}
        for name, rad in positions.items():
            if name == "gripper":
                model_range = self.model.jnt_range[self.joint_indices[name]]
                calibrated_positions[name] = (rad - model_range[0]) / (model_range[1] - model_range[0]) * 100.0
            else:
                calibrated_positions[name] = rad / self.deg_to_rad

        self.bus.sync_write("Goal_Position", calibrated_positions, normalize=True)

    def advance_control_step(self) -> None:
        # The servos run their own loop, so a control period is wall-clock time rather than steps.
        # Paced from the end of the previous period so the caller's own work counts against the
        # budget instead of being added on top of it.
        remaining = (self.last_control_step_end + self.control_step_seconds) - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        else:
            self.control_step_overruns += 1
            if self.control_step_overruns % 100 + 1 == 1:
                log.warning(
                    "Control step overran its %.1f ms budget by %.1f ms. The loop is running slower " "than the configured rate; further overruns are counted, not logged.",
                    self.control_step_seconds * 1e3,
                    -remaining * 1e3,
                )
        self.last_control_step_end = time.perf_counter()

    def disconnect(self):
        # Immediate hardware emergency stop broadcast packet for Feetech servos
        # Packet breakdown:
        # \xFF\xFF : Standard 2-byte header
        # \xFE     : Broadcast ID (targets all servos simultaneously)
        # \x04     : Length of remaining bytes
        # \x03     : Instruction (WRITE Data)
        # \x28     : Register Address 40 (Torque Enable). Address 41 is Acceleration, where 0 means
        #            no ramp limit, so aiming one register high both leaves torque on and makes
        #            every later move maximally abrupt.
        # \x00     : Data value 0 (Disable)
        # \xD2     : Checksum (~(0xFE + 0x04 + 0x03 + 0x28 + 0x00) & 0xFF)
        estop_packet = b"\xff\xff\xfe\x04\x03\x28\x00\xd2"

        try:
            # 1. Blindly spam the fast broadcast command first to instantly drop torque
            serial_port = self.bus.port_handler.ser
            for _ in range(3):
                serial_port.write(estop_packet)
                serial_port.flush()
                time.sleep(0.002)

            # 2. Follow up with the official API cleanup ensuring internal state matches
            # The follower object holds the disconnect method in our specific initialization chain
            self.follower_keepalive.disconnect()
        except Exception as e:
            # If serial completely died, re-raise so outer layers know the port crashed
            raise RuntimeError(f"Failed to execute emergency stop on hardware: {e}")
