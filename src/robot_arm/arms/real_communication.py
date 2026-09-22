import json
import math
import time
from collections import deque
from copy import deepcopy
from threading import Event, Lock, Thread, current_thread
from typing import Dict

from lerobot.motors.feetech import OperatingMode
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from robot_arm.arms.communication import Communication
from robot_arm.arms.read_sensors import read_block, read_configuration
from robot_arm.arms.servo import CURRENT_AMPS_PER_TICK, FULL_SCALE_DUTY
from robot_arm.robot_schema import MOTOR_ORDER


class RealCommunication(Communication):
    def __init__(self, cfg, joint_limits):
        super().__init__(cfg)
        cfg = cfg.hardware
        self.follower = SO101Follower(SO101FollowerConfig(port=cfg.port, id=cfg.calibration_id))
        self.bus = self.follower.bus
        self.connected = False
        self.joint_limits = joint_limits
        self.max_res = 4096
        self.deg_to_rad = math.pi / 180.0
        self.velocity_scale = 2.0 * math.pi / 4096.0
        self.max_feedback_age_ns = int(cfg.max_feedback_age_seconds * 1_000_000_000)
        assert self.max_feedback_age_ns > 0
        assert cfg.read_timeout_seconds > 0
        self.read_timeout_seconds = cfg.read_timeout_seconds
        assert cfg.read_history_size > 0
        self.read_attempts = deque(maxlen=cfg.read_history_size)
        self.last_successful_read_ns = None
        self.latest_state = None
        self.pending_command = None
        self.pending_events = []
        self.error = None
        self.lock = Lock()
        self.ready = Event()
        self.stop_requested = Event()
        self.thread = Thread(target=self._run, name="arm-communication", daemon=True)
        self.started = False

    def setup(self):
        self.follower.connect(calibrate=True)
        self.connected = True
        if not self.bus.calibration:
            raise RuntimeError("Bus has no calibration registered. Cannot convert units.")
        self.configuration = read_configuration(self.bus)
        self._set_pwm_mode()
        port = self.bus.port_handler

        def set_packet_timeout(packet_length):
            port.setPacketTimeoutMillis(self.read_timeout_seconds * 1000.0)

        port.setPacketTimeout = set_packet_timeout

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

    def read_sensors(self):
        return self._convert_sensor_units(read_block(self.bus))

    def _convert_sensor_units(self, raw_state):
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
                model_range = self.joint_limits[name]
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

    def _set_pwm_mode(self) -> None:
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
        self.send_duty({motor: 0.0 for motor in MOTOR_ORDER})
        self.bus.enable_torque()

    def send_duty(self, duties: Dict[str, float]) -> None:
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
        assert not self.started, "Position commands cannot share the bus with the PWM worker"
        calibrated_positions = {}
        for name, rad in positions.items():
            if name == "gripper":
                model_range = self.joint_limits[name]
                calibrated_positions[name] = (rad - model_range[0]) / (model_range[1] - model_range[0]) * 100.0
            else:
                calibrated_positions[name] = rad / self.deg_to_rad

        self.bus.sync_write("Goal_Position", calibrated_positions, normalize=True)

    def shutdown(self):
        if not self.connected:
            return
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

        serial_port = self.bus.port_handler.ser
        for _ in range(3):
            serial_port.write(estop_packet)
            serial_port.flush()
            time.sleep(0.002)

        self.follower.disconnect()
        self.connected = False

    def start(self, log_path):
        assert not self.started
        self.log_path = log_path
        self.started = True
        self.thread.start()
        self.ready.wait()
        self._check_running()

    def _check_running(self):
        if self.error is not None:
            raise self.error
        if not self.started or self.stop_requested.is_set():
            raise RuntimeError("Arm communication is not running")

    def get_state(self):
        with self.lock:
            self._check_running()
            state = deepcopy(self.latest_state)
            used_ns = time.perf_counter_ns()
            self._check_deadline(used_ns, self.last_successful_read_ns)
            state["feedback_used_ns"] = used_ns
            state["feedback_age_seconds"] = (used_ns - state["read_completed_ns"]) / 1_000_000_000
            self.pending_events.append({"event": "feedback_used", "time_ns": used_ns, "sample_time_ns": state["sample_time_ns"]})
            return state

    def submit_duty(self, duties, sample_time_ns):
        with self.lock:
            self._check_running()
            command = {"duties": duties.copy(), "submitted_ns": time.perf_counter_ns(), "sample_time_ns": sample_time_ns}
            self.pending_command = command
            self.pending_events.append({"event": "command_submitted", **command})

    def close(self):
        self.stop_requested.set()
        if not self.started:
            self.shutdown()
        elif current_thread() is not self.thread:
            self.thread.join()

    def _check_deadline(self, now_ns, reference_ns):
        if now_ns - reference_ns >= self.max_feedback_age_ns:
            self.stop_requested.set()
            raise TimeoutError(f"No timely motor feedback within {self.max_feedback_age_ns / 1_000_000_000:g} seconds")

    def _flush_events(self, log):
        with self.lock:
            events = self.pending_events
            self.pending_events = []
        for event in events:
            log.write(json.dumps(event) + "\n")

    def _send_pending_duty(self, log, reference_ns):
        with self.lock:
            command = self.pending_command
            self.pending_command = None
        if command is None or self.stop_requested.is_set():
            return
        self._check_deadline(time.perf_counter_ns(), reference_ns)
        started_ns = time.perf_counter_ns()
        self.send_duty(command["duties"])
        log.write(json.dumps({"event": "command_sent", **command, "started_ns": started_ns, "completed_ns": time.perf_counter_ns()}) + "\n")

    def _read_once(self, log, reference_ns):
        self._check_deadline(time.perf_counter_ns(), reference_ns)
        started_ns = time.perf_counter_ns()
        try:
            state = self.read_sensors()
        except ConnectionError as error:
            attempt = {"started_ns": started_ns, "completed_ns": time.perf_counter_ns(), "success": False, "error": str(error)}
            with self.lock:
                self.read_attempts.append(attempt)
            log.write(json.dumps({"event": "read_attempt", **attempt}) + "\n")
            return reference_ns
        completed_ns = time.perf_counter_ns()
        attempt = {"started_ns": started_ns, "completed_ns": completed_ns, "success": True, "error": ""}
        with self.lock:
            self.read_attempts.append(attempt)
        log.write(json.dumps({"event": "read_attempt", **attempt, "state": state}) + "\n")
        self._check_deadline(completed_ns, reference_ns)
        self.check_safety(state)
        with self.lock:
            self.latest_state = state
            self.last_successful_read_ns = completed_ns
        self.ready.set()
        return completed_ns

    def _run(self):
        try:
            with open(self.log_path, "w", buffering=1) as log:
                try:
                    self.setup()
                    reference_ns = time.perf_counter_ns()
                    while not self.stop_requested.is_set():
                        self._check_deadline(time.perf_counter_ns(), reference_ns)
                        self._flush_events(log)
                        self._send_pending_duty(log, reference_ns)
                        if self.stop_requested.is_set():
                            break
                        reference_ns = self._read_once(log, reference_ns)
                finally:
                    self._flush_events(log)
        except BaseException as error:
            # Forward thread failures to the policy caller as well as threading.excepthook.
            self.error = error
            raise
        finally:
            self.stop_requested.set()
            try:
                self.shutdown()
            except BaseException as error:
                self.error = error
                raise
            finally:
                self.ready.set()
