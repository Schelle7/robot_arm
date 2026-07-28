import math
import time
import numpy as np
from typing import Dict
import mujoco
from scipy.spatial.transform import Rotation

from robot_arm.backends.arm import Arm
from robot_arm.backends.read_sensors import read_block


class RealArm(Arm):
    """
    Hardware adapter for the SO-101 using the LeRobot bus.
    Translates hardware integer ticks to standard SI radians for position.
    Uses a headless MuJoCo model to compute Forward Kinematics (FK) for pseudo-privileged poses.
    """

    def __init__(self, bus, model_path: str):
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

        # Initialize Headless MuJoCo for Forward Kinematics (FK)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.joint_indices = {
            # Map motor names directly to their mujoco joint IDs to update qpos efficiently
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        }

        # Cache MuJoCo IDs during initialization
        self.fixed_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "fixed_finger_tip")
        self.moving_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "moving_finger_tip")
        self.gripper_frame_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

        # Validate IDs
        if -1 in (self.fixed_id, self.moving_id, self.gripper_frame_id):
            raise RuntimeError("One or more MuJoCo site IDs could not be found in the XML.")

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

    def get_end_effector_pose_7d_forward_kinematics(self, present_positions: dict) -> np.ndarray:
        # Bind raw Joint radians back into MuJoCo FK structural limits
        for name, rad in present_positions.items():
            if name in self.joint_indices:
                qpos_idx = self.model.jnt_qposadr[self.joint_indices[name]]
                self.data.qpos[qpos_idx] = rad
                
        # Simulate geometric kinematics
        mujoco.mj_kinematics(self.model, self.data)

        # Re-derive exact geometric variables
        fixed_pos = self.data.site_xpos[self.fixed_id]
        moving_pos = self.data.site_xpos[self.moving_id]
        
        # Pseudo TCP (middle point)
        tcp_pos = (fixed_pos + moving_pos) / 2.0
        
        # Pseudo Rotation Matrix
        rot_mat = self.data.site_xmat[self.gripper_frame_id].reshape(3, 3)
        euler = Rotation.from_matrix(rot_mat).as_euler("xyz")
        
        # Raw Gripper Aperture Rads
        gripper_radians = present_positions["gripper"]
        
        return np.array(
            [tcp_pos[0], tcp_pos[1], tcp_pos[2], euler[0], euler[1], euler[2], gripper_radians],
            dtype=np.float32,
        )

    def get_tcp(self) -> np.ndarray:
        raise NotImplementedError("Real arm does not have access to pinch point")

    def read_camera(self) -> np.ndarray:
        print("\033[93mWARNING: READ_CAMERA NOT IMPLEMENTED FOR REAL ARM YET! RETURN DUMMY IMAGE\033[0m")
        return np.zeros((480, 640, 3), dtype=np.uint8)

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
