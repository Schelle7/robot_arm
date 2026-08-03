from typing import Dict
import mujoco
import numpy as np

from robot_arm.backends.arm import Arm
from robot_arm.pose import Pose
from scipy.spatial.transform import Rotation


class SimBackend(Arm):
    """
    Simulation adapter for the SO-101 using MuJoCo.
    Operates in radians (unlike RealArm which uses raw steps/bits).
    Includes and manages simulation scene elements (like the target box).
    Unit conversion is done higher up the stack.
    """

    def __init__(self, model_path: str, height: int, width: int):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.camera_scene_option = mujoco.MjvOption()
        self.camera_scene_option.geomgroup[5] = 0

        # Build explicit mappings for actuator and joint indices
        self.actuator_indices = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(self.model.nu)
        }

        self.joint_indices = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.actuator_indices
        }

    @property
    def fixed_finger_tip(self) -> np.ndarray:
        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "fixed_finger_tip"
        )
        return self.data.site_xpos[site_id].copy()

    @property
    def moving_finger_tip(self) -> np.ndarray:
        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "moving_finger_tip"
        )
        return self.data.site_xpos[site_id].copy()

    @property
    def tcp(self) -> np.ndarray:
        return (self.fixed_finger_tip + self.moving_finger_tip) / 2.0

    def get_tcp(self):
        # this is obviously kinda stupid but Ill leave it for now
        return self.tcp

    @property
    def aperture(self) -> float:
        return float(np.linalg.norm(self.moving_finger_tip - self.fixed_finger_tip))

    @property
    def mujoco_hand_pose(self) -> Pose:
        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
        )
        rot_mat = self.data.site_xmat[site_id].copy().reshape(3, 3)

        # Aperture in radians from the servo itself
        qpos_idx = self.model.jnt_qposadr[self.joint_indices["gripper"]]
        gripper_radians = float(self.data.qpos[qpos_idx])

        return Pose.from_matrix(self.tcp, rot_mat, gripper_radians)

    def get_tcp_pose(self) -> Pose:
        closing, secondary = self.get_tcp_axes()
        return Pose.from_tcp_axes(
            self.tcp,
            closing,
            secondary,
            self.mujoco_hand_pose.gripper,
        )

    def get_privileged_box_pose(self) -> Pose:
        # The box is defined as a body named "target_box" in scene.xml
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_box")
        if body_id == -1:
            raise KeyError("Body 'target_box' not found in MuJoCo model.")

        pos = self.data.xpos[body_id].copy()
        rot_mat = self.data.xmat[body_id].reshape(3, 3)

        return Pose.from_matrix(
            pos, rot_mat, 1.0
        )  # pose with gripper info is a bit weird but ok for now

    def read_state(self) -> Dict[str, Dict[str, float]]:
        # Map MuJoCo qpos, qvel, ctrl (as a proxy for load) to our expected dictionary format
        state = {
            "Present_Position": {},
            "Present_Velocity": {},
            "Present_Load": {},  # Returning actuator control effort as load
            "Present_Voltage": {},  # Dummy data
            "Present_Temperature": {},  # Dummy data
        }

        for name, actuator_idx in self.actuator_indices.items():
            qpos_idx = self.model.jnt_qposadr[self.joint_indices[name]]
            qvel_idx = self.model.jnt_dofadr[self.joint_indices[name]]

            state["Present_Position"][name] = float(self.data.qpos[qpos_idx])
            state["Present_Velocity"][name] = float(self.data.qvel[qvel_idx])
            # Simulated load normalization: MuJoCo forces are in N or N-m.
            # We divide by a nominal stall torque (e.g., 2.0 N-m for STS3215) to get a pseudo-percentage.
            # Clip between -1.0 and 1.0 to match hardware behavior.

            # todo use
            # raw_force = float(self.data.actuator_force[actuator_idx])
            state["Present_Load"][name] = 0  # max(-1.0, min(1.0, raw_force / 2.0))
            # todo decide some sensible strategy how to do this in simulation
            # read a bit about it.

            state["Present_Voltage"][name] = 12.0
            state["Present_Temperature"][name] = 40.0

        state["sim_state"] = {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
        }

        return state

    def write_goal(self, positions: Dict[str, float]) -> None:
        for name, target_pos in positions.items():
            self.data.ctrl[self.actuator_indices[name]] = target_pos

        # Advance simulation one step
        mujoco.mj_step(self.model, self.data)

    def disconnect(self):
        """Simulation doesn't need to physically disconnect power."""
        pass

    def randomize_box(self):
        # Randomize box placement
        # Reach is ~60cm. Goal is 25cm-45cm outward (X-axis) and -10cm to 10cm sideways (Y-axis)
        box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_box")
        if box_id == -1:
            raise Exception("Box is missing")

        dist = np.random.uniform(0.35, 0.15)
        y_shift = np.random.uniform(-0.10, 0.10)

        # Directly updating the freejoint associated with the box
        jnt_idx = self.model.body_jntadr[box_id]
        if jnt_idx != -1 and self.model.jnt_type[jnt_idx] == mujoco.mjtJoint.mjJNT_FREE:
            qpos_adr = self.model.jnt_qposadr[jnt_idx]

            self.data.qpos[qpos_adr] = dist
            self.data.qpos[qpos_adr + 1] = y_shift
            # Z explicitly left alone

    def randomize_arm_pos(self):
        for name, joint_id in self.joint_indices.items():
            jnt_range = self.model.jnt_range[joint_id]
            qpos_idx = self.model.jnt_qposadr[joint_id]

            jmin, jmax = jnt_range[0], jnt_range[1]
            span = jmax - jmin
            safe_min = jmin + 0.05 * span
            safe_max = jmax - 0.05 * span
            new_pos = np.random.uniform(safe_min, safe_max)
            self.data.qpos[qpos_idx] = new_pos

            # Sync control target so the arm doesn't instantly snap back
            actuator_id = self.actuator_indices[name]
            self.data.ctrl[actuator_id] = new_pos

    def reset_sim(self):
        mujoco.mj_resetData(self.model, self.data)

        self.randomize_box()
        self.randomize_arm_pos()

        mujoco.mj_forward(self.model, self.data)

    def read_camera(self) -> np.ndarray:
        self.renderer.update_scene(
            self.data, camera="pixel_cam", scene_option=self.camera_scene_option
        )
        return self.renderer.render()

    def get_tcp_axes(self) -> tuple[np.ndarray, np.ndarray]:
        fixed = self.fixed_finger_tip
        moving = self.moving_finger_tip
        closing = fixed - moving
        closing = closing / np.linalg.norm(closing)

        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
        )
        frame = self.data.site_xmat[site_id].reshape(3, 3)
        secondary = frame[:, 1]
        secondary = secondary - np.dot(secondary, closing) * closing
        secondary = secondary / np.linalg.norm(secondary)

        return closing, secondary

    def draw_tcp(self):
        """Draw the live finger axes without affecting physics or camera images."""
        fixed = self.fixed_finger_tip
        moving = self.moving_finger_tip
        closing, secondary = self.get_tcp_axes()
        closing_length = np.linalg.norm(fixed - moving)

        self._draw_debug_arrow(
            "debug_actual_closing", (fixed + moving) / 2, closing, closing_length
        )
        self._draw_debug_arrow(
            "debug_actual_secondary", (fixed + moving) / 2, secondary, 0.09
        )

    def _draw_debug_arrow(
        self, body_name: str, origin: np.ndarray, direction: np.ndarray, length: float
    ):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        mocap_id = self.model.body_mocapid[body_id]
        self.data.mocap_pos[mocap_id] = origin + direction * length / 2
        rotation, _ = Rotation.align_vectors([direction], [[0, 0, 1]])
        rotation_matrix = rotation.as_matrix()
        self.data.mocap_quat[mocap_id] = Pose(
            origin,
            rotation,
            0.0,
            rotation_matrix[:, 0],
            rotation_matrix[:, 1],
        ).as_mujoco_quat()

    def draw_waypoints(self, waypoints: np.ndarray):
        """
        Takes up to 4 waypoints of 10D arrays
        and updates the 'ghost_wp_X' mocap models to visualize them.
        Lines ('ghost_line_X') are drawn between them.
        """
        num_wp = min(len(waypoints), 4)

        for i in range(num_wp):
            wp = waypoints[i]
            pose = Pose.from_10d(wp)

            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"ghost_wp_{i}"
            )
            if body_id == -1:
                raise KeyError(
                    f"Body 'ghost_wp_{i}' not found in MuJoCo model. Ensure ghost_waypoints.xml is included."
                )
            mocap_id = self.model.body_mocapid[body_id]
            self.data.mocap_pos[mocap_id] = pose.position
            self.data.mocap_quat[mocap_id] = pose.as_mujoco_quat()

        # Draw cylinders between waypoints
        for i in range(num_wp - 1):
            p1_pose = Pose.from_10d(waypoints[i])
            p2_pose = Pose.from_10d(waypoints[i + 1])
            p1 = p1_pose.position
            p2 = p2_pose.position

            line_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"ghost_line_{i}"
            )
            line_geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"ghost_line_{i}_geom"
            )

            if line_body_id == -1 or line_geom_id == -1:
                raise KeyError(
                    f"Body or geom for 'ghost_line_{i}' not found in MuJoCo model."
                )

            mocap_id = self.model.body_mocapid[line_body_id]

            # Position is midpoint
            midpoint = (p1 + p2) / 2.0
            self.data.mocap_pos[mocap_id] = midpoint

            # Rotation to align z-axis of cylinder with the direction vector
            vec = p2 - p1
            dist = np.linalg.norm(vec)

            if dist > 1e-5:
                vec = vec / dist
                # Default cylinder acts along Z (0, 0, 1)
                # We compute quaternion that maps (0,0,1) to vec
                z_axis = np.array([0, 0, 1])
                rot, _ = Rotation.align_vectors([vec], [z_axis])

                rotation_matrix = rot.as_matrix()
                quat = Pose(
                    midpoint,
                    rot,
                    1.0,
                    rotation_matrix[:, 0],
                    rotation_matrix[:, 1],
                ).as_mujoco_quat()
                self.data.mocap_quat[mocap_id] = quat

            # Update geom length (size is [radius, half-length])
            self.model.geom_size[line_geom_id][1] = dist / 2.0

        # Send unused waypoints / lines underground
        for i in range(num_wp, 4):
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"ghost_wp_{i}"
            )
            if body_id == -1:
                raise KeyError(f"Body 'ghost_wp_{i}' not found in MuJoCo model.")
            mocap_id = self.model.body_mocapid[body_id]
            self.data.mocap_pos[mocap_id] = [0, 0, -10]

        for i in range(max(0, num_wp - 1), 3):
            line_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"ghost_line_{i}"
            )
            if line_body_id == -1:
                raise KeyError(f"Body 'ghost_line_{i}' not found in MuJoCo model.")
            mocap_id = self.model.body_mocapid[line_body_id]
            self.data.mocap_pos[mocap_id] = [0, 0, -10]
