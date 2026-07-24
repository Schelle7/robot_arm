from typing import Dict
import mujoco
import numpy as np

from robot_arm.backends.arm import Arm


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
    def gripper_euler(self) -> np.ndarray:
        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
        )
        rot_mat = self.data.site_xmat[site_id].copy()

        # We can extract xyz directly with scipy if we dont want to write it out
        from scipy.spatial.transform import Rotation

        euler = Rotation.from_matrix(rot_mat.reshape(3, 3)).as_euler("xyz")

        return euler.astype(np.float32)

    def get_end_effector_pose_7d(self) -> np.ndarray:
        # Get position of TCP
        pos = self.tcp

        # Get euler angles from the wrist
        euler = self.gripper_euler

        # Aperture in radians from the servo itself
        qpos_idx = self.model.jnt_qposadr[self.joint_indices["gripper"]]
        gripper_radians = float(self.data.qpos[qpos_idx])

        return np.array(
            [pos[0], pos[1], pos[2], euler[0], euler[1], euler[2], gripper_radians],
            dtype=np.float32,
        )

    def get_privileged_box_pose_6d(self) -> np.ndarray:
        # The box is defined as a body named "target_box" in scene.xml
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_box")
        if body_id == -1:
            raise KeyError("Body 'target_box' not found in MuJoCo model.")

        pos = self.data.xpos[body_id]

        rot_mat = self.data.xmat[body_id].reshape(3, 3)
        from scipy.spatial.transform import Rotation

        euler = Rotation.from_matrix(rot_mat).as_euler("xyz")

        return np.array(
            [pos[0], pos[1], pos[2], euler[0], euler[1], euler[2]],
            dtype=np.float32,
        )

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
            state["Present_Load"][name] = float(self.data.ctrl[actuator_idx])
            state["Present_Voltage"][name] = 12.0
            state["Present_Temperature"][name] = 40.0

        return state

    def write_goal(self, positions: Dict[str, float]) -> None:
        for name, target_pos in positions.items():
            self.data.ctrl[self.actuator_indices[name]] = target_pos

        # Advance simulation one step
        mujoco.mj_step(self.model, self.data)

    def reset_sim(self):
        mujoco.mj_resetData(self.model, self.data)

        # Randomize box placement
        # Reach is ~60cm. Goal is 25cm-45cm outward (X-axis) and -10cm to 10cm sideways (Y-axis)
        box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_box")
        if box_id != -1:
            dist = np.random.uniform(0.25, 0.45)
            y_shift = np.random.uniform(-0.10, 0.10)

            # Directly updating the freejoint associated with the box
            jnt_idx = self.model.body_jntadr[box_id]
            if (
                jnt_idx != -1
                and self.model.jnt_type[jnt_idx] == mujoco.mjtJoint.mjJNT_FREE
            ):
                qpos_adr = self.model.jnt_qposadr[jnt_idx]

                self.data.qpos[qpos_adr] = dist
                self.data.qpos[qpos_adr + 1] = y_shift
                # Z explicitly left alone

        mujoco.mj_forward(self.model, self.data)

    def read_camera(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera="pixel_cam")
        return self.renderer.render()
