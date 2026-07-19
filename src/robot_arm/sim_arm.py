from typing import Dict
import mujoco
import numpy as np

from robot_arm.arm import Arm


class SimArm(Arm):
    """
    Simulation adapter for the SO-101 using MuJoCo.
    Operates in radians (unlike RealArm which uses raw steps/bits).
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
    def pinch_point(self) -> np.ndarray:
        return (self.fixed_finger_tip + self.moving_finger_tip) / 2.0

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

    def get_pinch_point(self) -> np.ndarray:
        # Get position of TCP
        pos = self.pinch_point

        # Get euler angles from the wrist
        euler = self.gripper_euler

        # Aperture in radians from the servo itself
        qpos_idx = self.model.jnt_qposadr[self.joint_indices["gripper"]]
        gripper_radians = float(self.data.qpos[qpos_idx])

        return np.array(
            [pos[0], pos[1], pos[2], euler[0], euler[1], euler[2], gripper_radians],
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

    def read_image(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera="pixel_cam")
        return self.renderer.render()
