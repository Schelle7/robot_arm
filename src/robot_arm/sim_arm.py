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

    def read_state(self) -> Dict[str, Dict[str, float]]:
        # Map MuJoCo qpos, qvel, ctrl (as a proxy for load) to our expected dictionary format
        state = {
            "Present_Position": {},
            "Present_Velocity": {},
            "Present_Load": {},      # Returning actuator control effort as load
            "Present_Voltage": {},   # Dummy data
            "Present_Temperature": {} # Dummy data
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
