from typing import Any

import numpy as np
from omegaconf import DictConfig

from robot_arm.pose import Pose
from robot_arm.primitives import ActionPrimitive, generate_action_primitives


class ScriptedPrimitiveGeneratorPolicy:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.primitives: list[ActionPrimitive] = []
        self.next_primitive_index = 0

    @property
    def target_poses(self) -> list[np.ndarray]:
        return [primitive.target_pose.as_10d() for primitive in self.primitives]

    def generate(self, model: Any, data: Any, start_pose: Pose) -> None:
        self.primitives = generate_action_primitives(model, data, self.cfg, start_pose)
        self.next_primitive_index = 0

    def has_next_primitive(self) -> bool:
        return self.next_primitive_index < len(self.primitives)

    def get_next_primitive(self, start_pose: Pose) -> tuple[int, ActionPrimitive]:
        primitive_index = self.next_primitive_index
        primitive = self.primitives[primitive_index]
        primitive.start_pose = start_pose
        self.next_primitive_index += 1
        return primitive_index, primitive

    def build_vla_input_state(self, primitive: ActionPrimitive) -> np.ndarray:
        initial_pose_7d = primitive.start_pose.as_7d()
        if primitive.has_explicit_goal:
            target_offset_7d = primitive.start_pose.delta_to(primitive.target_pose)
            goal_flag = 1.0
        else:
            target_offset_7d = np.zeros(7, dtype=np.float32)
            goal_flag = 0.0

        return np.concatenate([initial_pose_7d, target_offset_7d, [goal_flag]]).astype(np.float32)
