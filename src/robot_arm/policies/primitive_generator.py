from typing import Any, List

import numpy as np
from omegaconf import DictConfig

from robot_arm.control_types import ActionPrimitive, EnvironmentState
from robot_arm.geometry.pose import Pose
from robot_arm.policies.action_primitives import PickAndPlacePrimitives, relative_move_primitive, random_waypoint_primitive
from robot_arm.robot_schema import BOX_BODY_NAMES


class ScriptedPrimitiveGeneratorPolicy:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.primitives: list[ActionPrimitive] = []
        self.next_primitive_index = 0

    @property
    def target_poses(self) -> list[np.ndarray]:
        return [primitive.target_pose.as_10d() for primitive in self.primitives]

    def select_task(self) -> None:
        self.task = select_task(self.cfg)

    def generate(self, model: Any, data: Any, start_pose: Pose) -> None:
        self.primitives = generate_action_primitives(model, data, self.cfg, start_pose, self.task)
        self.next_primitive_index = 0

    def has_next_primitive(self) -> bool:
        return self.next_primitive_index < len(self.primitives)

    def get_next_primitive(self, start_pose: Pose) -> tuple[int, ActionPrimitive]:
        primitive_index = self.next_primitive_index
        primitive = self.primitives[primitive_index]
        primitive.start_pose = start_pose
        self.next_primitive_index += 1
        return primitive_index, primitive

    def build_vla_input_state(self, primitive: ActionPrimitive, state: EnvironmentState) -> np.ndarray:
        current_pose = state.end_effector_pose
        gripper_duty = float(state.observation["gripper_duty"][0])
        current_pose_7d = current_pose.as_7d()
        if primitive.include_target_offset:
            target_offset_7d = current_pose.delta_to(primitive.target_pose)
            target_offset_flag = 1.0
        else:
            target_offset_7d = np.zeros(7, dtype=np.float32)
            target_offset_flag = 0.0

        return np.concatenate([current_pose_7d, target_offset_7d, [target_offset_flag], [gripper_duty]]).astype(np.float32)


def select_task(cfg: DictConfig) -> str:
    probabilities = cfg.waypoint.primitive_probabilities
    task_probabilities = np.array(
        [
            probabilities.pick_and_place,
            probabilities.relative_move,
            probabilities.random_waypoint,
        ],
        dtype=np.float64,
    )
    assert np.all(task_probabilities >= 0.0)
    assert np.isclose(task_probabilities.sum(), 1.0)

    return str(
        np.random.choice(
            ["pick_and_place", "relative_move", "random_waypoint"],
            p=task_probabilities,
        )
    )


def generate_action_primitives(
    model,
    data,
    cfg: DictConfig,
    start_pose: Pose,
    task: str,
) -> List[ActionPrimitive]:
    if task == "pick_and_place":
        return generate_pick_and_place(model, data, cfg, start_pose)
    elif task == "relative_move":
        return [relative_move_primitive(model, data, cfg, start_pose)]
    elif task == "random_waypoint":
        return [random_waypoint_primitive(model, data, cfg, start_pose)]
    raise ValueError(f"Unknown primitive task: {task}")


def generate_pick_and_place(model, data, cfg: DictConfig, start_pose: Pose) -> list[ActionPrimitive]:
    box_body_name = str(np.random.choice(BOX_BODY_NAMES))
    constructors = PickAndPlacePrimitives(model, data, cfg, box_body_name)
    sequence = (
        constructors.open_gripper,
        constructors.move_above_box,
        constructors.move_to_box,
        constructors.close_gripper,
        constructors.lift_object,
        constructors.move_above_tile,
        constructors.lower_object,
        constructors.release_object,
    )
    primitives = []
    pose = start_pose
    for construct in sequence:
        primitive = construct(pose)
        primitives.append(primitive)
        pose = primitive.target_pose
    return primitives
