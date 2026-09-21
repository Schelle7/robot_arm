from typing import Dict

import mujoco
import numpy as np

from robot_arm.geometry.waypoints import shoulder_pan_position
from robot_arm.robot_schema import BOX_BODY_NAMES, MOTOR_ORDER, OBJECT_COLORS, TILE_BODY_NAME


class SceneSetup:
    def __init__(self, model, data, cfg):
        self.model = model
        self.data = data
        self.servo = cfg.servo
        self.joint_indices = {name: model.joint(name).id for name in MOTOR_ORDER}
        self.actuator_order = [model.actuator(index).name for index in range(model.nu)]
        self.actuator_dof_indices = np.array([model.jnt_dofadr[self.joint_indices[name]] for name in self.actuator_order])
        self.added_weight_body_id = self.model.body("added_weight").id
        self.box_mass_kg = float(self.model.body(BOX_BODY_NAMES[0]).mass[0])
        added_weight_radius = float(self.model.geom("added_weight_geom").size[0])
        self.added_weight_inertia_per_kg = 2.0 / 5.0 * added_weight_radius**2
        if cfg.runtime.disable_box_collisions:
            for body_name in BOX_BODY_NAMES:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                assert body_id != -1, f"Body {body_name!r} not found in MuJoCo model."
                geom_start = self.model.body_geomadr[body_id]
                geom_count = self.model.body_geomnum[body_id]
                self.model.geom_contype[geom_start : geom_start + geom_count] = 0
                self.model.geom_conaffinity[geom_start : geom_start + geom_count] = 0
        self.initial_joint_mode = cfg.control.initial_joints.mode
        self.initial_joint_range_percent = cfg.control.initial_joints.range_percent
        self.initial_joint_positions = cfg.control.initial_joints.positions_radians
        self.object_placement = cfg.scene.object_placement

    def _sample_object_angles(self, count: int) -> np.ndarray:
        """
        Draws azimuths that are already far enough apart to satisfy the separation, so placement
        never needs rejection sampling. Two objects at the inner radius need the widest angle for a
        given separation, so sizing the gap there covers every radius.
        """
        min_distance = min(self.object_placement.shoulder_distance_meters)
        min_gap = 2.0 * np.arcsin(self.object_placement.min_separation_meters / (2.0 * min_distance))
        low, high = np.deg2rad(self.object_placement.base_rotation_degrees)
        span = high - low - (count - 1) * min_gap
        assert span >= 0.0, f"base_rotation_degrees is too narrow to separate {count} objects by min_separation_meters."

        angles = np.sort(np.random.uniform(low, low + span, size=count)) + np.arange(count) * min_gap
        np.random.shuffle(angles)
        return angles

    def _sample_object_positions(self, count: int) -> np.ndarray:
        pivot = shoulder_pan_position(self.model, self.data)
        angles = self._sample_object_angles(count)
        radii = np.random.uniform(*self.object_placement.shoulder_distance_meters, size=count)
        return pivot[:2] + radii[:, None] * np.stack([np.cos(angles), np.sin(angles)], axis=1)

    def _resting_height(self, body_id: int) -> float:
        """Half the geom's vertical extent, so the object sits on the floor rather than in it."""
        return float(self.model.geom_size[self.model.body_geomadr[body_id]][2])

    def _paint(self, body_id: int, color: str) -> None:
        material_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, f"matte_{color}")
        assert material_id != -1, f"Material 'matte_{color}' not found in MuJoCo model."
        self.model.geom_matid[self.model.body_geomadr[body_id]] = material_id

    def randomize_physics(self, enable_added_weight: bool):
        ranges = self.servo.physics_randomization
        count = len(self.actuator_dof_indices)
        self.model.dof_damping[self.actuator_dof_indices] = np.random.uniform(*ranges.damping, size=count)
        self.model.dof_frictionloss[self.actuator_dof_indices] = np.random.uniform(*ranges.frictionloss, size=count)
        self.model.dof_armature[self.actuator_dof_indices] = np.random.uniform(*ranges.armature, size=count)
        added_weight_kg = self.box_mass_kg * np.random.uniform(*ranges.added_weight.box_mass_fraction) if enable_added_weight else 0.0
        self.model.body_mass[self.added_weight_body_id] = added_weight_kg
        self.model.body_inertia[self.added_weight_body_id] = added_weight_kg * self.added_weight_inertia_per_kg
        mujoco.mj_setConst(self.model, self.data)

    def physics_metrics(self) -> Dict[str, float]:
        randomized = {
            "damping": self.model.dof_damping,
            "frictionloss": self.model.dof_frictionloss,
            "armature": self.model.dof_armature,
        }
        metrics = {
            f"physics/{parameter}/{name}": float(values[dof_index])
            for parameter, values in randomized.items()
            for name, dof_index in zip(self.actuator_order, self.actuator_dof_indices)
        }
        metrics["physics/added_weight_kg"] = float(self.model.body_mass[self.added_weight_body_id])
        return metrics

    def randomize_objects(self):
        body_names = (*BOX_BODY_NAMES, TILE_BODY_NAME)
        positions = self._sample_object_positions(len(body_names))
        colors = np.random.choice(OBJECT_COLORS, size=len(body_names), replace=False)
        box_offsets = positions[: len(BOX_BODY_NAMES)] - shoulder_pan_position(self.model, self.data)[:2]
        # The grasp closes tangentially around the shoulder, along the rotated box's local Y axis.
        half_yaws = 0.5 * np.arctan2(box_offsets[:, 1], box_offsets[:, 0])
        box_quaternions = np.zeros((len(BOX_BODY_NAMES), 4))
        box_quaternions[:, 0] = np.cos(half_yaws)
        box_quaternions[:, 3] = np.sin(half_yaws)

        for body_name, position, color, quaternion in zip(BOX_BODY_NAMES, positions, colors, box_quaternions):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            qpos_adr = self.model.jnt_qposadr[self.model.body_jntadr[body_id]]
            self.data.qpos[qpos_adr : qpos_adr + 3] = (*position, self._resting_height(body_id))
            self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = quaternion
            self._paint(body_id, color)

        tile_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, TILE_BODY_NAME)
        self.model.body_pos[tile_id] = (*positions[-1], self._resting_height(tile_id))
        self._paint(tile_id, colors[-1])

    def initialize_arm_pos(self):
        if self.initial_joint_mode == "random_range":
            for name, joint_id in self.joint_indices.items():
                jnt_range = self.model.jnt_range[joint_id]
                qpos_idx = self.model.jnt_qposadr[joint_id]

                jmin, jmax = jnt_range[0], jnt_range[1]
                span = jmax - jmin
                min_percent, max_percent = self.initial_joint_range_percent
                safe_min = jmin + (min_percent / 100.0) * span
                safe_max = jmin + (max_percent / 100.0) * span
                self.data.qpos[qpos_idx] = np.random.uniform(safe_min, safe_max)
        elif self.initial_joint_mode == "fixed":
            for name, joint_id in self.joint_indices.items():
                qpos_idx = self.model.jnt_qposadr[joint_id]
                self.data.qpos[qpos_idx] = self.initial_joint_positions[name]
        else:
            raise ValueError(f"Unknown initial joint mode: {self.initial_joint_mode!r}")

    def apply(self, enable_added_weight: bool):
        # Object placement uses the shoulder anchor calculated by forward kinematics.
        mujoco.mj_forward(self.model, self.data)
        self.randomize_physics(enable_added_weight)
        self.randomize_objects()
        self.initialize_arm_pos()
        mujoco.mj_forward(self.model, self.data)
