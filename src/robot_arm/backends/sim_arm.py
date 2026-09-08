from typing import Dict
import mujoco
import numpy as np

from robot_arm.backends.arm import Arm
from robot_arm.waypoints import shoulder_pan_position
from robot_arm.gripper_geometry import get_tcp_geometry
from robot_arm.pose import Pose
from robot_arm.backends.servo import duty_from_action, duty_to_torque
from robot_arm.robot_schema import BOX_BODY_NAMES, CAMERA_NAMES, OBJECT_COLORS, TILE_BODY_NAME


def object_color(model, body_name: str) -> str:
    """Reads back the colour assigned at reset, so the model stays the only record of the scene."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    assert body_id != -1, f"Body {body_name!r} not found in MuJoCo model."
    material_id = model.geom_matid[model.body_geomadr[body_id]]
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, material_id).removeprefix("matte_")


def tcp_debug_segments(model, data):
    pose, fixed, moving = get_tcp_geometry(model, data)

    return (
        ("debug_actual_closing", moving, fixed - moving),
        ("debug_actual_secondary", pose.position, pose.secondary_axis * 0.09),
    )


def update_tcp_debug_user_scene(scene, model, data):
    scene.ngeom = 0
    draw_scale = 2.0  # Tests were inconclusive; keep the rendered vector unscaled.
    # draw scale shouldnt be needed but is for whatever reason
    colors = ((0.1, 0.9, 0.2, 0.35), (0.1, 0.5, 1.0, 0.35))
    for (_, origin, vector), color in zip(tcp_debug_segments(model, data), colors):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).reshape(-1),
            color,
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            0.004,
            origin,
            origin + draw_scale * vector,
        )
        scene.ngeom += 1

    pose, _, _ = get_tcp_geometry(model, data)
    tcp_size = np.full(3, 0.003)
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        tcp_size,
        pose.position,
        np.eye(3).reshape(-1),
        (1.0, 0.8, 0.1, 1.0),
    )
    scene.ngeom += 1


def update_waypoint_debug_user_scene(scene, waypoints, active_waypoint_index):
    waypoint_arrow_length = 0.09
    inactive_alpha = 0.25
    active_alpha = 0.6
    arrow_specs = (
        (lambda pose: pose.closing_axis, (0.1, 0.9, 0.2)),
        (lambda pose: pose.secondary_axis, (0.1, 0.5, 1.0)),
    )
    poses = [Pose.from_10d(waypoint) for waypoint in waypoints]

    for index, pose in enumerate(poses):
        alpha = active_alpha if index == active_waypoint_index else inactive_alpha
        for vector_getter, color in arrow_specs:
            origin = pose.position
            vector = vector_getter(pose) * waypoint_arrow_length
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).reshape(-1),
                (*color, alpha),
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                0.004,
                origin,
                origin + vector,
            )
            scene.ngeom += 1


def build_desired_poses(
    start_pose: Pose,
    cartesian_action: np.ndarray,
):
    return [start_pose.apply_delta(cartesian_action)]


def update_desired_pose_debug_user_scene(scene, desired_poses):
    arrow_specs = (
        (lambda pose: pose.closing_axis, (0.0, 0.35, 0.05, 0.75)),
        (lambda pose: pose.secondary_axis, (0.0, 0.15, 0.45, 0.75)),
    )
    desired_arrow_length = 0.09
    for pose in desired_poses:
        for vector_getter, color in arrow_specs:
            if np.linalg.norm(vector_getter(pose)) <= 1e-5:
                continue
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).reshape(-1),
                color,
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                0.003,
                pose.position,
                pose.position + vector_getter(pose) * desired_arrow_length,
            )
            scene.ngeom += 1


class SimBackend(Arm):
    """
    Simulation adapter for the SO-101 using MuJoCo.
    Operates in radians (unlike RealArm which uses raw steps/bits).
    Includes and manages simulation scene elements (like the target box).
    Unit conversion is done higher up the stack.
    """

    def __init__(
        self,
        model_path: str,
        camera_configs,
        initial_joint_mode: str,
        initial_joint_range_percent: tuple[float, float],
        initial_joint_positions,
        disable_box_collisions: bool,
        object_placement,
        mujoco_steps_per_control_step: int,
        servo,
    ):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.added_weight_body_id = self.model.body("added_weight").id
        self.box_mass_kg = float(self.model.body(BOX_BODY_NAMES[0]).mass[0])
        added_weight_radius = float(self.model.geom("added_weight_geom").size[0])
        self.added_weight_inertia_per_kg = 2.0 / 5.0 * added_weight_radius**2
        if disable_box_collisions:
            for body_name in BOX_BODY_NAMES:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                assert body_id != -1, f"Body {body_name!r} not found in MuJoCo model."
                geom_start = self.model.body_geomadr[body_id]
                geom_count = self.model.body_geomnum[body_id]
                self.model.geom_contype[geom_start : geom_start + geom_count] = 0
                self.model.geom_conaffinity[geom_start : geom_start + geom_count] = 0
        self.data = mujoco.MjData(self.model)
        self.initial_joint_mode = initial_joint_mode
        self.initial_joint_range_percent = initial_joint_range_percent
        self.initial_joint_positions = initial_joint_positions
        self.object_placement = object_placement
        self.mujoco_steps_per_control_step = mujoco_steps_per_control_step
        self.servo = servo

        self.camera_configs = camera_configs
        assert tuple(camera_configs) == CAMERA_NAMES
        self.renderers = {
            camera_name: mujoco.Renderer(self.model, height=config.height, width=config.width)
            for camera_name, config in camera_configs.items()
        }
        self.waypoints = []
        self.active_waypoint_index = 0
        self.desired_poses = []
        self.camera_scene_option = mujoco.MjvOption()
        self.camera_scene_option.geomgroup[5] = 0

        # Build explicit mappings for actuator and joint indices
        self.actuator_indices = {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(self.model.nu)}

        self.joint_indices = {name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.actuator_indices}

        # Indexed in actuator order so the servo law runs on all six joints as one vector operation.
        self.actuator_order = sorted(self.actuator_indices, key=self.actuator_indices.get)
        self.actuator_dof_indices = np.array([self.model.jnt_dofadr[self.joint_indices[name]] for name in self.actuator_order])
        self.max_duty = np.array([float(servo.max_duty[name]) for name in self.actuator_order])
        self.commanded_duty = np.zeros(self.model.nu)

    @property
    def fixed_finger_tip(self) -> np.ndarray:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "fixed_finger_tip")
        return self.data.site_xpos[site_id].copy()

    @property
    def moving_finger_tip(self) -> np.ndarray:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "moving_finger_tip")
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

    def get_tcp_pose(self, state: Dict[str, Dict[str, float]]) -> Pose:
        pose, _, _ = get_tcp_geometry(self.model, self.data)
        return pose

    def get_privileged_box_pose(self, body_name: str = BOX_BODY_NAMES[0]) -> Pose:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise KeyError(f"Body {body_name!r} not found in MuJoCo model.")

        pos = self.data.xpos[body_id].copy()
        rot_mat = self.data.xmat[body_id].reshape(3, 3)

        return Pose.from_matrix(pos, rot_mat, 1.0)  # pose with gripper info is a bit weird but ok for now

    def read_state(self) -> Dict[str, Dict[str, float]]:
        # Map MuJoCo qpos, qvel and the commanded duty to our expected dictionary format
        sample_time_ns = round(self.data.time * 1_000_000_000)
        state = {
            "Present_Position": {},
            "Present_Velocity": {},
            "Present_Load": {},  # Returning actuator control effort as load
            "Present_Voltage": {},  # Dummy data
            "Present_Temperature": {},  # Dummy data
            "read_started_ns": sample_time_ns,
            "read_completed_ns": sample_time_ns,
            "sample_time_ns": sample_time_ns,
        }

        for name, actuator_idx in self.actuator_indices.items():
            qpos_idx = self.model.jnt_qposadr[self.joint_indices[name]]
            qvel_idx = self.model.jnt_dofadr[self.joint_indices[name]]

            state["Present_Position"][name] = float(self.data.qpos[qpos_idx])
            state["Present_Velocity"][name] = float(self.data.qvel[qvel_idx])
            # The duty the servo law commanded, which is what Present_Load reports on the real arm:
            # a fraction of full output, not a torque.
            state["Present_Load"][name] = float(self.commanded_duty[actuator_idx] / self.servo.full_scale_duty)

            state["Present_Voltage"][name] = 12.0
            state["Present_Temperature"][name] = 40.0

        return state

    def sim_state(self) -> Dict[str, np.ndarray]:
        """The simulator's own state, which no sensor on the real arm can report."""
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
        }

    def restore_sim_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.commanded_duty[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def write_duty(self, duties: Dict[str, float]) -> None:
        requested = np.zeros(self.model.nu)
        for name, duty_fraction in duties.items():
            requested[self.actuator_indices[name]] = duty_fraction
        self.commanded_duty = duty_from_action(
            requested,
            self.servo.full_scale_duty,
            self.servo.min_startup_duty,
            self.max_duty,
        )

    def _apply_servo_torques(self) -> None:
        """
        The duty is held for the whole control period the way the servo's open loop PWM mode holds
        it, so only the back-EMF term varies as the joint picks up speed.
        """
        self.data.ctrl[:] = duty_to_torque(
            self.commanded_duty,
            self.data.qvel[self.actuator_dof_indices],
            self.servo.stall_torque_newton_meters,
            self.servo.no_load_speed_radians_per_second,
        )

    def advance_control_step(self) -> None:
        for _ in range(self.mujoco_steps_per_control_step):
            self._apply_servo_torques()
            mujoco.mj_step(self.model, self.data)

    def disconnect(self):
        """Simulation doesn't need to physically disconnect power."""
        pass

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
        added_weight_kg = (
            self.box_mass_kg * np.random.uniform(*ranges.added_weight.box_mass_fraction) if enable_added_weight else 0.0
        )
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

    def reset_sim(self, enable_added_weight: bool):
        mujoco.mj_resetData(self.model, self.data)
        self.commanded_duty[:] = 0.0
        # Placement is measured from the shoulder anchor, which is only valid once kinematics have run.
        mujoco.mj_forward(self.model, self.data)

        self.randomize_physics(enable_added_weight)
        self.randomize_objects()
        self.initialize_arm_pos()

        mujoco.mj_forward(self.model, self.data)

    def read_cameras(self) -> dict[str, np.ndarray]:
        images = {}
        for camera_name, renderer in self.renderers.items():
            renderer.update_scene(
                self.data,
                camera=self.camera_configs[camera_name].mujoco_name,
                scene_option=self.camera_scene_option,
            )
            images[camera_name] = renderer.render()
        return images

    def get_tcp_axes(self) -> tuple[np.ndarray, np.ndarray]:
        pose, _, _ = get_tcp_geometry(self.model, self.data)
        return pose.closing_axis, pose.secondary_axis

    def draw_tcp(self):
        pass

    def draw_waypoints(self, waypoints: np.ndarray):
        self.waypoints = [Pose.from_10d(waypoint) for waypoint in waypoints]
        self.active_waypoint_index = 0

    def update_waypoint_index(self, active_waypoint_index: int):
        self.active_waypoint_index = active_waypoint_index

    def draw_desired_path(
        self,
        start_pose: Pose,
        cartesian_action: np.ndarray,
    ):
        self.desired_poses = build_desired_poses(
            start_pose,
            cartesian_action,
        )

    def _draw_waypoint_arrows(self, renderer):
        waypoint_arrow_length = 0.09
        inactive_alpha = 0.25
        active_alpha = 0.6
        arrow_specs = (
            (lambda pose: pose.closing_axis, (0.1, 0.9, 0.2)),
            (lambda pose: pose.secondary_axis, (0.1, 0.5, 1.0)),
        )

        for index, pose in enumerate(self.waypoints):
            alpha = active_alpha if index == self.active_waypoint_index else inactive_alpha
            for vector_getter, color in arrow_specs:
                self._add_arrow(
                    renderer,
                    pose.position,
                    vector_getter(pose) * waypoint_arrow_length,
                    (*color, alpha),
                    0.004,
                )

    def _draw_desired_pose_path(self, renderer):
        update_desired_pose_debug_user_scene(renderer.scene, self.desired_poses)

    def _add_arrow(self, renderer, origin, vector, color, width):
        if np.linalg.norm(vector) <= 1e-5:
            return

        geom = renderer.scene.geoms[renderer.scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).reshape(-1),
            color,
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            width,
            origin,
            origin + vector,
        )
        renderer.scene.ngeom += 1
