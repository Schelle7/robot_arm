from typing import Dict
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from robot_arm.backends.arm import Arm
from robot_arm.pose import Pose


def get_tcp_geometry(model, data):
    fixed_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "fixed_finger_tip"
    )
    moving_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "moving_finger_tip"
    )
    fixed = data.site_xpos[fixed_id].copy()
    moving = data.site_xpos[moving_id].copy()

    frame_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
    )
    secondary = data.site_xmat[frame_id].reshape(3, 3)[:, 1]
    closing = fixed - moving
    closing = closing / np.linalg.norm(closing)
    secondary = secondary - np.dot(secondary, closing) * closing
    secondary = secondary / np.linalg.norm(secondary)

    gripper_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "gripper"
    )
    gripper_qpos_id = model.jnt_qposadr[gripper_joint_id]
    pose = Pose.from_tcp_axes(
        (fixed + moving) / 2.0,
        closing,
        secondary,
        float(data.qpos[gripper_qpos_id]),
    )

    return pose, fixed, moving


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
    active_alpha = 1.0
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
    high_level_delta_action: np.ndarray,
    desired_path_visual_exaggeration_factor: float,
):
    desired_poses = []
    for delta in high_level_delta_action:
        displayed_position = (
            start_pose.position
            + desired_path_visual_exaggeration_factor * delta[:3]
        )
        displayed_rotation = start_pose.rotation * Rotation.from_rotvec(
            desired_path_visual_exaggeration_factor * delta[3:6]
        )
        desired_poses.append(
            Pose.from_matrix(
                displayed_position,
                displayed_rotation.as_matrix(),
                start_pose.gripper,
            )
        )
    return desired_poses


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
        height: int,
        width: int,
        initial_joint_range_percent: tuple[float, float],
    ):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.initial_joint_range_percent = initial_joint_range_percent

        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.waypoints = []
        self.active_waypoint_index = 0
        self.desired_poses = []
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

    def get_tcp_pose(self) -> Pose:
        pose, _, _ = get_tcp_geometry(self.model, self.data)
        return pose

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
        # Keep the target around x=0.4 m and centered across y with 3 cm and 5 cm variation.
        box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_box")
        if box_id == -1:
            raise Exception("Box is missing")

        dist = np.random.uniform(0.37, 0.43)
        y_shift = np.random.uniform(-0.05, 0.05)

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
            min_percent, max_percent = self.initial_joint_range_percent
            safe_min = jmin + (min_percent / 100.0) * span
            safe_max = jmin + (max_percent / 100.0) * span
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
        self._draw_waypoint_arrows()
        self._draw_desired_pose_path()
        return self.renderer.render()

    def get_tcp_axes(self) -> tuple[np.ndarray, np.ndarray]:
        pose = self.get_tcp_pose()
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
        high_level_delta_action: np.ndarray,
        desired_path_visual_exaggeration_factor: float,
    ):
        self.desired_poses = build_desired_poses(
            start_pose,
            high_level_delta_action,
            desired_path_visual_exaggeration_factor,
        )

    def _draw_waypoint_arrows(self):
        waypoint_arrow_length = 0.09
        inactive_alpha = 0.25
        active_alpha = 1.0
        arrow_specs = (
            (lambda pose: pose.closing_axis, (0.1, 0.9, 0.2)),
            (lambda pose: pose.secondary_axis, (0.1, 0.5, 1.0)),
        )

        for index, pose in enumerate(self.waypoints):
            alpha = active_alpha if index == self.active_waypoint_index else inactive_alpha
            for vector_getter, color in arrow_specs:
                self._add_arrow(
                    pose.position,
                    vector_getter(pose) * waypoint_arrow_length,
                    (*color, alpha),
                    0.004,
                )

    def _draw_desired_pose_path(self):
        update_desired_pose_debug_user_scene(self.renderer.scene, self.desired_poses)

    def _add_arrow(self, origin, vector, color, width):
        if np.linalg.norm(vector) <= 1e-5:
            return

        geom = self.renderer.scene.geoms[self.renderer.scene.ngeom]
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
        self.renderer.scene.ngeom += 1
