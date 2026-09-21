from typing import Dict
import mujoco
import numpy as np

from robot_arm.simulation.scene_setup import SceneSetup
from robot_arm.arms.arm import Arm
from robot_arm.arms.sim_communication import SimCommunication
from robot_arm.geometry.gripper_geometry import get_tcp_geometry
from robot_arm.geometry.pose import Pose
from robot_arm.robot_schema import BOX_BODY_NAMES, CAMERA_NAMES


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


class SimArm(Arm):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.mujoco_steps_per_control_step = cfg.control.frequencies.mujoco // cfg.control.frequencies.joint

        self.camera_configs = cfg.camera.cameras
        assert tuple(cfg.camera.cameras) == CAMERA_NAMES
        self.renderers = {camera_name: mujoco.Renderer(self.model, height=config.height, width=config.width) for camera_name, config in cfg.camera.cameras.items()}
        self.waypoints = []
        self.active_waypoint_index = 0
        self.desired_poses = []
        self.camera_scene_option = mujoco.MjvOption()
        self.camera_scene_option.geomgroup[5] = 0

        self.scene_setup = SceneSetup(self.model, self.data, cfg)
        self.communication = SimCommunication(cfg, self.model, self.data)

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

    def sim_state(self) -> Dict[str, np.ndarray]:
        """The simulator's own state, which no sensor on the real arm can report."""
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "body_pos": self.model.body_pos.copy(),
            "geom_matid": self.model.geom_matid.copy(),
        }

    def _reset_sim_data(self) -> None:
        mujoco.mj_resetData(self.model, self.data)

    def restore_sim_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        self.communication.reset()
        self._reset_sim_data()
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def advance_control_step(self) -> None:
        for _ in range(self.mujoco_steps_per_control_step):
            self.communication.apply_servo_torques()
            mujoco.mj_step(self.model, self.data)

    def disconnect(self):
        self.communication.close()

    def physics_metrics(self) -> Dict[str, float]:
        return self.scene_setup.physics_metrics()

    def reset_sim(self, enable_added_weight: bool):
        self.communication.reset()
        self._reset_sim_data()
        self.scene_setup.apply(enable_added_weight)

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
