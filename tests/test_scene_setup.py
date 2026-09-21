from pathlib import Path

import mujoco
import numpy as np
from omegaconf import OmegaConf

from robot_arm.geometry.waypoints import shoulder_pan_position
from robot_arm.robot_schema import BOX_BODY_NAMES, MOTOR_ORDER, TILE_BODY_NAME
from robot_arm.simulation.scene_setup import SceneSetup


def test_scene_setup_places_objects_and_initializes_joints():
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.create(
        {
            "servo": OmegaConf.load(root / "conf/servo/default.yaml"),
            "scene": OmegaConf.load(root / "conf/scene/default.yaml"),
            "control": OmegaConf.load(root / "conf/control/default.yaml"),
            "runtime": {"disable_box_collisions": False},
        }
    )
    cfg.control.initial_joints.mode = "fixed"
    model = mujoco.MjModel.from_xml_path(str(root / "models/so101/scene.xml"))
    data = mujoco.MjData(model)
    setup = SceneSetup(model, data, cfg)

    setup.apply(enable_added_weight=False)

    joint_ids = np.array([model.joint(name).id for name in MOTOR_ORDER])
    expected = np.array([cfg.control.initial_joints.positions_radians[name] for name in MOTOR_ORDER])
    np.testing.assert_allclose(data.qpos[model.jnt_qposadr[joint_ids]], expected)
    positions = np.array([data.body(name).xpos[:2] for name in (*BOX_BODY_NAMES, TILE_BODY_NAME)])
    radii = np.linalg.norm(positions - shoulder_pan_position(model, data)[:2], axis=1)
    minimum, maximum = cfg.scene.object_placement.shoulder_distance_meters
    assert np.all((radii >= minimum) & (radii <= maximum))
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    assert np.all(distances[np.triu_indices(len(positions), k=1)] >= cfg.scene.object_placement.min_separation_meters)
    assert setup.physics_metrics()["physics/added_weight_kg"] == 0.0
