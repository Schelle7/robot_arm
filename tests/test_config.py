from pathlib import Path

from hydra import compose, initialize_config_dir


CONFIG_DIR = Path(__file__).resolve().parents[1] / "conf"


def test_one_step_experiment_composes_expected_control_config():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config", overrides=["experiment=one_step_cuda"])

    assert cfg.device == "cuda"
    assert cfg.control.frequencies.high_level == 5
    assert cfg.control.frequencies.low_level == 5
    assert cfg.control.frequencies.mujoco == 200
    assert cfg.waypoint.trajectory_length == 1
    assert cfg.control.action_scale_radians_per_second == 0.05
    assert cfg.waypoint.position_speed_meters_per_second == 0.30
    assert cfg.waypoint.rotation_speed_radians_per_second == 1.5707963267948966
    assert cfg.waypoint.gripper_speed_units_per_second == 1.0
