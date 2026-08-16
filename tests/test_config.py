from pathlib import Path

from hydra import compose, initialize_config_dir

CONFIG_DIR = Path(__file__).resolve().parents[1] / "conf"


def test_debug_experiment_composes_expected_control_config():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config", overrides=["experiment=debug"])

    assert cfg.device == "cuda"
    assert cfg.control.frequencies.mid_level == 5
    assert cfg.control.frequencies.low_level == 5
    assert cfg.control.frequencies.mujoco == 200
    assert cfg.waypoint.trajectory_length == 1
    assert cfg.control.action_scale_radians_per_second == 0.5
    assert cfg.waypoint.position_speed_meters_per_second == 0.05
    assert cfg.waypoint.rotation_speed_radians_per_second == 0.2
    assert cfg.waypoint.gripper_speed_radians_per_second == 1.0
    assert cfg.reward.tracking.deviation is False
    assert cfg.training.learning_starts == 5000
