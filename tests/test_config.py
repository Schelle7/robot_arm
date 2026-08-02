from pathlib import Path

from hydra import compose, initialize_config_dir


CONFIG_DIR = Path(__file__).resolve().parents[1] / "conf"


def test_one_step_experiment_composes_expected_control_config():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config", overrides=["experiment=one_step_cuda"])

    assert cfg.device == "cuda"
    assert cfg.frequencies.high_level == 5
    assert cfg.frequencies.low_level == 5
    assert cfg.frequencies.mujoco == 200
    assert cfg.trajectory_length == 1
    assert cfg.training.action_scale_radians == 0.01
    assert cfg.training.waypoint_speed == 0.005
