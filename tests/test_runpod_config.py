from pathlib import Path

import pytest

from deployment.runpod.common import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "deployment/runpod"


def test_training_configs_share_infrastructure():
    vla = load_config(CONFIG_DIR / "train.toml", "train")
    joint = load_config(CONFIG_DIR / "train_joint.toml", "train")
    assert vla["training"]["type"] == "vla_dagger"
    assert joint["training"]["type"] == "joint"
    assert joint["training"]["overrides"] == ["training.check_power_profile=false", "training.tensorboard_enabled=true"]
    assert vla["training"]["archive"] == joint["training"]["archive"]
    assert vla["environment"] == joint["environment"]
    vla["pod"].pop("name")
    joint["pod"].pop("name")
    assert vla["pod"] == joint["pod"]


def test_preparation_keeps_cpu_settings_and_shared_volume():
    preparation = load_config(CONFIG_DIR / "prepare_environment.toml", "prepare")
    training = load_config(CONFIG_DIR / "train.toml", "train")
    assert preparation["pod"]["computeType"] == "CPU"
    assert "gpuTypeIds" not in preparation["pod"]
    assert preparation["pod"]["networkVolumeId"] == training["pod"]["networkVolumeId"]
    assert preparation["local_run_dir"] == "outputs/runpod/preparation"


def test_downloads_and_pods_use_shared_region():
    training = load_config(CONFIG_DIR / "train.toml", "train")
    vla = load_config(CONFIG_DIR / "download.toml", "download")
    joint = load_config(CONFIG_DIR / "download_joint.toml", "download")
    assert vla["endpoint_url"] == joint["endpoint_url"]
    assert training["pod"]["dataCenterIds"] == [vla["region"]] == [joint["region"]]
    assert vla["local_policy_dir"] != joint["local_policy_dir"]


def test_nested_job_overrides_keep_other_shared_fields(tmp_path):
    path = tmp_path / "job.toml"
    path.write_text('region = "test-region"\n[pod]\nname = "test"\ngpuCount = 2\n')
    cfg = load_config(path, "train")
    assert cfg["pod"]["name"] == "test"
    assert cfg["pod"]["gpuCount"] == 2
    assert cfg["pod"]["computeType"] == "GPU"
    assert cfg["pod"]["dataCenterIds"] == ["test-region"]


def test_missing_config_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.toml", "train")
