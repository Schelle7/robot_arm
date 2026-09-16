import json
from pathlib import Path
import tomllib

import pytest

from deployment.runpod import download


@pytest.fixture
def run(tmp_path, monkeypatch):
    with Path(download.__file__).with_suffix(".toml").open("rb") as file:
        cfg = tomllib.load(file)
    run_id = "20260915T160633763177Z"
    local = tmp_path / cfg["local_run_dir"] / run_id
    local.mkdir(parents=True)
    saved = {
        "run_id": run_id,
        "training": {"persistent_root": "/workspace/training"},
        "pod": {"volumeMountPath": "/workspace", "networkVolumeId": "volume"},
    }
    (local / "config.json").write_text(json.dumps(saved))
    prefix = f"training/{run_id}"
    objects = {f"{prefix}/console.log": b"training log"}
    events = []

    class FakeVolume:
        def __init__(self, config, bucket):
            assert bucket == "volume"

        def keys(self, key):
            events.append(("list", key))
            return sorted(name for name in objects if name.startswith(key))

        def download(self, key, destination):
            events.append(("download", key))
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(objects[key])

    monkeypatch.setattr(download, "Volume", FakeVolume)
    return cfg, tmp_path, local, prefix, objects, events


@pytest.mark.parametrize("exit_code", [None, 1])
def test_log_downloaded_before_unavailable_status(run, capsys, exit_code):
    cfg, root, local, prefix, objects, events = run
    if exit_code is not None:
        objects[f"{prefix}/exit_code"] = str(exit_code).encode()
    download.download_latest(cfg, root)
    assert events[0] == ("download", f"{prefix}/console.log")
    assert (local / "console.log").read_bytes() == b"training log"
    assert "Final checkpoint is not available" in capsys.readouterr().out
    assert not (root / cfg["local_policy_dir"]).exists()


def finished_run(run):
    cfg, root, local, prefix, objects, events = run
    round_prefix = f"{prefix}/dagger/round_008"
    model_prefix = f"{round_prefix}/training/checkpoints/00010000/pretrained_model"
    objects.update({
        f"{prefix}/exit_code": b"0",
        f"{prefix}/config.json": b"{}",
        f"{prefix}/dagger/.hydra/config.yaml": b"vla_rounds: 8",
        f"{round_prefix}/round.yaml": b"round_index: 8",
        f"{prefix}/dagger/rounds.json": json.dumps([{
            "checkpoint": f"/workspace/{round_prefix}/training/checkpoints/last",
            "training_step": 10000,
        }]).encode(),
        f"{model_prefix}/model.safetensors": b"weights",
        f"{model_prefix}/config.json": b"{}",
        f"{model_prefix}/policy_preprocessor.json": b'{"steps": []}',
        f"{model_prefix}/policy_postprocessor.json": b'{"steps": [{"state_file": "normalizer.safetensors"}]}',
        f"{model_prefix}/normalizer.safetensors": b"normalization",
    })
    return model_prefix


def test_finished_run_downloads_numbered_checkpoint_and_configs(run):
    cfg, root, local, prefix, objects, events = run
    finished_run(run)
    download.download_latest(cfg, root)
    destination = root / cfg["local_policy_dir"] / "runpod_2026-09-15_16-06-33"
    assert (destination / "checkpoints/last/pretrained_model/model.safetensors").read_bytes() == b"weights"
    assert (destination / "round.yaml").read_bytes() == b"round_index: 8"
    assert (destination / "dagger_config.yaml").is_file()
    assert (destination / "deployment_config.json").is_file()
    assert (destination / "console.log").read_bytes() == b"training log"
    assert not (root / cfg["local_policy_dir"] / "latest_run.txt").exists()


def test_missing_processor_state_does_not_publish_checkpoint(run, capsys):
    cfg, root, local, prefix, objects, events = run
    model_prefix = finished_run(run)
    del objects[f"{model_prefix}/normalizer.safetensors"]
    download.download_latest(cfg, root)
    assert "Final checkpoint is not available" in capsys.readouterr().out
    assert (local / "console.log").is_file()
    assert list((root / cfg["local_policy_dir"]).iterdir()) == []
