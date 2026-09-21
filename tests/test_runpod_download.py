import json
from pathlib import Path

import pytest

from deployment.runpod.common import load_config

from deployment.runpod import download


@pytest.fixture
def run(tmp_path, monkeypatch):
    cfg = load_config(Path(download.__file__).with_suffix(".toml"), "download")
    run_id = "2026-09-15_16-06-33"
    local = tmp_path / cfg["local_run_dir"] / run_id
    local.mkdir(parents=True)
    saved = {
        "run_id": run_id,
        "training": {"type": "vla_dagger", "persistent_root": "/workspace/training"},
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
    objects.update(
        {
            f"{prefix}/exit_code": b"0",
            f"{prefix}/result.json": json.dumps(
                {"training_type": "vla_dagger", "checkpoint": f"/workspace/{round_prefix}/training/checkpoints/00010000", "training_step": 10000}
            ).encode(),
            f"{prefix}/config.json": b"{}",
            f"{prefix}/dagger/.hydra/config.yaml": b"vla_rounds: 8",
            f"{round_prefix}/round.yaml": b"round_index: 8",
            f"{prefix}/dagger/rounds.json": json.dumps(
                [
                    {
                        "checkpoint": f"/workspace/{round_prefix}/training/checkpoints/last",
                        "training_step": 10000,
                    }
                ]
            ).encode(),
            f"{model_prefix}/model.safetensors": b"weights",
            f"{model_prefix}/config.json": b"{}",
            f"{model_prefix}/policy_preprocessor.json": b'{"steps": []}',
            f"{model_prefix}/policy_postprocessor.json": b'{"steps": [{"state_file": "normalizer.safetensors"}]}',
            f"{model_prefix}/normalizer.safetensors": b"normalization",
        }
    )
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


def finished_joint_run(run):
    cfg, root, local, prefix, objects, events = run
    cfg["training_type"] = "joint"
    cfg["local_policy_dir"] = "outputs/train_joint_policy"
    saved = json.loads((local / "config.json").read_text())
    saved["training"]["type"] = "joint"
    (local / "config.json").write_text(json.dumps(saved))
    checkpoint = f"{prefix}/joint/checkpoints/jax_sac_final_30000.pkl"
    objects.update(
        {
            f"{prefix}/exit_code": b"0",
            f"{prefix}/result.json": json.dumps({"training_type": "joint", "checkpoint": f"/workspace/{checkpoint}"}).encode(),
            checkpoint: b"training checkpoint",
            checkpoint.replace(".pkl", ".actor.npz"): b"actor",
            f"{prefix}/joint/.hydra/config.yaml": b"model_path: models/so101/scene.xml",
            f"{prefix}/joint/model/scene.xml": b"<mujoco/>",
            f"{prefix}/joint/model/assets/arm.stl": b"mesh",
        }
    )
    return root / cfg["local_policy_dir"] / "2026-09-15/16-06-33"


def test_joint_download_includes_checkpoint_pair_config_and_model(run):
    cfg, root, local, prefix, objects, events = run
    destination = finished_joint_run(run)
    download.download_latest(cfg, root)
    assert events[0] == ("download", f"{prefix}/console.log")
    assert (destination / "checkpoints/jax_sac_final_30000.pkl").read_bytes() == b"training checkpoint"
    assert (destination / "checkpoints/jax_sac_final_30000.actor.npz").read_bytes() == b"actor"
    assert (destination / ".hydra/config.yaml").is_file()
    assert (destination / "model/assets/arm.stl").read_bytes() == b"mesh"


def test_joint_download_does_not_publish_without_actor(run, capsys):
    cfg, root, local, prefix, objects, events = run
    destination = finished_joint_run(run)
    del objects[f"{prefix}/joint/checkpoints/jax_sac_final_30000.actor.npz"]
    download.download_latest(cfg, root)
    assert not destination.exists()
    assert "Final checkpoint is not available" in capsys.readouterr().out


def test_download_ignores_newer_run_of_other_type(run):
    cfg, root, local, prefix, objects, events = run
    finished_joint_run(run)
    newer = local.parent / "2026-09-16_16-06-33"
    newer.mkdir()
    (newer / "config.json").write_text(json.dumps({"training": {"type": "vla_dagger"}}))
    download.download_latest(cfg, root)
    assert events[0] == ("download", f"{prefix}/console.log")


@pytest.mark.parametrize("kind", ["joint", "vla_dagger"])
def test_existing_download_is_left_unchanged(run, kind, capsys):
    cfg, root, local, prefix, objects, events = run
    if kind == "joint":
        destination = finished_joint_run(run)
    else:
        finished_run(run)
        destination = root / cfg["local_policy_dir"] / "runpod_2026-09-15_16-06-33"
    destination.mkdir(parents=True)
    (destination / "keep").write_text("existing")
    download.download_latest(cfg, root)
    assert list(destination.iterdir()) == [destination / "keep"]
    assert "already downloaded" in capsys.readouterr().out
