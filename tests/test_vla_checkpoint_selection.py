import json

import pytest

from robot_arm.policies import cartesian


def make_checkpoint(training_dir):
    checkpoint = training_dir / "checkpoints/last/pretrained_model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "config.json").write_text("{}")
    for name in ("policy_preprocessor", "policy_postprocessor"):
        (checkpoint / f"{name}.json").write_text(json.dumps({
            "steps": [{"state_file": f"{name}.safetensors"}],
        }))
        (checkpoint / f"{name}.safetensors").write_bytes(b"state")
    return checkpoint


@pytest.fixture
def outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(cartesian, "__file__", str(tmp_path / "src/robot_arm/policies/cartesian.py"))
    return tmp_path / "outputs"


def test_latest_run_uses_folder_timestamp_and_ignores_pointer(outputs, capsys):
    newest = make_checkpoint(outputs / "train_vla/runpod_2026-09-15_16-06-33")
    older = make_checkpoint(outputs / "train_vla/2026-09-11/15-08-32/training")
    (outputs / "train_vla/latest_run.txt").write_text(str(older))
    assert cartesian.latest_vla_checkpoint_path() == str(newest)
    assert capsys.readouterr().out == ""


def test_incomplete_newest_checkpoint_uses_older_run_with_message(outputs, capsys):
    older = make_checkpoint(outputs / "train_vla/runpod_2026-09-15_16-06-33")
    newest = make_checkpoint(outputs / "train_vla/2026-09-16/10-00-00/training")
    (newest / "policy_postprocessor.safetensors").unlink()
    assert cartesian.latest_vla_checkpoint_path() == str(older)
    message = capsys.readouterr().out
    assert "Using an older available VLA checkpoint" in message
    assert "2026-09-16/10-00-00" in message


def test_dagger_round_without_training_uses_previous_round(outputs, capsys):
    run = outputs / "train_vla_dagger/2026-09-16/10-00-00"
    checkpoint = make_checkpoint(run / "round_000/training")
    (run / "round_001").mkdir()
    assert cartesian.latest_vla_checkpoint_path() == str(checkpoint)
    assert "round_001" in capsys.readouterr().out


def test_no_available_checkpoint_raises(outputs):
    (outputs / "train_vla/2026-09-16/10-00-00").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No complete VLA inference checkpoint"):
        cartesian.latest_vla_checkpoint_path()
