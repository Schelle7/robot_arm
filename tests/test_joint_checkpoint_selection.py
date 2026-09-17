from pathlib import Path

from robot_arm.policies import checkpoints


def test_latest_inference_export_does_not_require_training_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoints, "__file__", str(tmp_path / "src/robot_arm/policies/checkpoints.py"))
    outputs = tmp_path / "outputs/train_joint_policy"
    actor = outputs / "2026-09-16/12-00-00/checkpoints/jax_sac_final_100.actor.npz"
    actor.parent.mkdir(parents=True)
    actor.write_bytes(b"actor")
    training_only = outputs / "2026-09-17/12-00-00/checkpoints/jax_sac_final_200.pkl"
    training_only.parent.mkdir(parents=True)
    training_only.write_bytes(b"training state")

    assert checkpoints.find_latest_joint_checkpoint() == str(actor)


def test_explicit_checkpoint_path_is_not_replaced(tmp_path):
    checkpoint = tmp_path / "chosen.actor.npz"
    assert Path(checkpoints.resolve_joint_checkpoint(str(checkpoint))) == checkpoint
