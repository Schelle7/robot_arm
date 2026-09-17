import json
from pathlib import Path

from robot_arm.run_paths import find_runs


def find_latest_joint_checkpoint() -> str:
    outputs_dir = Path(__file__).resolve().parents[3] / "outputs/train_joint_policy"
    for run in find_runs([outputs_dir]):
        for checkpoint in sorted(run.glob("checkpoints/jax_sac_final_*.actor.npz"), reverse=True):
            return str(checkpoint)
    raise FileNotFoundError("No final joint policy checkpoints found in any outputs directory.")


def resolve_joint_checkpoint(policy_name: str) -> str:
    if policy_name == "latest":
        return find_latest_joint_checkpoint()

    policy_path = Path(policy_name)
    if not policy_path.is_absolute():
        policy_path = Path(__file__).resolve().parents[3] / policy_path
    return str(policy_path.resolve())


def latest_vla_checkpoint_path() -> str:
    outputs = Path(__file__).resolve().parents[3] / "outputs"
    skipped = []
    for run in find_runs([outputs / "train_vla", outputs / "train_vla_dagger"]):
        if run.name.startswith("runpod_"):
            training_dirs = [run]
        elif run.parent.parent.name == "train_vla_dagger":
            training_dirs = [round_dir / "training" for round_dir in sorted(run.glob("round_*"), reverse=True)]
        else:
            training_dirs = [run / "training"]
        for training_dir in training_dirs:
            checkpoint = training_dir / "checkpoints" / "last" / "pretrained_model"
            if not _vla_checkpoint_available(checkpoint):
                skipped.append(training_dir)
                continue
            if skipped:
                print(
                    f"Using an older available VLA checkpoint: {checkpoint}. "
                    f"Newer run or round {skipped[0]} has no complete inference checkpoint; "
                    "it may still be training, downloading, or may have failed."
                )
            return str(checkpoint)
        if not training_dirs:
            skipped.append(run)
    raise FileNotFoundError(f"No complete VLA inference checkpoint found under {outputs}.")


def _vla_checkpoint_available(checkpoint: Path) -> bool:
    required = [checkpoint / name for name in (
        "model.safetensors", "config.json", "policy_preprocessor.json", "policy_postprocessor.json",
    )]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        processor = json.loads((checkpoint / name).read_text())
        for step in processor["steps"]:
            if "state_file" in step:
                state_file = checkpoint / step["state_file"]
                if not state_file.is_file() or state_file.stat().st_size == 0:
                    return False
    return True
