import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from omegaconf import OmegaConf

from lerobot.datasets import aggregate

from robot_arm import rollout_config
from robot_arm.data import collection, lerobot_converter
from robot_arm.policies import checkpoints
from robot_arm.policies.cartesian import ScriptedCartesianPolicy, VLACartesianPolicy
from robot_arm.training.vla_bc import train_round


def run_stage(stage: str, cfg, config_path: Path) -> None:
    OmegaConf.save(cfg, config_path, resolve=True)
    subprocess.run(
        [sys.executable, "-m", "robot_arm.training.dagger", stage, str(config_path)],
        check=True,
        env={**os.environ, **dict(cfg.environment)},
    )


def collect_round(cfg) -> None:
    np.random.seed(cfg.seed)
    run_dir = Path(cfg.collection_dir)
    merged, env, joint_policy = rollout_config.setup_rollout_context(cfg.collection, str(run_dir))
    if cfg.round_index == 0:
        policy = ScriptedCartesianPolicy(merged)
    else:
        policy = VLACartesianPolicy(str(Path(cfg.previous_checkpoint) / "pretrained_model"))
    collection.collect_episodes(merged, env, joint_policy, policy, run_dir, cfg.num_episodes)


def prepare_dataset(cfg, round_datasets: list[Path]) -> Path:
    round_dataset = Path(cfg.data_dir) / f"dataset_{cfg.round_index:03d}"
    lerobot_converter.convert_to_lerobot(
        source_dir=str(Path(cfg.collection_dir) / "recordings"),
        target_dir=str(round_dataset),
        fps=cfg.collection.control.frequencies.cartesian,
        video_files_size_in_mb=cfg.dataset.video_files_size_in_mb,
    )
    round_datasets.append(round_dataset)
    if cfg.round_index == 0:
        return round_dataset
    combined = Path(cfg.data_dir) / "combined_dataset"
    aggregate.aggregate_datasets(
        repo_ids=[path.name for path in round_datasets],
        roots=round_datasets,
        aggr_repo_id=combined.name,
        aggr_root=combined,
        video_files_size_in_mb=cfg.dataset.video_files_size_in_mb,
        data_files_size_in_mb=cfg.dataset.data_files_size_in_mb,
        chunk_size=cfg.dataset.chunk_size,
    )
    return combined


def validate_config(cfg) -> None:
    if cfg.keep_last_checkpoints < 2:
        raise ValueError("Keep at least the latest two checkpoints.")
    if cfg.initial_episodes <= 0 or cfg.episodes_per_round <= 0:
        raise ValueError("Episode counts must be positive.")
    if cfg.initial_training_steps <= 0 or cfg.training_steps_per_round <= 0:
        raise ValueError("Training step counts must be positive.")
    if cfg.vla_rounds < 0:
        raise ValueError("VLA round count must be nonnegative.")
    if cfg.collection.backend != "sim":
        raise ValueError("DAgger collection requires the sim backend.")


def build_round_config(cfg, run_dir: Path, round_index: int):
    initial = round_index == 0
    round_dir = run_dir / f"round_{round_index:03d}"
    data_dir = Path(cfg.data_root).resolve() / f"round_{round_index:03d}"
    start_step = 0 if initial else cfg.initial_training_steps + (round_index - 1) * cfg.training_steps_per_round
    updates = cfg.initial_training_steps if initial else cfg.training_steps_per_round
    previous_checkpoint = "" if initial else str(run_dir / f"round_{round_index - 1:03d}/training/checkpoints/last")
    return OmegaConf.merge(
        OmegaConf.to_container(cfg, resolve=True),
        {
            "round_index": round_index,
            "round_dir": str(round_dir),
            "data_dir": str(data_dir),
            "collection_dir": str(data_dir / "collection"),
            "training_dir": str(round_dir / "training"),
            "previous_checkpoint": previous_checkpoint,
            "dataset_root": "",
            "num_episodes": cfg.initial_episodes if initial else cfg.episodes_per_round,
            "start_step": start_step,
            "end_step": start_step + updates,
            "total_steps": cfg.initial_training_steps + cfg.vla_rounds * cfg.training_steps_per_round,
            "seed": cfg.seed + round_index,
        },
    )


def record_completed_round(cfg, run_dir: Path, report: list[dict]) -> None:
    checkpoint = Path(cfg.training_dir) / "checkpoints" / "last"
    model_path = checkpoint / "pretrained_model" / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    report.append(
        {
            "round": cfg.round_index,
            "episodes": cfg.num_episodes,
            "dataset": cfg.dataset_root,
            "training_step": cfg.end_step,
            "checkpoint": str(checkpoint),
            "checkpoint_retained": True,
        }
    )
    expired_round = cfg.round_index - cfg.keep_last_checkpoints
    if expired_round >= 0:
        expired_checkpoints = run_dir / f"round_{expired_round:03d}" / "training" / "checkpoints"
        shutil.rmtree(expired_checkpoints)
        report[expired_round]["checkpoint_retained"] = False
    (run_dir / "rounds.json").write_text(json.dumps(report, indent=2) + "\n")


def run_dagger(cfg, run_dir: Path) -> None:
    validate_config(cfg)
    cfg.collection.policy_name = str(Path(checkpoints.resolve_joint_checkpoint(cfg.collection.policy_name)).resolve())
    run_dir = run_dir.resolve()
    round_datasets = []
    report = []
    for round_index in range(cfg.vla_rounds + 1):
        round_cfg = build_round_config(cfg, run_dir, round_index)
        round_dir = Path(round_cfg.round_dir)
        round_dir.mkdir()
        config_path = round_dir / "round.yaml"
        updates = round_cfg.end_step - round_cfg.start_step
        print(f"Round {round_index}: collect {round_cfg.num_episodes} episodes, then train {updates} updates.", flush=True)
        run_stage("collect", round_cfg, config_path)
        round_cfg.dataset_root = str(prepare_dataset(round_cfg, round_datasets))
        run_stage("train", round_cfg, config_path)
        record_completed_round(round_cfg, run_dir, report)


def main() -> None:
    cfg = OmegaConf.load(sys.argv[2])
    stage = sys.argv[1]
    stages = {"collect": collect_round, "train": train_round}
    stages[stage](cfg)


if __name__ == "__main__":
    main()
