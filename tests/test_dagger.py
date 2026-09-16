import json
from pathlib import Path

import pytest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from robot_arm.training import dagger
from robot_arm.training import vla_bc


def test_rounds_collect_then_train_and_continue_the_previous_checkpoint(tmp_path, monkeypatch):
    from robot_arm.policies import joint

    cfg = OmegaConf.create({
        "environment": {}, "seed": 42, "vla_rounds": 2,
        "initial_episodes": 3, "episodes_per_round": 2,
        "initial_training_steps": 5, "training_steps_per_round": 4,
        "keep_last_checkpoints": 2,
        "data_root": str(tmp_path / "local-data"),
        "collection": {"backend": "sim", "policy_name": "latest"},
    })
    OmegaConf.set_struct(cfg, True)
    monkeypatch.setattr(joint, "resolve_low_level_checkpoint", lambda name: str(tmp_path / "low.pkl"))
    monkeypatch.setattr(dagger, "__file__", str(tmp_path / "src" / "robot_arm" / "training" / "dagger.py"))
    events = []

    def run_stage(stage, config, path):
        events.append((stage, OmegaConf.to_container(config, resolve=True)))
        if stage == "train":
            model = Path(config.training_dir) / "checkpoints" / "last" / "pretrained_model"
            model.mkdir(parents=True)
            (model / "model.safetensors").touch()

    datasets_seen = []

    def prepare_dataset(config, datasets):
        dataset = Path(config.data_dir) / "dataset"
        datasets.append(dataset)
        datasets_seen.append(list(datasets))
        return dataset

    monkeypatch.setattr(dagger, "run_stage", run_stage)
    monkeypatch.setattr(dagger, "prepare_dataset", prepare_dataset)
    dagger.run_dagger(cfg, tmp_path)

    assert [stage for stage, _ in events] == ["collect", "train"] * 3
    collections = [config for stage, config in events if stage == "collect"]
    assert [item["num_episodes"] for item in collections] == [3, 2, 2]
    assert [item["start_step"] for item in collections] == [0, 5, 9]
    assert [item["end_step"] for item in collections] == [5, 9, 13]
    assert all(item["total_steps"] == 13 for item in collections)
    assert collections[0]["collection_dir"] == str(tmp_path / "local-data/round_000/collection")
    assert collections[0]["training_dir"] == str(tmp_path / "round_000/training")
    assert collections[1]["previous_checkpoint"] == str(tmp_path / "round_000/training/checkpoints/last")
    assert collections[2]["previous_checkpoint"] == str(tmp_path / "round_001/training/checkpoints/last")
    assert [len(datasets) for datasets in datasets_seen] == [1, 2, 3]
    assert not (tmp_path / "round_000/training/checkpoints").exists()
    assert (tmp_path / "round_001/training/checkpoints/last/pretrained_model/model.safetensors").is_file()
    assert (tmp_path / "round_002/training/checkpoints/last/pretrained_model/model.safetensors").is_file()
    report = json.loads((tmp_path / "rounds.json").read_text())
    assert [entry["checkpoint_retained"] for entry in report] == [False, True, True]


def test_missing_new_checkpoint_preserves_previous_checkpoints(tmp_path):
    previous = tmp_path / "round_000/training/checkpoints/00000005"
    previous.mkdir(parents=True)
    (previous / "model.safetensors").touch()
    cfg = OmegaConf.create({
        "training_dir": str(tmp_path / "round_002/training"),
        "round_index": 2, "keep_last_checkpoints": 2,
    })
    with pytest.raises(FileNotFoundError):
        dagger.record_completed_round(cfg, tmp_path, [])
    assert (previous / "model.safetensors").is_file()


def test_merge_preserves_video_file_limit_and_uses_all_rounds(tmp_path, monkeypatch):
    from lerobot.datasets import aggregate
    from robot_arm.data import lerobot_converter

    calls = []
    monkeypatch.setattr(lerobot_converter, "convert_to_lerobot", lambda **kwargs: calls.append(("convert", kwargs)))
    monkeypatch.setattr(aggregate, "aggregate_datasets", lambda **kwargs: calls.append(("merge", kwargs)))
    cfg = OmegaConf.create({
        "round_index": 1, "data_dir": str(tmp_path / "local-data/round_001"),
        "collection_dir": str(tmp_path / "collection"),
        "collection": {"control": {"frequencies": {"cartesian": 5}}},
        "dataset": {"video_files_size_in_mb": 0.000001, "data_files_size_in_mb": 100, "chunk_size": 1000},
    })
    datasets = [tmp_path / "round_000/dataset_000"]
    result = dagger.prepare_dataset(cfg, datasets)
    assert result == tmp_path / "local-data/round_001/combined_dataset"
    assert calls[0][1]["target_dir"] == str(tmp_path / "local-data/round_001/dataset_001")
    assert calls[1][1]["roots"] == datasets
    assert len(datasets) == 2
    assert calls[1][1]["video_files_size_in_mb"] == 0.000001


def test_later_round_loads_processors_without_replacing_statistics(monkeypatch):
    calls = []
    monkeypatch.setattr(vla_bc, "make_pre_post_processors", lambda **kwargs: calls.append(kwargs))
    cfg = OmegaConf.create({
        "pretrained_path": "/checkpoint/pretrained_model", "device": "cpu",
        "input_features": {}, "output_features": {}, "normalization_mapping": {},
    })
    vla_bc.round_processors(cfg, {"new_statistics": 100}, False)
    assert calls == [{"policy_cfg": cfg, "pretrained_path": cfg.pretrained_path}]


def test_vla_round_loads_one_policy_for_all_episodes(tmp_path, monkeypatch):
    from robot_arm.data import collection
    from robot_arm import rollout_config

    cfg = OmegaConf.create({
        "seed": 42, "round_index": 1, "collection_dir": str(tmp_path),
        "collection": {}, "previous_checkpoint": str(tmp_path / "previous"), "num_episodes": 10,
    })
    policies = []
    calls = []

    def make_vla(path):
        policies.append(path)
        return "vla"

    def scripted_policy(config):
        raise AssertionError("A VLA round must not execute a scripted policy.")

    monkeypatch.setattr(dagger, "VLACartesianPolicy", make_vla)
    monkeypatch.setattr(dagger, "ScriptedCartesianPolicy", scripted_policy)
    monkeypatch.setattr(rollout_config, "setup_rollout_context", lambda config, path: (config, "env", "low"))
    monkeypatch.setattr(collection, "collect_episodes", lambda *args: calls.append(args))
    dagger.collect_round(cfg)
    assert policies == [str(tmp_path / "previous/pretrained_model")]
    assert calls[0][3] == "vla"
    assert calls[0][-1] == 10


def test_dagger_config_exposes_teacher_accuracy():
    config_dir = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="train_vla_dagger")
    assert cfg.collection.backend == "sim"
    assert cfg.collection.waypoint.primitive_probabilities.pick_and_place == 1.0
    assert cfg.collection.waypoint.completion_tolerance.position_meters == 0.01
    assert cfg.collection.waypoint.completion_tolerance.primary_rotation_radians == 0.1
    assert cfg.collection.waypoint.duty_completion_tolerance == 0.1
