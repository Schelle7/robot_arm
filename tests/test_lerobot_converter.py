import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from robot_arm.data import lerobot_converter


def test_conversion_uses_teacher_labels_even_when_learner_never_lifts(tmp_path, monkeypatch):
    episode = tmp_path / "recordings" / "episode_0000"
    episode.mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(episode / "camera.jpg")
    teacher_action = np.full((1, 7), 0.1, dtype=np.float32)
    np.savez(
        episode / "episode.npz",
        step=[0, 1],
        primitive_index=[0, 0],
        primitive_prompt=["close gripper"],
        cartesian_action=np.full((1, 7), -0.2),
        completes_active_primitive=[False],
        teacher_cartesian_action=teacher_action,
        teacher_completes_active_primitive=[True],
        teacher_completion_score=[0.75],
        external_camera_image_path=["camera.jpg", "camera.jpg"],
        wrist_camera_image_path=["camera.jpg", "camera.jpg"],
    )
    frames = []
    saved_episodes = []
    finalized = []
    dataset = SimpleNamespace(
        meta=SimpleNamespace(update_chunk_settings=lambda **kwargs: None),
        add_frame=frames.append,
        save_episode=lambda: saved_episodes.append(True),
        finalize=lambda: finalized.append(True),
    )
    monkeypatch.setattr(lerobot_converter, "LeRobotDataset", SimpleNamespace(create=lambda **kwargs: dataset))
    monkeypatch.setattr(lerobot_converter, "_validate_and_load_configs", lambda episodes, fps: {})
    monkeypatch.setattr(lerobot_converter, "_build_dataset_features", lambda cfg: {})
    monkeypatch.setattr(lerobot_converter, "_reconstruct_vla_input_state", lambda data, index: torch.zeros(16))

    target = tmp_path / "converted"
    lerobot_converter.convert_to_lerobot(str(episode.parent), str(target), 5, 1.0)

    np.testing.assert_array_equal(frames[0]["action"].numpy()[:7], teacher_action[0])
    assert frames[0]["action"].numpy()[7] == 0.75
    assert saved_episodes == [True]
    assert finalized == [True]
    report = json.loads((tmp_path / "converted.conversion_report.json").read_text())
    assert report["episode_count"] == 1
    assert report["conversion_complete"] is True
    assert report["episodes"][0]["conversion_status"] == "converted"


def test_conversion_requires_teacher_labels():
    with pytest.raises(KeyError, match="teacher_cartesian_action"):
        lerobot_converter.teacher_labels({"step": [0, 1], "cartesian_action": np.zeros((1, 7))})


def test_conversion_rejects_incomplete_teacher_labels():
    with pytest.raises(AssertionError, match="Every transition requires a teacher action"):
        lerobot_converter.teacher_labels({
            "step": [0, 1, 2],
            "teacher_cartesian_action": np.zeros((1, 7)),
            "teacher_completion_score": [0.25],
        })
