import glob
import os

import numpy as np
import pytest

from robot_arm.policies.cartesian import build_vla_observation
from robot_arm.robot_schema import CAMERA_NAMES


def load_recorded_frame():
    roots = sorted(os.path.dirname(os.path.dirname(path)) for path in glob.glob("datasets/*/meta/info.json"))
    if not roots:
        pytest.skip("No LeRobot dataset under datasets/ to compare the rollout observation against.")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(os.path.basename(roots[-1]), root=roots[-1])[0]


def test_rollout_observation_matches_the_recorded_training_frame():
    frame = load_recorded_frame()
    recorded_state = frame["observation.state"].numpy()
    recorded_images = {camera_name: frame[f"observation.images.{camera_name}"].numpy() for camera_name in CAMERA_NAMES}
    camera_frames = {
        camera_name: np.transpose(recorded_image * 255.0, (1, 2, 0)).round().astype(np.uint8)
        for camera_name, recorded_image in recorded_images.items()
    }

    observation = build_vla_observation(camera_frames, recorded_state, frame["task"])

    for camera_name, recorded_image in recorded_images.items():
        np.testing.assert_allclose(observation[f"observation.images.{camera_name}"], recorded_image, atol=1.0 / 255.0)
    np.testing.assert_array_equal(observation["observation.state"], recorded_state)
    assert observation["task"] == frame["task"]


def test_rollout_observation_rejects_images_that_are_already_scaled():
    images = {
        "external_camera": np.zeros((4, 4, 3), dtype=np.float32),
        "wrist_camera": np.zeros((4, 4, 3), dtype=np.uint8),
    }
    with pytest.raises(AssertionError):
        build_vla_observation(images, np.zeros(15, dtype=np.float32), "hold position")


def test_rollout_observation_requires_both_named_cameras():
    with pytest.raises(AssertionError):
        build_vla_observation(
            {"external_camera": np.zeros((4, 4, 3), dtype=np.uint8)},
            np.zeros(15, dtype=np.float32),
            "hold position",
        )
