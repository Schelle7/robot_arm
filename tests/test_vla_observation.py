import glob
import os

import numpy as np
import pytest

from robot_arm.policies import build_vla_observation


def load_recorded_frame():
    roots = sorted(os.path.dirname(os.path.dirname(path)) for path in glob.glob("datasets/*/meta/info.json"))
    if not roots:
        pytest.skip("No LeRobot dataset under datasets/ to compare the rollout observation against.")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(os.path.basename(roots[-1]), root=roots[-1])[0]


def test_rollout_observation_matches_the_recorded_training_frame():
    frame = load_recorded_frame()
    recorded_image = frame["observation.images.camera1"].numpy()
    recorded_state = frame["observation.state"].numpy()

    # Recover the raw frame the recorder wrote, so the comparison starts where the rollout starts.
    camera_frame = np.transpose(recorded_image * 255.0, (1, 2, 0)).round().astype(np.uint8)

    observation = build_vla_observation(camera_frame, recorded_state, frame["task"])

    np.testing.assert_allclose(observation["observation.images.camera1"], recorded_image, atol=1.0 / 255.0)
    np.testing.assert_array_equal(observation["observation.state"], recorded_state)
    assert observation["task"] == frame["task"]


def test_rollout_observation_rejects_images_that_are_already_scaled():
    with pytest.raises(AssertionError):
        build_vla_observation(np.zeros((4, 4, 3), dtype=np.float32), np.zeros(15, dtype=np.float32), "hold position")
