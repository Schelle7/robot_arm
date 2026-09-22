from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from omegaconf import OmegaConf

from robot_arm.arms.camera_capture import CameraCapture
from robot_arm.robot_schema import CAMERA_NAMES


@pytest.fixture
def camera_cfg():
    return OmegaConf.load(Path(__file__).resolve().parents[1] / "conf/camera/default.yaml")


def test_capture_configures_named_cameras_and_returns_frames(camera_cfg, monkeypatch):
    cameras = [Mock(), Mock()]
    frames = [np.full((480, 640, 3), value, dtype=np.uint8) for value in (20, 40)]
    for camera, frame in zip(cameras, frames, strict=True):
        camera.async_read.return_value = frame
    factory = Mock(side_effect=cameras)
    monkeypatch.setattr("robot_arm.arms.camera_capture.OpenCVCamera", factory)

    capture = CameraCapture(camera_cfg)
    images = capture.read()
    assert tuple(images) == CAMERA_NAMES
    for index, name in enumerate(CAMERA_NAMES):
        config = factory.call_args_list[index].args[0]
        assert config.index_or_path == camera_cfg.cameras[name].device
        assert (config.width, config.height, config.fps) == (640, 480, 30)
        assert config.fourcc == "MJPG"
        assert config.color_mode == "rgb"
        assert images[name] is frames[index]
        cameras[index].async_read.assert_called_once_with(timeout_ms=200)

    capture.close()
    for camera in cameras:
        camera.disconnect.assert_called_once_with()


def test_second_camera_failure_closes_first(camera_cfg, monkeypatch):
    cameras = [Mock(), Mock()]
    cameras[1].connect.side_effect = ConnectionError("Camera unavailable")
    monkeypatch.setattr("robot_arm.arms.camera_capture.OpenCVCamera", Mock(side_effect=cameras))
    with pytest.raises(ConnectionError, match="Camera unavailable"):
        CameraCapture(camera_cfg)
    cameras[0].disconnect.assert_called_once_with()


def test_capture_can_open_only_wrist_camera(camera_cfg, monkeypatch):
    camera_cfg.cameras = {"wrist_camera": camera_cfg.cameras.wrist_camera}
    camera = Mock()
    factory = Mock(return_value=camera)
    monkeypatch.setattr("robot_arm.arms.camera_capture.OpenCVCamera", factory)
    capture = CameraCapture(camera_cfg)
    assert tuple(capture.read()) == ("wrist_camera",)
    factory.assert_called_once()
    assert factory.call_args.args[0].index_or_path == camera_cfg.cameras.wrist_camera.device
    capture.close()
    camera.disconnect.assert_called_once_with()
