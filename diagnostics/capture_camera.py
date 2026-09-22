from contextlib import ExitStack
from pathlib import Path

import cv2
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from robot_arm.arms.camera_capture import CameraCapture


@hydra.main(version_base=None, config_path="../conf", config_name="test_camera")
def main(cfg: DictConfig):
    assert cfg.frames > 0
    assert cfg.camera.color_mode == "rgb"
    camera_cfg = OmegaConf.create(OmegaConf.to_container(cfg.camera, resolve=True))
    camera_cfg.cameras = {cfg.camera_name: cfg.camera.cameras[cfg.camera_name]}
    output = Path(HydraConfig.get().runtime.output_dir)

    with ExitStack() as cleanup:
        capture = CameraCapture(camera_cfg)
        cleanup.callback(capture.close)
        frame = capture.read()[cfg.camera_name]
        Image.fromarray(frame).save(output / "frame.png")
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(str(output / "capture.avi"), cv2.VideoWriter_fourcc(*cfg.video_codec), cfg.camera.fps, (width, height))
        cleanup.callback(writer.release)
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {output / 'capture.avi'}")
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        for _ in range(cfg.frames - 1):
            frame = capture.read()[cfg.camera_name]
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    print(f"Saved {cfg.frames} frames from {cfg.camera_name} ({width}x{height}) to {output}")


if __name__ == "__main__":
    main()
