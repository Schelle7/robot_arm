from contextlib import ExitStack

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

class CameraCapture:
    def __init__(self, cfg):
        self.cameras = {}
        self.timeout_seconds = cfg.read_timeout_seconds
        with ExitStack() as cleanup:
            for name, settings in cfg.cameras.items():
                camera = OpenCVCamera(
                    OpenCVCameraConfig(
                        index_or_path=str(settings.device),
                        width=settings.width,
                        height=settings.height,
                        fps=cfg.fps,
                        fourcc=cfg.fourcc,
                        color_mode=cfg.color_mode,
                        rotation=cfg.rotation,
                        warmup_s=cfg.warmup_seconds,
                        backend=cfg.backend,
                    )
                )
                camera.connect()
                cleanup.callback(camera.disconnect)
                self.cameras[name] = camera
            self.cleanup = cleanup.pop_all()

    def read(self):
        return {name: camera.async_read(timeout_ms=self.timeout_seconds * 1000) for name, camera in self.cameras.items()}

    def close(self):
        self.cleanup.close()
