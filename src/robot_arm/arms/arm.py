import abc
from typing import Dict, final
import mujoco
import numpy as np
from robot_arm.geometry.pose import Pose
from robot_arm.robot_schema import MOTOR_ORDER


class Arm(abc.ABC):
    """
    Unified interface for hardware and simulation backends.
    """

    def __init_subclass__(cls):
        super().__init_subclass__()
        for name in ("get_state", "submit_duty"):
            if name in cls.__dict__:
                raise TypeError(f"{cls.__name__} cannot override final method Arm.{name}")

    def __init__(self, cfg):
        self.model = mujoco.MjModel.from_xml_path(cfg.model_path)
        self.data = mujoco.MjData(self.model)
        self.joint_indices = {name: self.model.joint(name).id for name in MOTOR_ORDER}
        self.control_step_seconds = 1.0 / cfg.control.frequencies.joint

    @property
    def joint_limits(self) -> Dict[str, tuple[float, float]]:
        return {name: tuple(map(float, self.model.jnt_range[index])) for name, index in self.joint_indices.items()}

    @final
    def get_state(self) -> Dict[str, Dict[str, float]]:
        return self.communication.get_state()

    @abc.abstractmethod
    def get_tcp(self) -> np.ndarray:
        """
        Returns the Tool Center Point (TCP) pose as a 7D array:
        [x, y, z, roll, pitch, yaw, aperture]
        Raises NotImplementedError if the backend cannot compute this.
        """
        pass

    def get_tcp_pose(self, state: Dict[str, Dict[str, float]]) -> Pose:
        """
        Returns the TCP pose for an already-read state. The state is passed in so that
        backends deriving the pose from joint readings do not add a bus round trip.
        """
        raise NotImplementedError("Arm backend does not expose a TCP pose.")

    def get_tcp_axes(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Arm backend does not expose TCP axes.")

    def physics_metrics(self) -> Dict[str, float]:
        """
        Physics parameters that vary between episodes, for logging. Empty where they cannot,
        which is every backend driving real hardware.
        """
        return {}

    @abc.abstractmethod
    def read_cameras(self) -> dict[str, np.ndarray]:
        pass

    @abc.abstractmethod
    def disconnect(self):
        """
        Emergency power cutoff or safe shutdown routine.
        """
        pass

    @final
    def submit_duty(self, duties: Dict[str, float], positions: Dict[str, float], sample_time_ns: int) -> Dict[str, float]:
        # Suppress only outward duties at a limit, so the arm can still move back.
        safe_duties = {}
        for motor, duty in duties.items():
            lower, upper = self.joint_limits[motor]
            position = positions[motor]
            duty = float(duty)
            if (position <= lower and duty < 0.0) or (position >= upper and duty > 0.0):
                duty = 0.0
            safe_duties[motor] = duty
        self.communication.submit_duty(safe_duties, sample_time_ns)
        return safe_duties

    @abc.abstractmethod
    def advance_control_step(self) -> None:
        """
        Lets exactly one joint control period elapse against the duty last written. Simulation
        advances physics, hardware waits, so callers do not need to know which backend they hold.
        """
        pass
