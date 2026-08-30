import abc
from typing import Dict
import numpy as np
from robot_arm.pose import Pose


class Arm(abc.ABC):
    """
    Unified interface for hardware and simulation backends.
    """

    @abc.abstractmethod
    def read_state(self) -> Dict[str, Dict[str, float]]:
        """
        Returns a dictionary of registers to motor name to value.
        e.g., {'Present_Position': {'shoulder_pan': 0.0, ...}, ...}
        """
        pass

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

    def gravity_compensation_duty(self) -> np.ndarray:
        """
        Returns the duty each joint needs to hold its current configuration, one signed fraction of
        full output per motor. It is a model prediction rather than a reading, so it knows nothing
        about a payload the gripper is carrying.
        """
        raise NotImplementedError("Arm backend does not model gravity compensation.")

    def get_tcp_axes(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Arm backend does not expose TCP axes.")

    @abc.abstractmethod
    def disconnect(self):
        """
        Emergency power cutoff or safe shutdown routine.
        """
        pass

    @abc.abstractmethod
    def write_duty(self, duties: Dict[str, float]) -> None:
        """
        Send open loop PWM duties to the arm without letting any time pass.
        `duties` maps motor name to a signed fraction of full output, -1.0 to 1.0.
        """
        pass

    @abc.abstractmethod
    def advance_control_step(self) -> None:
        """
        Lets exactly one low-level control period elapse against the duty last written. Simulation
        advances physics, hardware waits, so callers do not need to know which backend they hold.
        """
        pass
