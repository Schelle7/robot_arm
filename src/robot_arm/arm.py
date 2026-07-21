import abc
from typing import Dict
import numpy as np


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

    @abc.abstractmethod
    def write_goal(self, positions: Dict[str, float]) -> None:
        """
        Send goal positions to the arm.
        `positions` maps motor name to target position.
        """
        pass
