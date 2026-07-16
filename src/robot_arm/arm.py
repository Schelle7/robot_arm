import abc
from typing import Dict


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
    def write_goal(self, positions: Dict[str, float]) -> None:
        """
        Send goal positions to the arm.
        `positions` maps motor name to target position.
        """
        pass
