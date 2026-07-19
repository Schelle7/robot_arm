from abc import ABC, abstractmethod
import numpy as np


class LowLevelController(ABC):
    """
    Base interface for mapping a high-level action to a sequence of low-level actions.
    """

    @abstractmethod
    def generate_commands(
        self, start_pos: np.ndarray, target_pos: np.ndarray
    ) -> np.ndarray:
        pass


class LinearInterpolatorController(LowLevelController):
    """
    Linearly interpolates between the starting pose and the target position over a given number of frames.
    Behaves as a placeholder for a real low-level controller running at low_level_hz.
    """

    def __init__(self, steps_per_action: int):
        self.steps_per_action = steps_per_action

    def generate_commands(
        self, start_pos: np.ndarray, target_pos: np.ndarray
    ) -> np.ndarray:
        """
        Generates interpolated targets from start_pos to target_pos over exactly `steps_per_action` frames.
        """
        return np.linspace(start_pos, target_pos, self.steps_per_action)
