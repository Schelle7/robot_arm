import numpy as np
from typing import Dict
from robot_arm.arm import Arm

class SafeArmWrapper(Arm):
    """
    Wraps an Arm interface to enforce safety bounds on commanded actions.
    If an action violates bounds, it is clipped before being sent to hardware.
    Returns the actually applied (safe) action dictionary from `write_goal`.
    """

    def __init__(self, backend_arm: Arm, min_pos: float, max_pos: float):
        self.backend_arm = backend_arm
        self.min_pos = min_pos
        self.max_pos = max_pos

    def get_tcp(self) -> np.ndarray:
        return self.backend_arm.get_tcp()

    def read_state(self) -> Dict[str, Dict[str, float]]:
        return self.backend_arm.read_state()

    def write_goal(self, positions: Dict[str, float]) -> Dict[str, float]:
        safe_positions = {}
        for motor, pos in positions.items():
            # Clip position to absolute safety bounds
            safe_pos = max(self.min_pos, min(self.max_pos, float(pos)))
            safe_positions[motor] = safe_pos

        # Forward safely clamped commands down the chain
        self.backend_arm.write_goal(safe_positions)

        return safe_positions

    # Proxy all other attribute accesses to the inner backend
    def __getattr__(self, name):
        return getattr(self.backend_arm, name)
