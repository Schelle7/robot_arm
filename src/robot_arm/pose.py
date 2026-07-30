import numpy as np
from scipy.spatial.transform import Rotation as R

class Pose:
    """
    Encapsulates 3D position and rotation state. 
    Provides bridges between Euler, Quaternions, 3x3 Matrices, 
    and Neural Network-friendly 6D continuous representations.
    """
    def __init__(self, position: np.ndarray, rotation: R):
        self.position = np.array(position, dtype=np.float32)
        self.rotation = rotation

    @classmethod
    def from_euler(cls, position: np.ndarray, angles: np.ndarray, seq: str = "xyz", degrees: bool = False) -> "Pose":
        return cls(position, R.from_euler(seq, angles, degrees=degrees))

    @classmethod
    def from_quat(cls, position: np.ndarray, quat: np.ndarray) -> "Pose":
        # SciPy expects scalar-last quaternions: (x, y, z, w)
        return cls(position, R.from_quat(quat))

    @classmethod
    def from_matrix(cls, position: np.ndarray, matrix: np.ndarray) -> "Pose":
        return cls(position, R.from_matrix(matrix))

    @classmethod
    def from_6d(cls, position: np.ndarray, rep_6d: np.ndarray) -> "Pose":
        """
        Reconstructs a rotation from the 6D continuous representation.
        Uses Gram-Schmidt orthogonalization to build the 3x3 matrix.
        """
        rep_6d = np.array(rep_6d, dtype=np.float32)
        a1 = rep_6d[:3]
        a2 = rep_6d[3:]
        
        # Gram-Schmidt orthogonalization
        v1 = a1 / np.linalg.norm(a1)
        v2 = a2 - np.dot(v1, a2) * v1
        v2 = v2 / np.linalg.norm(v2)
        v3 = np.cross(v1, v2)
        
        matrix = np.column_stack((v1, v2, v3))
        return cls(position, R.from_matrix(matrix))

    def as_euler(self, seq: str = "xyz", degrees: bool = False) -> np.ndarray:
        return self.rotation.as_euler(seq, degrees=degrees).astype(np.float32)

    def as_quat(self) -> np.ndarray:
        """Returns quaternion in scalar-last format: (x, y, z, w)"""
        return self.rotation.as_quat().astype(np.float32)

    def as_matrix(self) -> np.ndarray:
        return self.rotation.as_matrix().astype(np.float32)

    def as_6d(self) -> np.ndarray:
        """Returns the first two column vectors of the rotation matrix concatenated."""
        matrix = self.as_matrix()
        return np.concatenate((matrix[:, 0], matrix[:, 1])).astype(np.float32)
        
    def angular_distance(self, other: "Pose") -> float:
        """Returns the absolute angular difference between two orientations in radians."""
        diff = self.rotation * other.rotation.inv()
        return diff.magnitude()
        
    def positional_distance(self, other: "Pose") -> float:
        """Returns the Euclidean distance between two positions."""
        return np.linalg.norm(self.position - other.position)
