import numpy as np
from scipy.spatial.transform import Rotation as R


class Pose:
    """
    Encapsulates 3D position, rotation state, and gripper status.
    Provides bridges between Euler, Quaternions, 3x3 Matrices,
    and Neural Network-friendly 6D continuous representations.
    """

    def __init__(self, position: np.ndarray, rotation: R, gripper: float):
        self.position = np.array(position, dtype=np.float32)
        self.rotation = rotation
        self.gripper = float(gripper)

    @classmethod
    def from_euler(
        cls,
        position: np.ndarray,
        angles: np.ndarray,
        gripper: float,
        seq: str,
        degrees: bool,
    ) -> "Pose":
        assert seq == "XYZ"
        return cls(position, R.from_euler(seq, angles, degrees=degrees), gripper)

    @classmethod
    def from_mujoco_quat(
        cls, position: np.ndarray, quat: np.ndarray, gripper: float
    ) -> "Pose":
        # MuJoCo uses scalar-first (w, x, y, z), but SciPy expects scalar-last (x, y, z, w)
        scipy_quat = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)
        return cls(position, R.from_quat(scipy_quat), gripper)

    @classmethod
    def from_scipy_quat(
        cls, position: np.ndarray, quat: np.ndarray, gripper: float
    ) -> "Pose":
        # SciPy format: (x, y, z, w)
        return cls(position, R.from_quat(quat), gripper)

    @classmethod
    def from_matrix(
        cls, position: np.ndarray, matrix: np.ndarray, gripper: float
    ) -> "Pose":
        return cls(position, R.from_matrix(matrix), gripper)

    @classmethod
    def from_rot_6d(
        cls, position: np.ndarray, rep_6d: np.ndarray, gripper: float
    ) -> "Pose":
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
        return cls(position, R.from_matrix(matrix), gripper)

    @classmethod
    def from_10d(cls, state_10d: np.ndarray) -> "Pose":
        """Reconstructs a Pose from the full 10D state array."""
        state_10d = np.array(state_10d, dtype=np.float32)
        return cls.from_rot_6d(state_10d[:3], state_10d[3:9], state_10d[9])

    def as_euler(self, seq: str, degrees: bool) -> np.ndarray:
        return self.rotation.as_euler(seq, degrees=degrees).astype(np.float32)

    def as_mujoco_quat(self) -> np.ndarray:
        """Returns quaternion in scalar-first format (w, x, y, z) for MuJoCo compatibility."""
        quat = self.rotation.as_quat()  # SciPy format: (x, y, z, w)
        return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32)

    def as_scipy_quat(self) -> np.ndarray:
        """Returns quaternion in scalar-last format (x, y, z, w)."""
        return self.rotation.as_quat().astype(np.float32)

    def as_matrix(self) -> np.ndarray:
        return self.rotation.as_matrix().astype(np.float32)

    def as_rot_6d(self) -> np.ndarray:
        """Returns the first two column vectors of the rotation matrix concatenated."""
        matrix = self.as_matrix()
        return np.concatenate((matrix[:, 0], matrix[:, 1])).astype(np.float32)

    def as_10d(self) -> np.ndarray:
        """Returns the full 10D state array: [x, y, z, r1, r2, r3, r4, r5, r6, gripper]"""
        return np.concatenate([self.position, self.as_rot_6d(), [self.gripper]]).astype(
            np.float32
        )

    def angular_distance(self, other: "Pose") -> float:
        """Returns the absolute angular difference between two orientations in radians."""
        diff = self.rotation * other.rotation.inv()
        return diff.magnitude()

    def positional_distance(self, other: "Pose") -> float:
        """Returns the Euclidean distance between two positions."""
        return np.linalg.norm(self.position - other.position)
