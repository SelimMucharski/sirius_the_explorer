"""Probe detection value objects. Pure Python + numpy, no ROS dependencies."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Keypoints along the probe axis, distance from the flat face of the O30 mm head [m]:
# tip, cone/rod, rod/head, face.
KEYPOINT_AXIS_Z = np.array([-0.200, -0.160, -0.050, 0.0])
PROBE_LENGTH = 0.200
PROBE_DIAMETER = 0.030


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera (rectified / undistorted image)."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_k(cls, k, width: int, height: int) -> 'CameraIntrinsics':
        k = np.asarray(k, float).reshape(3, 3)
        return cls(k[0, 0], k[1, 1], k[0, 2], k[1, 2], int(width), int(height))

    def backproject(self, uv: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Image points (N x 2) at depths z (N,) -> points in the optical frame (N x 3)."""
        return np.stack([(uv[:, 0] - self.cx) * z / self.fx,
                         (uv[:, 1] - self.cy) * z / self.fy, z], 1)

    def project(self, p: np.ndarray) -> np.ndarray:
        return np.stack([self.fx * p[:, 0] / p[:, 2] + self.cx,
                         self.fy * p[:, 1] / p[:, 2] + self.cy], 1)


@dataclass(frozen=True)
class ProbeKeypoints:
    """One detection of the pose3d model: 4 x (u, v, confidence, log depth) and the box."""

    keypoints: np.ndarray            # (4, 4)
    box_xyxy: Tuple[float, float, float, float]
    score: float


@dataclass(frozen=True)
class ProbePose:
    """
    Probe pose in the camera optical frame (x right, y down, z forward) [m].

    The probe is rotationally symmetric, so the centre and the axis direction (tip -> face)
    are its full pose.
    """

    centre: np.ndarray               # middle of the probe (origin of the Gazebo probe model)
    face: np.ndarray                 # centre of the flat face of the head
    axis: np.ndarray                 # unit vector tip -> face
    spread: float                    # mean distance of the 3D keypoints from the fitted axis [m]
    score: float

    def quaternion(self) -> Tuple[float, float, float, float]:
        """
        Return the orientation as a quaternion, ordered x, y, z, w.

        It is the shortest rotation taking +z to the axis, i.e. the probe model frame, whose
        z axis points tip -> face.
        """
        z = np.array([0.0, 0.0, 1.0])
        a = self.axis / np.linalg.norm(self.axis)
        c = float(z @ a)
        if c < -1 + 1e-9:
            return (1.0, 0.0, 0.0, 0.0)
        v = np.cross(z, a)
        w = 1.0 + c
        q = np.array([v[0], v[1], v[2], w])
        q /= np.linalg.norm(q)
        return tuple(float(x) for x in q)
