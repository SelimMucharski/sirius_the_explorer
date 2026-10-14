"""Depth image -> obstacle points, filtered by height relative to the robot. No ROS imports."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PinholeIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class ObstacleFilter:
    """Keeps points that stick out of a traversable slope in front of the robot.

    A point at horizontal distance r from the robot is an obstacle when its height above
    base_link is within [min_height + max_slope * r, max_height + max_slope * r], i.e. a
    uniform incline up to atan(max_slope) is treated as drivable ground.
    """

    min_height: float = 0.15
    max_height: float = 1.3
    min_range: float = 0.3
    max_range: float = 8.0
    stride: int = 4
    max_slope: float = 0.35

    def __post_init__(self):
        if self.min_height >= self.max_height or self.min_range >= self.max_range:
            raise ValueError('invalid filter bounds')
        if self.stride < 1:
            raise ValueError('stride must be >= 1')
        if self.max_slope < 0.0:
            raise ValueError('max_slope must be >= 0')


def depth_to_points(depth_m: np.ndarray, k: PinholeIntrinsics, stride: int) -> np.ndarray:
    """Back-project a depth image (meters, optical frame) into an (N, 3) array of points."""
    d = depth_m[::stride, ::stride]
    v, u = np.mgrid[0:depth_m.shape[0]:stride, 0:depth_m.shape[1]:stride]
    z = d.reshape(-1)
    x = (u.reshape(-1) - k.cx) * z / k.fx
    y = (v.reshape(-1) - k.cy) * z / k.fy
    pts = np.stack([x, y, z], axis=1)
    return pts[np.isfinite(z) & (z > 0.0)]


def filter_obstacles(points_optical: np.ndarray, rotation: np.ndarray, translation: np.ndarray,
                     f: ObstacleFilter) -> np.ndarray:
    """Transform optical-frame points into the base frame and keep obstacle-height points."""
    rng = np.linalg.norm(points_optical, axis=1)
    pts = points_optical[(rng >= f.min_range) & (rng <= f.max_range)]
    base = pts @ rotation.T + translation
    slope_offset = f.max_slope * np.hypot(base[:, 0], base[:, 1])
    keep = (base[:, 2] >= f.min_height + slope_offset) & (base[:, 2] <= f.max_height + slope_offset)
    return base[keep].astype(np.float32)
