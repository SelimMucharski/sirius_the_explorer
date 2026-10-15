"""Depth image -> obstacle and ground points, classified against the local ground. No ROS imports."""

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
    """Classifies points by their height above the local ground.

    The ground under a point is the lowest point within one grid cell around it
    (a window of 3 x ground_cell_size). A point is an obstacle when it sticks out of
    that ground by [min_height, max_height]; otherwise, if it is close to the ground,
    it is ground (used to clear the costmaps). Slopes up to roughly
    atan(min_height / (1.5 * ground_cell_size)) are therefore drivable ground,
    independently of the distance to the robot.
    """

    min_height: float = 0.15
    max_height: float = 1.3
    min_range: float = 0.3
    max_range: float = 8.0
    stride: int = 4
    ground_cell_size: float = 0.15

    def __post_init__(self):
        if not 0.0 < self.min_height < self.max_height or self.min_range >= self.max_range:
            raise ValueError('invalid filter bounds')
        if self.stride < 1:
            raise ValueError('stride must be >= 1')
        if self.ground_cell_size <= 0.0:
            raise ValueError('ground_cell_size must be > 0')


@dataclass(frozen=True)
class OrientedBox:
    """A box in the base frame: points p with |R^T (p - center)| <= half_extents are inside."""

    rotation: np.ndarray      # (3, 3) box axes in the base frame
    center: np.ndarray        # (3,)
    half_extents: np.ndarray  # (3,)


def remove_points_in_boxes(points: np.ndarray, boxes, padding: float = 0.0) -> np.ndarray:
    """Drop points inside any of the (padded) boxes, e.g. the robot's own arm."""
    keep = np.ones(points.shape[0], dtype=bool)
    for b in boxes:
        local = (points - b.center) @ b.rotation
        keep &= ~np.all(np.abs(local) <= b.half_extents + padding, axis=1)
    return points[keep]


def depth_to_points(depth_m: np.ndarray, k: PinholeIntrinsics, stride: int) -> np.ndarray:
    """Back-project a depth image (meters, optical frame) into an (N, 3) array of points."""
    d = depth_m[::stride, ::stride]
    v, u = np.mgrid[0:depth_m.shape[0]:stride, 0:depth_m.shape[1]:stride]
    z = d.reshape(-1)
    valid = np.isfinite(z) & (z > 0.0)
    z = z[valid]
    x = (u.reshape(-1)[valid] - k.cx) * z / k.fx
    y = (v.reshape(-1)[valid] - k.cy) * z / k.fy
    return np.stack([x, y, z], axis=1)


def _local_ground(base: np.ndarray, cell: float) -> np.ndarray:
    """Lowest z in the 3x3 cell neighbourhood of every point (same order as `base`)."""
    ij = np.floor(base[:, :2] / cell).astype(np.int64)
    ij -= ij.min(axis=0) - 1                    # 1-cell border for the neighbourhood
    shape = tuple(ij.max(axis=0) + 2)
    cell_min = np.full(shape, np.inf)
    np.minimum.at(cell_min, (ij[:, 0], ij[:, 1]), base[:, 2])
    neigh_min = cell_min.copy()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                shifted = np.roll(np.roll(cell_min, di, axis=0), dj, axis=1)
                np.minimum(neigh_min, shifted, out=neigh_min)
    return neigh_min[ij[:, 0], ij[:, 1]]


def classify_points(points_optical: np.ndarray, rotation: np.ndarray, translation: np.ndarray,
                    f: ObstacleFilter, self_boxes=(), self_padding: float = 0.0):
    """Transform optical-frame points into the base frame and split them.

    Points inside `self_boxes` (the robot's own body, base frame) are dropped first.
    Returns (obstacles, ground) as float32 (N, 3) arrays in the base frame.
    """
    rng = np.linalg.norm(points_optical, axis=1)
    pts = points_optical[(rng >= f.min_range) & (rng <= f.max_range)]
    base = remove_points_in_boxes(pts @ rotation.T + translation, self_boxes, self_padding)
    if base.shape[0] == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, empty
    height = base[:, 2] - _local_ground(base, f.ground_cell_size)
    obstacle = (height >= f.min_height) & (height <= f.max_height)
    ground = height < f.min_height
    return base[obstacle].astype(np.float32), base[ground].astype(np.float32)


def filter_obstacles(points_optical: np.ndarray, rotation: np.ndarray, translation: np.ndarray,
                     f: ObstacleFilter) -> np.ndarray:
    """Obstacle points only (see classify_points)."""
    return classify_points(points_optical, rotation, translation, f)[0]
