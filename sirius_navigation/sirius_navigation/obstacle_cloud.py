"""Depth image -> obstacle and ground points, classified against the local ground. No ROS imports.

Optionally the prior terrain height map (known before the mission) is a second ground
reference: in a pile of rocks the lowest point around a rock top is another rock (the gaps
are occluded), so the local ground alone sees the pile as drivable terrain.
"""

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
class TerrainPrior:
    """Prior terrain height map in the map frame.

    heights[i, j] is the terrain at the center of cell (origin_x + (j + 0.5) res,
    origin_y + (i + 0.5) res); NaN = unknown.

    A point counts as sticking out of the terrain when it is higher than the map by more
    than `margin` + the filter's min_height. The map is aligned to the live ground points
    first (offset and tilt, see fit_ground_offset), so the robot's z and roll / pitch
    estimate need not be known (a 2D EKF reports neither).
    """

    heights: np.ndarray
    origin_x: float
    origin_y: float
    resolution: float
    margin: float = 0.05            # [m] tolerated map error above the min_height threshold
    fit_tolerance: float = 0.05     # [m] ground points higher than the fit by this are dropped
    min_fit_points: int = 30

    def __post_init__(self):
        if self.heights.ndim != 2 or self.resolution <= 0.0 or self.margin < 0.0:
            raise ValueError('invalid terrain prior')

    def height_at(self, xy: np.ndarray) -> np.ndarray:
        """Height of the cell under each (x, y) in the map frame; NaN outside the map."""
        j = np.floor((xy[:, 0] - self.origin_x) / self.resolution).astype(np.int64)
        i = np.floor((xy[:, 1] - self.origin_y) / self.resolution).astype(np.int64)
        rows, cols = self.heights.shape
        inside = (i >= 0) & (i < rows) & (j >= 0) & (j < cols)
        h = np.full(xy.shape[0], np.nan)
        h[inside] = self.heights[i[inside], j[inside]]
        return h


def fit_ground_offset(xy: np.ndarray, residual: np.ndarray, tolerance: float,
                      min_points: int = 30, iterations: int = 4):
    """Plane a + b x + c y through the lower envelope of `residual` (z above the prior map).

    Fitted to ground points; rock tops mistaken for ground lie above the plane and are dropped
    iteratively (asymmetric trimming). Returns the coefficients (a, b, c) or None when there
    are too few points. With little lateral spread only the offset a is fitted.
    """
    keep = np.isfinite(residual)
    coef = None
    for _ in range(iterations):
        if keep.sum() < min_points:
            return coef
        x, y, r = xy[keep, 0], xy[keep, 1], residual[keep]
        if np.ptp(x) > 0.5 and np.ptp(y) > 0.5:
            a = np.column_stack([np.ones_like(x), x, y])
            coef = np.linalg.lstsq(a, r, rcond=None)[0]
        else:
            coef = np.array([np.median(r), 0.0, 0.0])
        dev = residual - (coef[0] + coef[1] * xy[:, 0] + coef[2] * xy[:, 1])
        keep = np.isfinite(residual) & (dev < tolerance) & (dev > -0.3)
    return coef


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


def height_above_prior(base: np.ndarray, local_height: np.ndarray, min_height: float,
                       prior: TerrainPrior, base_to_map) -> np.ndarray:
    """Height of every point above the (aligned) prior terrain; NaN where unknown."""
    rot, trans = base_to_map
    in_map = base @ np.asarray(rot).T + np.asarray(trans)
    residual = in_map[:, 2] - prior.height_at(in_map[:, :2])
    candidates = np.where(local_height < min_height, residual, np.nan)
    coef = fit_ground_offset(base[:, :2], candidates, prior.fit_tolerance, prior.min_fit_points)
    if coef is None:
        return np.full(base.shape[0], np.nan)
    return residual - (coef[0] + coef[1] * base[:, 0] + coef[2] * base[:, 1])


def classify_points(points_optical: np.ndarray, rotation: np.ndarray, translation: np.ndarray,
                    f: ObstacleFilter, self_boxes=(), self_padding: float = 0.0,
                    prior: TerrainPrior = None, base_to_map=None):
    """Transform optical-frame points into the base frame and split them.

    Points inside `self_boxes` (the robot's own body, base frame) are dropped first.
    With `prior` and `base_to_map` (rotation, translation of the base frame in the map frame)
    a point's height is the larger of its height above the local ground and above the prior
    terrain minus `prior.margin`: a rock top surrounded by other rocks is still an obstacle,
    and it does not clear the costmap as ground.
    Returns (obstacles, ground) as float32 (N, 3) arrays in the base frame.
    """
    rng = np.linalg.norm(points_optical, axis=1)
    pts = points_optical[(rng >= f.min_range) & (rng <= f.max_range)]
    base = remove_points_in_boxes(pts @ rotation.T + translation, self_boxes, self_padding)
    if base.shape[0] == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, empty
    height = base[:, 2] - _local_ground(base, f.ground_cell_size)
    if prior is not None and base_to_map is not None:
        above = height_above_prior(base, height, f.min_height, prior, base_to_map) - prior.margin
        height = np.where(np.isfinite(above), np.maximum(height, above), height)
    obstacle = (height >= f.min_height) & (height <= f.max_height)
    ground = height < f.min_height
    return base[obstacle].astype(np.float32), base[ground].astype(np.float32)


def filter_obstacles(points_optical: np.ndarray, rotation: np.ndarray, translation: np.ndarray,
                     f: ObstacleFilter) -> np.ndarray:
    """Obstacle points only (see classify_points)."""
    return classify_points(points_optical, rotation, translation, f)[0]
