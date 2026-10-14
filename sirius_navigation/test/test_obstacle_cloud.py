import numpy as np
import pytest

from sirius_navigation.obstacle_cloud import (
    ObstacleFilter,
    PinholeIntrinsics,
    depth_to_points,
    filter_obstacles,
)

K = PinholeIntrinsics(fx=100.0, fy=100.0, cx=2.0, cy=2.0)
# Optical frame (z forward, x right, y down) -> base (x forward, y left, z up), camera 0.5 m up.
R_OPT_TO_BASE = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
T_CAM = np.array([0.0, 0.0, 0.5])


def test_invalid_filter():
    with pytest.raises(ValueError):
        ObstacleFilter(min_height=1.0, max_height=0.5)


def test_center_pixel_projects_on_axis():
    depth = np.zeros((5, 5))
    depth[2, 2] = 3.0
    pts = depth_to_points(depth, K, stride=1)
    assert pts.shape == (1, 3)
    assert pts[0] == pytest.approx([0.0, 0.0, 3.0])


def test_height_filter_relative_to_base():
    # Point straight ahead at camera height (0.5 m above base) is kept,
    # a point at ground level (0.5 m below the camera) is dropped.
    obstacle = np.array([[0.0, 0.0, 2.0]])
    ground = np.array([[0.0, 0.5, 2.0]])
    f = ObstacleFilter(min_height=0.15, max_height=1.3, stride=1, max_slope=0.0)
    assert filter_obstacles(obstacle, R_OPT_TO_BASE, T_CAM, f).shape[0] == 1
    assert filter_obstacles(ground, R_OPT_TO_BASE, T_CAM, f).shape[0] == 0
    kept = filter_obstacles(obstacle, R_OPT_TO_BASE, T_CAM, f)[0]
    assert kept == pytest.approx([2.0, 0.0, 0.5])


def test_uniform_incline_is_not_an_obstacle():
    # Ground rising 0.25 m per meter (14 deg): 2 m ahead it is 0.5 m above base_link,
    # i.e. at camera height -> straight ahead in the optical frame.
    incline_point = np.array([[0.0, 0.0, 2.0]])
    f = ObstacleFilter(min_height=0.15, max_height=1.3, stride=1, max_slope=0.35)
    assert filter_obstacles(incline_point, R_OPT_TO_BASE, T_CAM, f).shape[0] == 0
    # A rock 1 m tall at 2 m still is.
    rock = np.array([[0.0, -0.5, 2.0]])
    assert filter_obstacles(rock, R_OPT_TO_BASE, T_CAM, f).shape[0] == 1
