import numpy as np
import pytest

from sirius_navigation.obstacle_cloud import (
    ObstacleFilter,
    OrientedBox,
    PinholeIntrinsics,
    TerrainPrior,
    classify_points,
    depth_to_points,
    filter_obstacles,
    fit_ground_offset,
    remove_points_in_boxes,
)

K = PinholeIntrinsics(fx=100.0, fy=100.0, cx=2.0, cy=2.0)
# Optical frame (z forward, x right, y down) -> base (x forward, y left, z up), camera 0.5 m up.
R_OPT_TO_BASE = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
T_CAM = np.array([0.0, 0.0, 0.5])
FILTER = ObstacleFilter(min_height=0.15, max_height=1.3, stride=1, ground_cell_size=0.15)


def to_optical(base_points):
    """Inverse of the camera pose above: base-frame points -> optical-frame points."""
    return (np.asarray(base_points, dtype=float) - T_CAM) @ R_OPT_TO_BASE


def ground_patch(x0, x1, y0, y1, height=lambda x, y: 0.0 * x, step=0.05):
    xs, ys = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    return np.stack([xs.ravel(), ys.ravel(), height(xs.ravel(), ys.ravel())], axis=1)


def box(x, y, size, height, step=0.05):
    """Vertical faces of a box standing on flat ground (what a forward camera sees)."""
    zs = np.arange(0.0, height + 1e-9, step)
    ys = np.arange(y - size / 2, y + size / 2 + 1e-9, step)
    yy, zz = np.meshgrid(ys, zs)
    return np.stack([np.full(yy.size, x - size / 2), yy.ravel(), zz.ravel()], axis=1)


def test_invalid_filter():
    with pytest.raises(ValueError):
        ObstacleFilter(min_height=1.0, max_height=0.5)
    with pytest.raises(ValueError):
        ObstacleFilter(ground_cell_size=0.0)


def test_center_pixel_projects_on_axis():
    depth = np.zeros((5, 5))
    depth[2, 2] = 3.0
    pts = depth_to_points(depth, K, stride=1)
    assert pts.shape == (1, 3)
    assert pts[0] == pytest.approx([0.0, 0.0, 3.0])


def test_invalid_depth_is_dropped():
    depth = np.full((5, 5), np.nan)
    depth[0, 0] = np.inf
    depth[1, 1] = 2.0
    assert depth_to_points(depth, K, stride=1).shape[0] == 1


def test_flat_ground_is_ground_not_obstacle():
    ground = ground_patch(0.5, 6.0, -2.0, 2.0)
    obstacles, free = classify_points(to_optical(ground), R_OPT_TO_BASE, T_CAM, FILTER)
    assert obstacles.shape[0] == 0
    assert free.shape[0] == ground.shape[0]


def test_low_obstacle_far_away_is_detected():
    # Regression: a range-dependent slope allowance used to hide everything below ~1.5 m at 4 m.
    scene = np.vstack([ground_patch(0.5, 6.0, -1.0, 1.0), box(4.0, 0.0, 0.4, 0.3)])
    obstacles = filter_obstacles(to_optical(scene), R_OPT_TO_BASE, T_CAM, FILTER)
    assert obstacles.shape[0] > 0
    assert obstacles[:, 0].mean() == pytest.approx(3.8, abs=0.05)
    assert obstacles[:, 2].min() >= 0.15


def test_obstacle_points_are_in_base_frame():
    scene = np.vstack([ground_patch(1.0, 3.0, -0.5, 0.5), box(2.0, 0.0, 0.4, 0.5)])
    obstacles = filter_obstacles(to_optical(scene), R_OPT_TO_BASE, T_CAM, FILTER)
    assert obstacles[:, 1].min() >= -0.21 and obstacles[:, 1].max() <= 0.21
    assert obstacles[:, 2].max() == pytest.approx(0.5, abs=1e-6)


def test_uniform_incline_is_not_an_obstacle():
    # Ground rising 0.25 m per meter (14 deg) is drivable.
    incline = ground_patch(0.5, 6.0, -1.0, 1.0, height=lambda x, y: 0.25 * x)
    assert filter_obstacles(to_optical(incline), R_OPT_TO_BASE, T_CAM, FILTER).shape[0] == 0


def test_rock_on_incline_is_detected():
    incline = ground_patch(0.5, 6.0, -1.0, 1.0, height=lambda x, y: 0.25 * x)
    rock = box(3.0, 0.0, 0.3, 0.3) + np.array([0.0, 0.0, 0.25 * 2.85])
    obstacles = filter_obstacles(to_optical(np.vstack([incline, rock])), R_OPT_TO_BASE, T_CAM, FILTER)
    assert obstacles.shape[0] > 0


def test_overhang_above_max_height_is_ignored():
    branch = ground_patch(2.0, 2.5, -0.5, 0.5, height=lambda x, y: 0.0 * x + 2.0)
    scene = np.vstack([ground_patch(0.5, 4.0, -1.0, 1.0), branch])
    assert filter_obstacles(to_optical(scene), R_OPT_TO_BASE, T_CAM, FILTER).shape[0] == 0


def test_range_limits():
    scene = np.vstack([ground_patch(0.5, 12.0, -0.5, 0.5), box(10.0, 0.0, 0.4, 0.5)])
    f = ObstacleFilter(stride=1, max_range=8.0)
    assert filter_obstacles(to_optical(scene), R_OPT_TO_BASE, T_CAM, f).shape[0] == 0


def test_no_points():
    obstacles, ground = classify_points(np.zeros((0, 3)), R_OPT_TO_BASE, T_CAM, FILTER)
    assert obstacles.shape == (0, 3) and ground.shape == (0, 3)


def test_remove_points_in_rotated_box():
    yaw = np.pi / 4
    rot = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    b = OrientedBox(rot, np.array([1.0, 0.0, 0.5]), np.array([0.5, 0.05, 0.1]))
    inside_along_long_axis = np.array([1.0, 0.0, 0.5]) + rot @ np.array([0.45, 0.0, 0.0])
    outside_across = np.array([1.0, 0.0, 0.5]) + rot @ np.array([0.0, 0.2, 0.0])
    kept = remove_points_in_boxes(np.array([inside_along_long_axis, outside_across]), [b])
    assert kept.shape[0] == 1 and kept[0] == pytest.approx(outside_across)
    assert remove_points_in_boxes(np.array([outside_across]), [b], padding=0.2).shape[0] == 0


def test_own_arm_in_view_is_not_an_obstacle():
    arm = box(1.0, 0.0, 0.2, 0.6) + np.array([0.0, 0.0, 0.1])   # arm segment hanging in front
    scene = np.vstack([ground_patch(0.5, 4.0, -1.0, 1.0), arm])
    assert filter_obstacles(to_optical(scene), R_OPT_TO_BASE, T_CAM, FILTER).shape[0] > 0
    arm_box = OrientedBox(np.eye(3), np.array([0.9, 0.0, 0.4]), np.array([0.1, 0.1, 0.3]))
    obstacles, _ = classify_points(to_optical(scene), R_OPT_TO_BASE, T_CAM, FILTER, [arm_box], 0.05)
    assert obstacles.shape[0] == 0


def flat_prior(height=0.0):
    """Prior map of flat terrain at `height`, 20 x 20 m around the origin, 0.1 m cells."""
    return TerrainPrior(np.full((200, 200), height), -10.0, -10.0, 0.1)


def rock_pile(x0, x1, y0, y1, top):
    """Touching rocks seen as one block: a flat top and the front face (gaps are occluded)."""
    front = box(x0 + (y1 - y0) / 2, (y0 + y1) / 2, y1 - y0, top)     # face at x = x0
    return np.vstack([ground_patch(x0, x1, y0, y1, height=lambda x, y: 0.0 * x + top), front])


IDENTITY = (np.eye(3), np.zeros(3))


def pile_scene():
    """Ground with a 1.2 x 1.2 m pile of rocks 0.3 m high, 3 m ahead (no ground visible in it)."""
    ground = ground_patch(0.5, 6.0, -2.0, 2.0)
    outside = ~((ground[:, 0] >= 2.5) & (ground[:, 0] < 3.7) & (np.abs(ground[:, 1]) < 0.6))
    return np.vstack([ground[outside], rock_pile(2.5, 3.7, -0.6, 0.6, 0.3)])


def test_pile_top_is_ground_without_prior():
    # Regression (erc2025_real): the lowest point around a rock top is the next rock top.
    obstacles, ground = classify_points(to_optical(pile_scene()), R_OPT_TO_BASE, T_CAM, FILTER)
    on_top = (ground[:, 0] > 2.8) & (ground[:, 0] < 3.4) & (ground[:, 2] > 0.25)
    assert on_top.sum() > 0


def test_pile_is_obstacle_with_prior():
    obstacles, ground = classify_points(to_optical(pile_scene()), R_OPT_TO_BASE, T_CAM, FILTER,
                                        prior=flat_prior(), base_to_map=IDENTITY)
    on_top = (obstacles[:, 0] > 2.8) & (obstacles[:, 0] < 3.4)
    assert on_top.sum() > 100
    assert (ground[:, 2] > 0.25).sum() == 0          # rock tops do not clear the costmap


def test_prior_alignment_absorbs_unknown_height_and_tilt():
    # 2D EKF: the base frame is 0.4 m too low and 5 deg pitched with respect to the map.
    pitch = np.radians(5.0)
    rot = np.array([[np.cos(pitch), 0.0, np.sin(pitch)], [0.0, 1.0, 0.0],
                    [-np.sin(pitch), 0.0, np.cos(pitch)]])
    shifted = (rot, np.array([1.0, 2.0, -0.4]))
    ground = ground_patch(0.5, 6.0, -2.0, 2.0)
    obstacles, free = classify_points(to_optical(ground), R_OPT_TO_BASE, T_CAM, FILTER,
                                      prior=flat_prior(), base_to_map=shifted)
    assert obstacles.shape[0] == 0 and free.shape[0] == ground.shape[0]
    obstacles, _ = classify_points(to_optical(pile_scene()), R_OPT_TO_BASE, T_CAM, FILTER,
                                   prior=flat_prior(), base_to_map=shifted)
    assert ((obstacles[:, 0] > 2.8) & (obstacles[:, 0] < 3.4)).sum() > 100


def test_prior_slope_is_not_an_obstacle():
    # Terrain rising 0.25 m / m in the map too: the prior agrees with the local ground.
    xs = (np.arange(200) + 0.5) * 0.1 - 10.0
    prior = TerrainPrior(np.tile(0.25 * xs, (200, 1)), -10.0, -10.0, 0.1)
    incline = ground_patch(0.5, 6.0, -1.0, 1.0, height=lambda x, y: 0.25 * x)
    obstacles, _ = classify_points(to_optical(incline), R_OPT_TO_BASE, T_CAM, FILTER,
                                   prior=prior, base_to_map=IDENTITY)
    assert obstacles.shape[0] == 0


def test_outside_prior_falls_back_to_local_ground():
    unknown = TerrainPrior(np.full((10, 10), np.nan), -10.0, -10.0, 0.1)
    plain = classify_points(to_optical(pile_scene()), R_OPT_TO_BASE, T_CAM, FILTER)
    with_prior = classify_points(to_optical(pile_scene()), R_OPT_TO_BASE, T_CAM, FILTER,
                                 prior=unknown, base_to_map=IDENTITY)
    assert with_prior[0].shape == plain[0].shape and with_prior[1].shape == plain[1].shape


def test_fit_ground_offset_ignores_rock_tops():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-3.0, 3.0, (400, 2))
    residual = 0.2 + 0.05 * xy[:, 0] + rng.normal(0.0, 0.01, 400)
    residual[:120] += 0.3                              # 30 % of the "ground" is rock tops
    a, b, c = fit_ground_offset(xy, residual, tolerance=0.05)
    assert a == pytest.approx(0.2, abs=0.02) and b == pytest.approx(0.05, abs=0.01)
    assert c == pytest.approx(0.0, abs=0.01)
    assert fit_ground_offset(xy[:5], residual[:5], tolerance=0.05) is None


def test_invalid_prior():
    with pytest.raises(ValueError):
        TerrainPrior(np.zeros(5), 0.0, 0.0, 0.1)
    with pytest.raises(ValueError):
        TerrainPrior(np.zeros((5, 5)), 0.0, 0.0, 0.0)
