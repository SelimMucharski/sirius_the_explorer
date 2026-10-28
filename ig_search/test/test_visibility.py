"""M1: widoczność 2,5D (DESIGN §3.1, PDR M1)."""
import math

from conftest import default_params, flat, make_heightmap, ridge
from ig_search.domain.params import with_section
from ig_search.domain.visibility import base_rotation, Visibility
import numpy as np
import pytest


def make_vis(fn, params=None, **kw):
    p = params or default_params()
    hm = make_heightmap(fn, **kw)
    grid = hm.grid.coarsen(int(round(p.belief.cell_size / hm.grid.res)))
    return Visibility(hm, grid, p.camera, p.hfov, p.vfov), p


def xy(vis, ids):
    return vis.cx[ids], vis.cy[ids]


def test_base_rotation_flat_is_yaw():
    R = base_rotation(math.pi / 2, 0.0, 0.0)
    np.testing.assert_allclose(R[:, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(R[:, 2], [0, 0, 1], atol=1e-12)


def test_base_rotation_uphill_tilts_nose_up():
    R = base_rotation(0.0, 0.5, 0.0)            # h rośnie z x, jedziemy w +x
    assert R[2, 0] > 0                           # oś x w górę
    np.testing.assert_allclose(np.linalg.det(R), 1.0)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)


def test_flat_matches_2d_wedge():
    vis, p = make_vis(flat)
    v = vis.visible_from(5.0, 10.0, 0.0)
    x, y = xy(vis, v.ids)
    bearing = np.arctan2(y - 10.0, x - 5.0)
    assert len(v) > 100
    assert np.all(np.abs(bearing) <= p.hfov / 2 + 0.1)
    # zasięg liczony od kamery (mount_x przed base_link)
    far = np.hypot(x - 5.0 - p.camera.mount_x, y - 10.0)
    assert np.all(far <= p.camera.r_max)
    # na płaskim terenie nic nie zasłania - widać aż do r_max
    assert far.max() > p.camera.r_max - 0.3


def test_flat_vertical_fov_gives_min_range():
    vis, p = make_vis(flat)
    v = vis.visible_from(5.0, 10.0, 0.0)
    d = np.hypot(*(np.array(xy(vis, v.ids)) - np.array([[5.0], [10.0]])))
    cam_h = p.camera.mount_z - p.camera.target_height
    r_min_geom = cam_h / math.tan(p.vfov / 2) + p.camera.mount_x
    assert d.min() >= r_min_geom - 0.3


def test_cell_behind_ridge_invisible():
    vis, _ = make_vis(ridge(x0=8.0, height=1.0))
    v = vis.visible_from(5.0, 10.0, 0.0)
    x, y = xy(vis, v.ids)
    behind = (x > 8.6) & (np.abs(y - 10.0) < 1.0)
    assert not behind.any()
    assert ((x < 7.5) & (np.abs(y - 10.0) < 1.0)).any()     # przed grzbietem widać


def test_opposite_slope_visible():
    # dolina: łazik na dnie, przeciwległe zbocze rośnie przed nim
    def valley(cx, cy):
        return np.maximum(0.0, (cx - 7.0) * 0.3)
    vis, _ = make_vis(valley)
    v = vis.visible_from(5.0, 10.0, 0.0)
    x, _ = xy(vis, v.ids)
    assert (x > 9.0).sum() > 20


def test_steep_uphill_shortens_range():
    def wall(cx, cy):
        return np.maximum(0.0, (cx - 6.0) * 2.0)            # stroma ściana od x = 6
    vis_flat, _ = make_vis(flat)
    vis_wall, _ = make_vis(wall)
    reach_flat = vis_flat.cx[vis_flat.visible_from(4.0, 10.0, 0.0).ids].max()
    reach_wall = vis_wall.cx[vis_wall.visible_from(4.0, 10.0, 0.0).ids].max()
    assert reach_wall < reach_flat - 1.0


def test_mount_pitch_down_brings_near_cells():
    p = with_section(default_params(), 'camera', mount_pitch_deg=20.0)
    vis_down, _ = make_vis(flat, p)
    vis_lvl, _ = make_vis(flat)
    d = lambda vis: np.hypot(vis.cx - 5.0, vis.cy - 10.0)  # noqa: E731
    near_down = d(vis_down)[vis_down.visible_from(5.0, 10.0, 0.0).ids].min()
    near_lvl = d(vis_lvl)[vis_lvl.visible_from(5.0, 10.0, 0.0).ids].min()
    assert near_down < near_lvl


def test_cells_near_ignores_occlusion():
    vis, _ = make_vis(ridge(x0=8.0, height=1.0))
    pose = vis.camera_pose(5.0, 10.0, 0.0)
    near = vis.cells_near(pose, np.array([[9.5, 10.0]]), 0.3)
    assert len(near) > 0
    assert np.all(np.hypot(vis.cx[near.ids] - 9.5, vis.cy[near.ids] - 10.0) <= 0.3)


@pytest.mark.parametrize('psi', [0.0, math.pi / 2, math.pi, -2.3])
def test_alpha_within_fov(psi):
    vis, p = make_vis(flat)
    v = vis.visible_from(10.0, 10.0, psi)
    half_diag = math.atan(math.hypot(math.tan(p.hfov / 2), math.tan(p.vfov / 2)))
    assert np.all(v.alpha <= half_diag + 1e-6)


def test_frame_margin_excludes_cells_near_edge():
    """
    Komórki przy krawędzi kadru nie są widoczne (margines frame_margin_deg).

    Z bliska próbka przy dolnej krawędzi kadru nie jest wykrywana; gdyby liczyła się jako
    widoczna, pusta klatka zbijałaby belief prawdziwej próbki tuż przed weryfikacją.
    """
    p = default_params()
    vis0, _ = make_vis(flat, with_section(p, 'camera', frame_margin_deg=0.0))
    vis5, _ = make_vis(flat, p)
    v0 = set(vis0.visible_from(5.0, 10.0, 0.0).ids.tolist())
    v5 = vis5.visible_from(5.0, 10.0, 0.0)
    assert set(v5.ids.tolist()) < v0
    pose = vis5.camera_pose(5.0, 10.0, 0.0)
    _, _, vc = vis5._geometry(pose, v5.ids)
    down = np.degrees(np.arctan2(np.abs(vc[:, 2]), vc[:, 0]))
    side = np.degrees(np.arctan2(np.abs(vc[:, 1]), vc[:, 0]))
    assert down.max() <= math.degrees(p.vfov) / 2 - p.camera.frame_margin_deg + 1e-6
    assert side.max() <= math.degrees(p.hfov) / 2 - p.camera.frame_margin_deg + 1e-6
    # pas przy dolnej krawędzi: najbliższe komórki (koniec dojazdu do próbki) wypadają
    near = min(np.hypot(vis0.cx[i] - 5.0, vis0.cy[i] - 10.0) for i in v0)
    near5 = np.hypot(vis5.cx[v5.ids] - 5.0, vis5.cy[v5.ids] - 10.0).min()
    assert near5 > near + 0.1
