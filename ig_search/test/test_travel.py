"""M2: czas ruchu po terenie (DESIGN §3.7, PDR M2)."""
import math

from conftest import default_params, flat, make_heightmap
from ig_search.domain.params import with_section
from ig_search.domain.travel import TravelModel
import numpy as np


def make_travel(fn, params=None):
    p = params or default_params()
    hm = make_heightmap(fn)
    grid = hm.grid.coarsen(int(round(p.belief.cell_size / hm.grid.res)))
    return TravelModel(hm, grid, p.travel), p


def cell(tr, x, y):
    return int(tr.grid.flat_index(x, y))


def test_flat_time_is_distance_over_speed():
    tr, p = make_travel(flat)
    d = tr.drive_times(3.0, 10.0)
    t = d[cell(tr, 13.0, 10.0)]
    assert abs(t - 10.0 / p.travel.v_max) < 0.5


def test_route_avoids_steep_ridge():
    def ridge_with_gap(cx, cy):
        wall = np.maximum(0.0, 1.5 * (1.0 - np.abs(cx - 10.0) / 0.5))
        return np.where(np.abs(cy - 3.0) < 1.5, 0.0, wall)       # przejście przy y = 3
    tr, p = make_travel(ridge_with_gap)
    assert not tr.traversable[cell(tr, 10.0, 10.0)]
    d = tr.drive_times(7.0, 10.0)
    t = d[cell(tr, 13.0, 10.0)]
    straight = 6.0 / p.travel.v_max
    detour = (2 * math.hypot(3.0, 7.0)) / p.travel.v_max
    assert np.isfinite(t)
    assert t > straight * 1.5
    assert abs(t - detour) < 0.35 * detour


def test_uphill_slower_than_downhill():
    def slope(cx, cy):
        return 0.15 * cx                                         # ~8.5°
    tr, _ = make_travel(slope)
    up = tr.drive_times(4.0, 10.0)[cell(tr, 14.0, 10.0)]
    down = tr.drive_times(14.0, 10.0)[cell(tr, 4.0, 10.0)]
    assert up > down


def test_too_steep_slope_unreachable():
    def steep(cx, cy):
        return np.where(cx > 10.0, (cx - 10.0) * 1.0, 0.0)        # 45° od x = 10
    tr, _ = make_travel(steep)
    d = tr.drive_times(5.0, 10.0)
    assert not np.isfinite(d[cell(tr, 15.0, 10.0)])


def test_pose_times_include_rotation():
    p = with_section(default_params(), 'travel', t_overhead=0.0)
    tr, _ = make_travel(flat, p)
    d = tr.drive_times(10.0, 10.0)
    c = cell(tr, 10.0, 10.0)
    t_same = tr.pose_times((10.0, 10.0, 0.0), [10.0], [10.0], [0.0], d, [c])[0]
    t_turn = tr.pose_times((10.0, 10.0, 0.0), [10.0], [10.0], [math.pi], d, [c])[0]
    assert abs(t_turn - t_same - math.pi / p.travel.w_max) < 1e-6


def test_waypoints_follow_detour_around_ridge():
    def ridge_with_gap(cx, cy):
        wall = np.maximum(0.0, 1.5 * (1.0 - np.abs(cx - 10.0) / 0.5))
        return np.where(np.abs(cy - 3.0) < 1.5, 0.0, wall)
    tr, _ = make_travel(ridge_with_gap)
    wps = tr.waypoints(7.0, 10.0, 13.0, 10.0, spacing=1.5)
    assert len(wps) >= 4
    assert min(y for _, y in wps) < 5.0             # trasa schodzi do przejścia przy y = 3
    for x, y in wps:
        assert tr.traversable[cell(tr, x, y)]


def test_start_on_isolated_island_routes_from_main_component():
    """
    Łazik na odciętej wysepce (np. po wyłączeniu miejsca utknięcia): czasy nie są nieskończone.

    Start trasy z najbliższej komórki dużej składowej grafu, inaczej strategia nie ma żadnej
    osiągalnej akcji, a misja wpada w pętlę natychmiast nieudanych celów.
    """
    tm = TravelModel(make_heightmap(flat), make_heightmap(flat).grid.coarsen(2),
                     default_params().travel)
    x, y = 10.0, 10.0
    ring = (np.hypot(tm.cx - x, tm.cy - y) > 0.5) & (np.hypot(tm.cx - x, tm.cy - y) < 1.0)
    tm.traversable[ring] = False                # wysepka r < 0,5 m wokół łazika
    tm._build_graph()
    t = tm.drive_times(x, y)
    assert np.isfinite(t).mean() > 0.5


def test_edge_of_heightmap_not_traversable():
    """Pas edge_margin przy krawędzi mapy wysokości nieprzejezdny (poza mapą - uskok?)."""
    p = default_params().travel
    hm = make_heightmap(flat)
    tm = TravelModel(hm, hm.grid.coarsen(2), p)
    near_edge = np.minimum.reduce([tm.cx, tm.cy, 20.0 - tm.cx, 20.0 - tm.cy]) < p.edge_margin
    assert not tm.traversable[near_edge].any()
    assert tm.traversable[~near_edge].mean() > 0.9


def test_view_poses_keep_edge_clearance():
    from conftest import ridge
    from ig_search.application.world import build_world
    from ig_search.domain.params import from_dict
    p = from_dict({'views': {'vp_step': 1.0, 'edge_clearance': 2.0}})
    w = build_world(p, make_heightmap(ridge(x0=10.0, height=1.5)))
    assert len(w.views) > 0
    assert (w.travel.clearance[w.views.cells] >= 2.0).all()
    assert np.abs(w.views.xs - 10.0).min() > 2.0           # żadnej pozy przy grzbiecie
    hm = make_heightmap(ridge(x0=10.0, height=1.5))
    w0 = build_world(from_dict({'views': {'vp_step': 1.0}}), hm)
    assert len(w0.views) > len(w.views)
