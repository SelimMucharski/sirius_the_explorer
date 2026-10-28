"""M3: belief z fałszywymi alarmami i bramką różnorodności (DESIGN §3.2, T3)."""
import math

from conftest import default_params, flat, make_heightmap
from ig_search.domain.belief import Belief, presence
from ig_search.domain.detector_model import DetectorModel
from ig_search.domain.params import with_section
from ig_search.domain.visibility import Visibility
import numpy as np


def make_belief(params=None):
    p = params or default_params()
    hm = make_heightmap(flat)
    grid = hm.grid.coarsen(int(round(p.belief.cell_size / hm.grid.res)))
    vis = Visibility(hm, grid, p.camera, p.hfov, p.vfov)
    model = DetectorModel(p.detector, p.camera, p.hfov)
    prior = Belief.uniform_prior(vis.cell_valid, p.belief.n_samples)
    return Belief(grid, prior, p.belief, model), vis, model, p


def frame(b, vis, x, y, psi, dets=(), radius=None):
    pose = vis.camera_pose(x, y, psi)
    v = vis.visible(pose)
    dets = np.asarray(dets, float).reshape(-1, 2)
    if len(dets):
        v = v.union(vis.cells_near(pose, dets, radius or 0.5))
    return b.update(v, dets, pose.position[:2]), v


def test_prior_mass_is_m():
    b, _, _, p = make_belief()
    assert math.isclose(b.mass(), p.belief.n_samples, rel_tol=1e-9)


def test_empty_frame_lowers_only_visible():
    b, vis, _, _ = make_belief()
    p0 = b.p.copy()
    _, v = frame(b, vis, 5.0, 10.0, 0.0)
    assert np.all(b.p[v.ids] < p0[v.ids])
    other = np.setdiff1d(np.arange(len(p0)), v.ids)
    np.testing.assert_array_equal(b.p[other], p0[other])


def blob(b, p0, center, thresh=1.5):
    ratio = b.p / p0
    m = ratio > thresh
    d = np.hypot(b.cx[m] - center[0], b.cy[m] - center[1])
    return d.max() if m.any() else 0.0, ratio.max()


def test_far_detection_wide_weak_near_narrow_strong():
    b_far, vis, _, _ = make_belief()
    b_near, _, _, _ = make_belief()
    p0 = b_far.p.copy()
    target = (11.0, 10.0)
    frame(b_far, vis, 5.6, 10.0, 0.0, [target], radius=1.0)    # ~5 m od kamery
    frame(b_near, vis, 9.6, 10.0, 0.0, [target], radius=1.0)   # ~1 m
    w_far, peak_far = blob(b_far, p0, target)
    w_near, peak_near = blob(b_near, p0, target)
    assert w_far > w_near
    assert peak_near > peak_far


def test_gate_same_viewpoint_counts_once():
    b1, vis, _, _ = make_belief()
    b3, _, _, _ = make_belief()
    target = [(9.0, 10.0)]
    frame(b1, vis, 5.0, 10.0, 0.0, target)
    for _ in range(3):
        frame(b3, vis, 5.0, 10.0, 0.0, target)
    np.testing.assert_allclose(b3.ell, b1.ell)


def test_gate_off_accumulates():
    p = with_section(default_params(), 'belief')
    b, vis, _, _ = make_belief(p)
    target = np.array([[9.0, 10.0]])
    pose = vis.camera_pose(5.0, 10.0, 0.0)
    v = vis.visible(pose)
    c = int(b.grid.flat_index(9.0, 10.0))
    b.update(v, target, pose.position[:2], use_gate=False)
    e1 = b.ell[c]
    b.update(v, target, pose.position[:2], use_gate=False)
    assert b.ell[c] > e1


def test_update_matches_formula():
    """Pierwsza detekcja - wzrost log-szans zgodny z formułą z wykluczeniem komórki i."""
    b, vis, model, _ = make_belief()
    target = np.array([10.1, 10.1])                 # środek komórki
    c = int(b.grid.flat_index(*target))
    pose = vis.camera_pose(8.0, 10.1, 0.0)
    v = vis.visible(pose).union(vis.cells_near(pose, target[None], 0.6))
    keep = b.active[v.ids]
    v_ids = v.ids[keep]
    pd = model.pd(v.r[keep], v.alpha[keep])
    s = np.sqrt(model.sigma(v.r[keep]) ** 2 + b.grid.res ** 2 / 12)
    d2 = (b.cx[v_ids] - target[0]) ** 2 + (b.cy[v_ids] - target[1]) ** 2
    g = np.where(d2 <= 9 * s * s, np.exp(-d2 / (2 * s * s)) / (2 * math.pi * s * s), 0.0)
    k = int(np.flatnonzero(v_ids == c)[0])
    other = float((pd * g * b.p[v_ids]).sum() - pd[k] * g[k] * b.p[c])
    expected = b.ell[c] + math.log((1 - pd[k]) + pd[k] * g[k] / (model.lambda_fa + other))
    b.update(v, target[None], pose.position[:2])
    np.testing.assert_allclose(b.ell[c], expected, rtol=1e-9)


def repeated_detections(explain_away: bool, n_views=6, miss_after=False):
    p = with_section(default_params(), 'belief', explain_away=explain_away)
    b, vis, _, _ = make_belief(p)
    target = np.array([10.1, 10.1])
    for ang in np.linspace(0, 2 * math.pi, n_views, endpoint=False):
        x, y = target - 2.0 * np.array([math.cos(ang), math.sin(ang)])
        pose = vis.camera_pose(x, y, ang)
        v = vis.visible(pose).union(vis.cells_near(pose, target[None], 0.6))
        b.update(v, target[None], pose.position[:2])
    ball = np.hypot(b.cx - target[0], b.cy - target[1]) <= 0.25
    p_before = presence(b.p[ball])
    if miss_after:                          # pominięcie z nowego punktu widzenia
        pose = vis.camera_pose(target[0], target[1] + 1.5, -math.pi / 2)
        b.update(vis.visible(pose), np.empty((0, 2)), pose.position[:2])
    near = np.hypot(b.cx - target[0], b.cy - target[1]) <= 0.6
    return b.p[near].sum(), p_before, presence(b.p[ball])


def test_explain_away_mass_near_one():
    mass, p_ball, _ = repeated_detections(explain_away=True)
    assert 0.8 < mass < 1.6                 # jedna próbka ≈ jedna jednostka masy
    assert p_ball >= 0.95


def test_without_explain_away_one_sample_spreads_into_many():
    mass, _, _ = repeated_detections(explain_away=False)
    assert mass > 3.0                       # nadmierne „rozmnożenie” jednej próbki


def test_single_miss_keeps_confirmed_sample():
    _, before, after = repeated_detections(explain_away=True, miss_after=True)
    assert before >= 0.95
    assert 0.6 < after < before


def test_precise_detection_off_cell_center_still_counts():
    """σ ≪ a (nadgarstek): detekcja przy rogu komórki musi podnieść belief, nie obniżyć."""
    p = default_params()
    b, vis, _, _ = make_belief(p)
    from ig_search.domain.detector_model import DetectorModel
    wrist = DetectorModel(p.wrist_detector, p.wrist_camera, p.wrist_hfov)
    target = np.array([10.19, 10.19])               # 13 cm od środka komórki
    c = int(b.grid.flat_index(*target))
    before = b.p[c]
    pose = vis.camera_pose(9.0, 10.19, 0.0)
    v = vis.visible(pose).union(vis.cells_near(pose, target[None], 0.3))
    b.update(v, target[None], pose.position[:2], model=wrist)
    assert b.p[c] > before


def test_clear_zeroes_region():
    b, _, _, _ = make_belief()
    b.clear(10.0, 10.0, 0.5)
    m = np.hypot(b.cx - 10.0, b.cy - 10.0) <= 0.5
    assert np.all(b.p[m] == 0.0)
    assert b.mass() < 3.0


def test_cleared_cells_ignore_new_detections():
    b, vis, _, _ = make_belief()
    b.clear(9.0, 10.0, 0.5)
    frame(b, vis, 5.0, 10.0, 0.0, [(9.0, 10.0)])
    m = np.hypot(b.cx - 9.0, b.cy - 10.0) <= 0.5
    assert np.all(b.p[m] == 0.0)


def test_view_counts_distinct_viewpoints():
    b, vis, _, _ = make_belief()
    frame(b, vis, 5.0, 10.0, 0.0)
    frame(b, vis, 5.0, 10.0, 0.0)               # ten sam punkt widzenia - bez zmiany
    once = b.view_counts()
    assert once.max() == 1 and (once == 1).sum() == b.observed().sum()
    frame(b, vis, 13.0, 10.0, math.pi)          # z drugiej strony
    both = b.view_counts()
    assert both.max() == 2 and (both >= once).all()
