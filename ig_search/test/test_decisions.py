"""M4: kandydaci, reguła zgłoszenia, decyzje strategii (DESIGN §3.3–3.5, T4)."""
import math

from conftest import flat, make_heightmap
from ig_search.application.world import build_world
from ig_search.domain.candidates import (
    attach_approach_poses, find_candidates, mass_under_rover)
from ig_search.domain.params import from_dict
from ig_search.domain.report_rule import (
    check_report, in_workspace, reach_ring_width, threshold, workspace_r_max)
from ig_search.domain.strategies import make_strategy
from ig_search.domain.strategies.base import APPROACH, OBSERVE, PlanningContext
import numpy as np
import pytest

PARAMS = from_dict({'views': {'vp_step': 2.0}})


@pytest.fixture(scope='module')
def world():
    return build_world(PARAMS, make_heightmap(flat))


def put(belief, x, y, p):
    c = int(belief.grid.flat_index(x, y))
    belief.set_p(c, p)
    return c


def ctx_for(world, belief, q):
    cands = [attach_approach_poses(c, world.travel, PARAMS.candidates)
             for c in find_candidates(belief, PARAMS.candidates)]
    return PlanningContext(q, belief, world.views, world.travel,
                           world.travel.drive_times(q[0], q[1]), cands, PARAMS)


def test_candidate_mass_and_estimate(world):
    b = world.new_belief()
    put(b, 10.1, 10.1, 0.3)
    put(b, 10.3, 10.1, 0.2)
    cands = find_candidates(b, PARAMS.candidates)
    assert len(cands) == 1
    c = cands[0]
    assert math.isclose(c.mass, 1 - 0.7 * 0.8, rel_tol=1e-6)    # 1 − Π(1 − p)
    assert 10.1 < c.x < 10.3 and math.isclose(c.y, 10.1, abs_tol=1e-6)


def test_approach_poses_face_candidate(world):
    b = world.new_belief()
    put(b, 10.1, 10.1, 0.9)
    c = attach_approach_poses(find_candidates(b, PARAMS.candidates)[0], world.travel,
                              PARAMS.candidates)
    assert len(c.app_x) == PARAMS.candidates.n_approach_dirs
    for x, y, yaw in zip(c.app_x, c.app_y, c.app_yaw):
        assert math.isclose(math.hypot(c.x - x, c.y - y), PARAMS.candidates.approach_distance)
        assert in_workspace((x, y, yaw), [c.x], [c.y], PARAMS.report)[0]


def test_workspace_is_ring_around_arm_root_not_only_ahead():
    """R(q): pełny chwyt pionowy 0,60-0,74 m od podstawy ramienia (płasko), także z boku."""
    rp, q = PARAMS.report, (0.0, 0.0, 0.0)
    root = rp.workspace_root_x

    def at(rho, deg):
        return [root + rho * math.cos(math.radians(deg))], [rho * math.sin(math.radians(deg))]
    inside = [at(0.70, 0), at(0.70, 60), at(0.70, -70), at(0.62, 30), at(0.73, -30)]
    outside = [at(0.55, 0), at(0.80, 0), at(0.70, 90), at(0.70, 180)]
    assert all(in_workspace(q, *pt, rp)[0] for pt in inside)
    assert not any(in_workspace(q, *pt, rp)[0] for pt in outside)
    # z zapasem (koniec dojazdu) - węższy pierścień i zakres kątów
    assert not in_workspace(q, *at(0.62, 0), rp, margin=True)[0]
    assert not in_workspace(q, *at(0.67, 70), rp, margin=True)[0]
    assert in_workspace(q, *at(0.67, 60), rp, margin=True)[0]
    # próbka niżej niż łazik (w dół zbocza) - krótszy zasięg chwytu (IK)
    assert in_workspace(q, *at(0.70, 0), rp, dz=0.015)[0]
    assert not in_workspace(q, *at(0.70, 0), rp, dz=-0.06)[0]
    assert in_workspace(q, *at(0.70, 0), rp, dz=0.06)[0]


def test_confident_near_candidate_beats_observation(world):
    b = world.new_belief()
    put(b, 6.1, 10.1, 0.9)
    a = make_strategy(PARAMS).choose(ctx_for(world, b, (4.0, 10.0, 0.0)))
    assert a.kind == APPROACH


def test_uncertain_far_candidate_loses(world):
    b = world.new_belief()
    put(b, 18.1, 18.1, 0.05)
    a = make_strategy(PARAMS).choose(ctx_for(world, b, (2.0, 2.0, 0.0)))
    assert a.kind == OBSERVE


def test_thresholds_increase():
    assert threshold(0, PARAMS.report) < threshold(2, PARAMS.report)
    assert threshold(5, PARAMS.report) == PARAMS.report.thresholds[-1]


def test_report_check_in_reach(world):
    b = world.new_belief()
    put(b, 11.1, 10.1, 0.9)
    q = (10.3, 10.1, 0.0)
    chk = check_report(b, q, 11.0, 10.0, PARAMS.report)
    assert chk.in_reach
    assert chk.probability >= 0.9
    assert math.isclose(chk.x, 11.1, abs_tol=1e-6)
    # ta sama próbka za łazikiem: pewna, ale poza obszarem roboczym
    chk_back = check_report(b, (12.0, 10.1, 0.0), 11.0, 10.0, PARAMS.report)
    assert chk_back.probability >= 0.9 and not chk_back.in_reach


def test_tau3_blocks_uncertain_third_report(world):
    b = world.new_belief()
    put(b, 11.1, 10.1, 0.88)
    q = (10.3, 10.1, 0.0)
    p = check_report(b, q, 11.1, 10.1, PARAMS.report).probability
    assert p >= threshold(0, PARAMS.report)          # 1. zgłoszenie - wystarczy
    assert p < threshold(2, PARAMS.report)           # 3. zgłoszenie - za mało


def test_baselines_choose_something(world):
    b = world.new_belief()
    for name in ('meander', 'frontier'):
        p = from_dict({'views': {'vp_step': 2.0}, 'planner': {'strategy': name}})
        a = make_strategy(p).choose(ctx_for(world, b, (2.0, 2.0, 0.0)))
        assert a is not None and a.kind == OBSERVE, name
        # pewny kandydat -> dojazd (reguła progowa)
        put(b, 4.1, 2.1, 0.9)
        a = make_strategy(p).choose(ctx_for(world, b, (2.0, 2.0, 0.0)))
        assert a.kind == APPROACH, name
        b = world.new_belief()


def test_meander_lanes_cover_area(world):
    from ig_search.domain.strategies.meander import build_waypoints, lane_spacing
    xs, ys, _ = build_waypoints(world.travel, PARAMS)
    assert len(np.unique(np.round(ys, 3))) >= int(16 / lane_spacing(PARAMS))


def test_approach_pose_not_on_candidate_blob(world):
    """
    Poza dojazdu nie stawia łazika na skupisku kandydata.

    Skupisko wydłużone wzdłuż promienia widzenia (estymata z daleka): ŝ w środku, a masa
    ciągnie się 0,8 m w -x. Poza od strony -x (0,78 m od ŝ) stałaby na prawdziwej próbce.
    """
    b = world.new_belief()
    for k in range(5):                          # środki komórek (bez błędów zaokrągleń)
        put(b, 10.1 - 0.2 * k, 10.1, 0.25)
    for k in range(1, 5):
        put(b, 10.1 + 0.2 * k, 10.1, 0.05)
    c = find_candidates(b, world.params.candidates)[0]
    cp = world.params.candidates
    attach_approach_poses(c, world.travel, cp)
    under = mass_under_rover(c, world.travel, c.app_x, c.app_y)
    assert len(c.app_x) > 0
    assert np.all(under <= cp.max_mass_under_rover * c.mass + 1e-9), under
    # kierunek podejścia od strony -x (łazik w ŝ - 0,78 m w x) odrzucony
    assert not np.any(np.abs((c.app_yaw + math.pi) % (2 * math.pi) - math.pi) < 0.1)


def test_reach_shrinks_below_ik_table():
    """Próbka niżej niż tabela IK (-6 cm): zasięg maleje dalej, nie wartość brzegowa."""
    rp = PARAMS.report
    assert float(workspace_r_max(rp, 0.0)) == pytest.approx(0.72)
    assert float(workspace_r_max(rp, -0.09)) < rp.workspace_r_min
    assert float(reach_ring_width(rp, -0.07)) <= 0.0 < float(reach_ring_width(rp, 0.0))
    assert float(workspace_r_max(rp, 0.10)) == pytest.approx(rp.workspace_r_max_dz[-1])


def test_approach_pose_skips_empty_reach_ring(world):
    """Pozy, z których ŝ jest za nisko (pusty pierścień R(q) z zapasem), odpadają."""
    b = world.new_belief()
    put(b, 10.1, 10.1, 0.5)
    c = find_candidates(b, world.params.candidates)[0]
    attach_approach_poses(c, world.travel, world.params.candidates,
                          ring=lambda xs, ys: np.where(np.asarray(xs) < c.x, -0.05, 0.05))
    assert len(c.app_x) > 0 and np.all(c.app_x >= c.x - 1e-9)
    attach_approach_poses(c, world.travel, world.params.candidates,
                          ring=lambda xs, ys: -np.hypot(np.asarray(xs) - 9.0, 0 * np.asarray(ys)))
    assert len(c.app_x) >= 1                    # wszystkie złe - zostaje najlepsza
