"""M4: automat misji end-to-end offline (atrapy portów, detektor testowy)."""
import math

from conftest import flat, make_heightmap
from harness import FakeManip, FakeNav, ListReporter, run_mission, ToyDetector
from ig_search.application.mission import Mission, State
from ig_search.application.ports import Status
from ig_search.application.world import build_world
from ig_search.domain.params import from_dict
from ig_search.domain.strategies import make_strategy
import numpy as np
import pytest

PROBES = [(5.0, 15.0), (15.0, 5.0), (16.0, 16.0)]
START = (2.0, 2.0, 0.0)


def params(**sections):
    base = {'views': {'vp_step': 2.0}, 'mission': {'use_wrist': False}}
    for k, v in sections.items():
        base.setdefault(k, {}).update(v)
    return from_dict(base)


@pytest.fixture(scope='module')
def world():
    return build_world(params(), make_heightmap(flat))


def correct(reports, probes, eps=0.35):
    """Zgłoszenia poprawne: każda próbka liczona raz (jak sędzia)."""
    left = list(probes)
    n = 0
    for r in reports:
        d = [math.hypot(r.x - px, r.y - py) for px, py in left]
        if d and min(d) < eps:
            left.pop(int(np.argmin(d)))
            n += 1
    return n


def make_mission(world, p, manip=None, seed=0, fa_scale=1.0):
    world.params = p
    world.views.ok[:] = True
    nav = FakeNav(START)
    rep = ListReporter()
    m = Mission(world, make_strategy(p), nav, rep, manip=manip)
    det = ToyDetector(world.vis, world.model, PROBES, np.random.default_rng(seed), fa_scale)
    return m, nav, rep, det


def test_greedy_finds_all_samples(world):
    m, nav, rep, det = make_mission(world, params())
    t = run_mission(m, nav, det, dt=0.5)
    assert m.state == State.DONE, m.events[-5:]
    assert len(rep.reports) == 3
    assert correct(rep.reports, PROBES) == 3
    assert t < 1800


def test_greedy_with_wrist_scan_and_pregrasp(world):
    p = params(mission={'use_wrist': True})
    manip = FakeManip(duration=3.0)
    m, nav, rep, det = make_mission(world, p, manip)
    wrist = ToyDetector(world.wrist_vis, world.wrist_model, PROBES, np.random.default_rng(1))
    run_mission(m, nav, det, manip=manip, wrist_det=wrist, dt=0.5)
    assert correct(rep.reports, PROBES) == 3
    kinds = [c[0] for c in manip.calls]
    assert kinds.count('scan') >= 3 and kinds.count('pregrasp') >= 3
    assert kinds.count('grasp') == 3
    assert all(r.pregrasp_reached and r.grasped for r in rep.reports)
    assert all(r.sample_pose is not None for r in rep.reports)


def test_wrist_approach_stops_once_sample_is_in_reach(world):
    """Dojazd kończy się, gdy ŝ jest w R(q) - bez dojeżdżania do pozy i obrotu do kursu."""
    from ig_search.domain.report_rule import in_workspace
    p = params(mission={'use_wrist': True})
    manip = FakeManip(duration=3.0)
    m, nav, rep, det = make_mission(world, p, manip)
    wrist = ToyDetector(world.wrist_vis, world.wrist_model, PROBES, np.random.default_rng(1))
    at_scan = []                                # poza łazika na końcu dojazdu (start skanu)
    scan = manip.scan

    def scan_and_record(x, y, dz):
        at_scan.append((tuple(nav.q), x, y))
        scan(x, y, dz)
    manip.scan = scan_and_record
    run_mission(m, nav, det, manip=manip, wrist_det=wrist, dt=0.1)
    assert len(rep.reports) == 3
    for r in rep.reports:
        assert in_workspace(r.robot, [r.x], [r.y], p.report)[0]
    # zatrzymanie, zanim łazik dojechał do pozy dojazdu (ŝ wszedł w R(q) z zapasem)
    dist = [math.hypot(x - q[0], y - q[1]) for q, x, y in at_scan]
    assert max(dist) > p.candidates.approach_distance + 0.02, dist


@pytest.mark.parametrize('strategy', ['meander', 'frontier', 'chase', 'ig_horizon'])
def test_baselines_complete(world, strategy):
    m, nav, rep, det = make_mission(world, params(planner={'strategy': strategy}))
    run_mission(m, nav, det, dt=0.5)
    assert correct(rep.reports, PROBES) >= 2, m.events[-8:]


def test_false_candidate_is_abandoned(world):
    """
    Fałszywy kandydat jest porzucany bez zgłoszenia.

    Wysokie p w miejscu bez próbki (np. po fałszywych alarmach): dojazd, brak
    potwierdzenia z bliska -> spadek P_k i porzucenie.
    """
    m, nav, rep, det = make_mission(world, params(), fa_scale=0.0)
    det.probes = np.empty((0, 2))
    c = int(m.belief.grid.flat_index(6.1, 2.1))
    m.belief.set_p(c, 0.7)
    m.p = from_dict({'views': {'vp_step': 2.0}, 'mission': {'use_wrist': False, 't_max': 120.0}})
    run_mission(m, nav, det, dt=0.5, t_max=120.0)
    kinds = [e.kind for e in m.events]
    assert 'akcja' in kinds
    assert not rep.reports
    assert m.belief.p[c] < 0.05


def test_watchdog_breaks_endless_navigation(world):
    """Nav2 krąży tuż za tolerancją celu bez porażki - misja nie może stać w miejscu."""
    m, nav, rep, det = make_mission(world, params(), fa_scale=0.0)
    det.probes = np.empty((0, 2))
    nav.tick = lambda dt: None                   # cel nigdy nie zostaje osiągnięty
    run_mission(m, nav, det, dt=0.5, t_max=400.0)
    kinds = [e.detail for e in m.events if e.kind == 'nawigacja']
    assert any('przerwana' in d for d in kinds)
    assert len(nav.goals) >= 2


def test_stuck_rover_backs_up_and_blocks_place():
    """
    Koła buksują: estymowana poza skacze tam i z powrotem (odometria vs GNSS), netto stoi.

    Misja wykrywa utknięcie, cofa łazik, wyłącza miejsce z grafu przejezdności i jedzie dalej.
    """
    w = build_world(params(), make_heightmap(flat))
    m, nav, rep, det = make_mission(w, params(), fa_scale=0.0)
    det.probes = np.empty((0, 2))
    state = {'k': 0}

    def slipping(dt):
        if nav.st == Status.ACTIVE and not nav.backups:
            state['k'] += 1
            nav.q[0] += 0.3 if state['k'] % 2 else -0.3
    nav.tick = slipping
    trav0 = int(w.travel.traversable.sum())
    run_mission(m, nav, det, dt=0.5, t_max=120.0)
    kinds = [e.kind for e in m.events]
    assert 'utknięcie' in kinds, kinds
    assert nav.backups and nav.backups[0] == pytest.approx(0.8)
    assert int(w.travel.traversable.sum()) < trav0
    assert len(m.stuck_points) >= 1
    i = kinds.index('utknięcie')
    assert 'akcja' in kinds[i + 1:]              # misja wybiera dalej nowe cele


def test_repeated_failures_in_place_trigger_recovery():
    """
    Nav2 odrzuca każdy cel od razu (np. „Start occupied” przy brzegu warstwy statycznej).

    Łazik stoi, cele nie trwają, więc utknięcie z ruchu się nie wykrywa; po kilku nieudanych
    nawigacjach bez ruchu misja cofa łazik i wyłącza miejsce, zamiast wybierać cele w kółko.
    """
    w = build_world(params(), make_heightmap(flat))
    m, nav, rep, det = make_mission(w, params(), fa_scale=0.0)
    det.probes = np.empty((0, 2))
    nav.fail = lambda goal: not nav.backups              # do pierwszego cofnięcia
    run_mission(m, nav, det, dt=0.5, t_max=60.0)
    kinds = [e.kind for e in m.events]
    assert 'utknięcie' in kinds, kinds
    assert nav.backups and nav.backups[0] == pytest.approx(0.8)
    assert kinds[:kinds.index('utknięcie')].count('nawigacja') == 3
    assert len(m.stuck_points) == 1


def test_score_factor_follows_erc_rules():
    from ig_search.domain.params import MissionParams
    f = MissionParams().score_factor
    assert [f(t) for t in (0, 450, 451, 750, 1000, 1350, 1351, 1800)] == \
        [1.0, 1.0, 0.9, 0.9, 0.8, 0.7, 0.6, 0.6]


@pytest.mark.parametrize('strategy', ['meander', 'frontier'])
def test_baselines_start_next_pass_after_coverage(world, strategy):
    # bez próbek: jedno przejście kończy misję przed T_max, bez limitu - szuka do T_max
    one = params(planner={'strategy': strategy, 'coverage_passes': 1})
    m, nav, rep, det = make_mission(world, one)
    det.probes = np.empty((0, 2))
    run_mission(m, nav, det, dt=1.0)
    assert m.state == State.DONE and m.elapsed < 1800.0, m.events[-3:]
    m, nav, rep, det = make_mission(world, params(planner={'strategy': strategy}))
    det.probes = np.empty((0, 2))
    run_mission(m, nav, det, dt=1.0)
    assert m.strategy.coverage_pass >= 2, m.events[-3:]


def test_no_progress_while_driving_triggers_recovery():
    """
    Koła buksują, a poza (z Gazebo) stoi: cel aktywny, droga z odometrii nie rośnie.

    Po STALL_WINDOW s jazdy bez przesunięcia - cofnięcie i miejsce wyłączone, jak po utknięciu.
    """
    from ig_search.application.mission import STALL_WINDOW
    w = build_world(params(), make_heightmap(flat))
    m, nav, rep, det = make_mission(w, params(), fa_scale=0.0)
    det.probes = np.empty((0, 2))
    nav.v = 0.0                                          # jedzie, ale stoi w miejscu
    run_mission(m, nav, det, dt=0.5, t_max=STALL_WINDOW + 20.0)
    stuck = [e for e in m.events if e.kind == 'utknięcie']
    assert stuck and 'brak postępu' in stuck[0].detail, [e.kind for e in m.events]
    assert STALL_WINDOW <= stuck[0].t <= STALL_WINDOW + 10.0
