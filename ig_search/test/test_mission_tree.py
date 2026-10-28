"""
Drzewo zachowań misji: gałęzie weryfikacji i chwytu (atrapy portów, bez detektora testowego).

Misja jest ustawiana tuż przed końcem dojazdu do pewnej próbki w zasięgu ramienia; klatki
nadgarstka podawane są ręcznie, więc da się sprawdzić gałęzie, których przebiegi end-to-end
w test_mission.py nie wymuszają (korekta pozycji przed chwytem, nieudany skan i chwyt).
"""
from conftest import flat, make_heightmap
from harness import FakeManip, FakeNav, ListReporter, wrist_pose_above
from ig_search.application.mission import Mission, State
from ig_search.application.ports import Frame, SamplePose, Status
from ig_search.application.world import build_world
from ig_search.domain.params import from_dict
from ig_search.domain.report_rule import in_workspace
from ig_search.domain.strategies import make_strategy
import numpy as np
import pytest

SAMPLE = (10.0, 10.0)
DT = 0.1


def params(**wrist):
    return from_dict({'views': {'vp_step': 2.0},
                      'mission': {'use_wrist': True}, 'wrist': wrist})


@pytest.fixture(scope='module')
def world():
    return build_world(params(), make_heightmap(flat))


class ScriptedManip(FakeManip):
    """Atrapa ramienia; wybrane ruchy kończą się porażką albo nigdy się nie kończą."""

    def __init__(self, duration=1.0, fail=(), hang=()):
        super().__init__(duration)
        self.fail, self.hang = fail, hang

    def tick(self, dt):
        name = self.calls[-1][0] if self.calls else ''
        if self.st != Status.ACTIVE or name in self.hang:
            return
        self.t_left -= dt
        if self.t_left <= 0:
            self.st = Status.FAILED if name in self.fail else Status.SUCCEEDED


class Run:
    """Misja ustawiona na końcu dojazdu: łazik przed próbką, ŝ w R(q)."""

    def __init__(self, world, manip, p=None, p_sample=0.99, offset=0.0):
        p = p or params()
        world.params = p
        world.views.ok[:] = True
        rp = p.report
        d = rp.workspace_root_x + 0.5 * (rp.workspace_r_min + 0.72)
        self.q = (SAMPLE[0] - d - offset, SAMPLE[1], 0.0)
        assert in_workspace(self.q, [SAMPLE[0]], [SAMPLE[1]], rp)[0] == (offset == 0.0)
        self.nav = FakeNav(self.q)
        self.manip = manip
        self.rep = ListReporter()
        self.m = Mission(world, make_strategy(p), self.nav, self.rep, manip=manip)
        self.t = 0.0
        self.sample = np.array([SAMPLE[0], SAMPLE[1], 0.0])
        self.step()                                 # start: składanie ramienia
        while self.m.arm_busy():
            self.step()
        m = self.m
        m.belief.set_p(int(m.belief.grid.flat_index(*SAMPLE)), p_sample)
        m._plan()                                   # dojazd do jedynego kandydata
        assert m.target is not None, m.events
        self.nav.q = list(self.q)
        self.nav.st = Status.SUCCEEDED              # łazik już na miejscu
        self.nav.goal = None

    def step(self, sample_seen=True, seen_close=None):
        """
        Krok misji z klatką nadgarstka nad ŝ (z detekcją próbki albo pustą).

        Przy skanie kamera ~0,6 m nad ŝ, przy zgłoszeniu (pozycja przed chwytem) ~0,3 m -
        inny kubełek odległości bramki różnorodności. seen_close - czy próbka widoczna
        z bliska (domyślnie jak sample_seen).
        """
        self.manip.tick(DT)
        if self.m.state in (State.VERIFY, State.REPORT):
            x, y = SAMPLE
            close = self.m.state == State.REPORT
            seen = sample_seen if seen_close is None or not close else seen_close
            dets, poses = np.empty((0, 3)), []
            if seen:
                dets = self.sample[None, :].copy()
                poses = [SamplePose(self.sample.copy(), np.array([0.0, 1.0, 0.0]))]
            pose = wrist_pose_above(x, y, 0.0, height=0.3 if close else 0.6)
            self.m.on_wrist_frame(Frame(self.t, pose, dets, poses))
        self.m.step(self.t, self.q)
        self.t += DT

    def until_report(self, t_max=120.0, **kw):
        while not self.rep.reports and self.t < t_max:
            self.step(**kw)
        return self.rep.reports

    def kinds(self):
        return [c[0] for c in self.manip.calls]


def test_scan_pregrasp_grasp_publish(world):
    r = Run(world, ScriptedManip())
    reports = r.until_report()
    assert len(reports) == 1
    assert r.kinds() == ['stow', 'scan', 'pregrasp', 'grasp', 'stow']
    assert reports[0].pregrasp_reached and reports[0].grasped
    assert r.m.state == State.SEARCH and r.m.target is None


def test_refine_shift_repeats_pregrasp_once(world):
    """Pomiar z bliska przesunął próbkę o > 1 cm: druga pozycja przed chwytem, potem chwyt."""
    r = Run(world, ScriptedManip())
    while 'pregrasp' not in r.kinds():
        r.step()
    r.sample[0] += 0.03                             # pomiar z bliska: próbka 3 cm dalej
    reports = r.until_report()
    assert r.kinds() == ['stow', 'scan', 'pregrasp', 'pregrasp', 'grasp', 'stow']
    assert reports[0].x == pytest.approx(SAMPLE[0] + 0.03, abs=1e-6)
    assert r.manip.calls[3][1].position[0] == pytest.approx(SAMPLE[0] + 0.03, abs=1e-6)


def test_certain_sample_out_of_reach_moves_into_reach(world):
    """
    Pewna próbka poza R(q): krótki podjazd z bieżącej pozy, potem zgłoszenie bez skanu.

    Podjazd o 0,25 m - ramię zostaje w pozie skanu i idzie od razu do pozycji przed chwytem
    (bez składania do pozycji transportowej).
    """
    r = Run(world, ScriptedManip(), offset=0.25)
    while not r.rep.reports and r.t < 120.0:
        r.nav.tick(DT)
        r.q = tuple(r.nav.q)
        r.step()
    kinds = [e.kind for e in r.m.events]
    assert 'podjazd w zasięg' in kinds and 'ponowne podejście' not in kinds, kinds
    assert r.kinds() == ['stow', 'scan', 'pregrasp', 'grasp', 'stow']
    assert in_workspace(r.rep.reports[0].robot, [SAMPLE[0]], [SAMPLE[1]], r.m.p.report)[0]


def test_close_measurements_fused_by_median(world):
    """
    Pomiary z bliska się różnią: cel chwytu to mediana pomiarów, najwyżej 2 korekty.

    Pomiar 1: +3 cm, kolejne: 0 cm. Korekty do +3 i do mediany +1,5 cm, potem mediana 0 cm -
    korekt brak, ale chwyt celuje w medianę (nie w pozycję z ostatniej korekty).
    """
    r = Run(world, ScriptedManip())
    while 'pregrasp' not in r.kinds():
        r.step()
    r.sample[0] = SAMPLE[0] + 0.03
    while r.kinds().count('pregrasp') < 2:
        r.step()
    r.sample[0] = SAMPLE[0]
    reports = r.until_report()
    assert r.kinds() == ['stow', 'scan', 'pregrasp', 'pregrasp', 'pregrasp', 'grasp', 'stow']
    assert r.manip.calls[3][1].position[0] == pytest.approx(SAMPLE[0] + 0.03, abs=1e-6)
    assert r.manip.calls[4][1].position[0] == pytest.approx(SAMPLE[0] + 0.015, abs=1e-6)
    assert r.manip.calls[5][1].position[0] == pytest.approx(SAMPLE[0], abs=1e-6)
    assert reports[0].x == pytest.approx(SAMPLE[0], abs=1e-6)


def test_true_sample_not_rejected_by_empty_close_frames(world):
    """
    Próbka wykryta przy skanie, z pozycji przed chwytem niewidoczna: zgłoszenie jest.

    Z ~0,3 m kadr nadgarstka to ±0,13 m, a P_d z tej odległości nieskalibrowane - w Gazebo
    puste klatki z pozycji przed chwytem zbijały P_rep prawdziwej próbki z 1,00 do 0,20.
    Klatki z pozycji przed chwytem służą tylko do pomiaru pozycji.
    """
    p = from_dict({'views': {'vp_step': 2.0}, 'mission': {'use_wrist': True},
                   'report': {'thresholds': [0.95, 0.95, 0.95]}})
    r = Run(world, ScriptedManip(), p, p_sample=0.7)
    before = None
    while not r.rep.reports and r.t < 120.0:
        r.step(seen_close=False)
        if r.m.state == State.REPORT and before is None:
            before = float(r.m.belief.p.max())
    assert len(r.rep.reports) == 1
    assert float(r.m.belief.p.max()) >= before - 1e-9 or r.m.reports


def test_failed_grasp_still_reports(world):
    r = Run(world, ScriptedManip(fail=('grasp',)))
    reports = r.until_report()
    assert len(reports) == 1 and reports[0].pregrasp_reached and not reports[0].grasped
    assert any('nieudany' in e.detail for e in r.m.events)


def test_failed_pregrasp_reports_without_grasp(world):
    r = Run(world, ScriptedManip(fail=('pregrasp',)))
    reports = r.until_report()
    assert 'grasp' not in r.kinds()
    assert not reports[0].pregrasp_reached and not reports[0].grasped


def test_grasp_disabled(world):
    r = Run(world, ScriptedManip(), params(grasp=False))
    reports = r.until_report()
    assert r.kinds() == ['stow', 'scan', 'pregrasp', 'stow']
    assert reports[0].pregrasp_reached and not reports[0].grasped


def test_scan_timeout_evaluates_available_frames(world):
    """Ramię nie dochodzi do pozy skanu: po scan_timeout ocena z klatek zebranych w ruchu."""
    r = Run(world, ScriptedManip(hang=('scan',)), params(scan_timeout=3.0))
    reports = r.until_report()
    assert any('nie dotarło' in e.detail for e in r.m.events)
    assert len(reports) == 1


def test_uncertain_sample_reapproaches_then_abandons(world):
    """Brak potwierdzenia z nadgarstka: ponowne podejścia, potem porzucenie bez zgłoszenia."""
    r = Run(world, ScriptedManip(), p_sample=0.7)
    while r.t < 200.0 and not any(e.kind == 'porzucenie' for e in r.m.events):
        r.nav.tick(DT)
        r.q = tuple(r.nav.q)
        r.step(sample_seen=False)
    kinds = [e.kind for e in r.m.events]
    assert not r.rep.reports
    assert 'weryfikacja' in kinds and 'porzucenie' in kinds, kinds
    assert r.m.target is None and r.m.state == State.SEARCH


def test_tree_renders_with_statuses(world):
    r = Run(world, ScriptedManip())
    r.step()
    txt = r.m.tree_ascii()
    for name in ('Misja', 'Dojazd do kandydata', 'Skan weryfikujący', 'Podjazd w zasięg',
                 'Zgłoszenie', 'Podejście z innej strony', 'Przeszukiwanie'):
        assert name in txt


def test_no_new_approach_to_weak_candidate(world):
    """Kandydat z P_k < p_approach_min nie jest celem dojazdu (od razu byłby porzucony)."""
    p = params()
    world.params = p
    world.views.ok[:] = True
    for mass, approach in ((0.08, False), (0.5, True)):
        m = Mission(world, make_strategy(p), FakeNav((9.0, 10.0, 0.0)), ListReporter(),
                    manip=ScriptedManip())
        m.q = (9.0, 10.0, 0.0)
        m.belief.set_p(int(m.belief.grid.flat_index(10.1, 10.1)), mass)
        m._plan()
        assert (m.target is not None) == approach, (mass, m.events)


def test_no_report_without_wrist_detection(world):
    """
    Pewna z kamery przedniej, ale nadgarstek nie wykrył próbki: bez zgłoszenia.

    Fałszywe alarmy kamery przedniej z kilku stron (ten sam kamień) nie są niezależne;
    weryfikacją jest detekcja z nadgarstka.
    """
    r = Run(world, ScriptedManip(), p_sample=0.99999)
    while r.t < 60.0 and 'ponowne podejście' not in [e.kind for e in r.m.events]:
        r.nav.tick(DT)
        r.q = tuple(r.nav.q)
        r.step(sample_seen=False)
    assert not r.rep.reports
    assert any('brak detekcji z nadgarstka' in e.detail for e in r.m.events), r.m.events
