"""
Drzewo zachowań misji (py_trees): przeszukiwanie, dojazd, weryfikacja, zgłoszenie.

Drzewo jest tykane raz na `Mission.step()` (np. 5 Hz). Węzły czekają na status portów
(nawigacja, ramię) i wołają operacje kontekstu misji (`Mission`), który trzyma belief,
śledzonego kandydata i zgłoszenia. Wybór gałęzi wynika z danych (koniec misji, śledzony
kandydat), a nie z jawnego stanu automatu; `Mission.state` (numeracja MissionStatus) ustawiają
akcje - służy tylko do statusu i filtra klatek nadgarstka. Szczegóły wykonania (składanie
ramienia przed jazdą, odłożony cel nawigacji, czekanie na kamerę) są wewnątrz akcji.

    Misja                                   [Selector]
    ├── Koniec misji?                       M zgłoszeń, brak akcji albo T_max (→ TIMEOUT)
    ├── Kandydat                            [Sequence]
    │   ├── Śledzony kandydat?
    │   └── Obsługa kandydata               [Selector]
    │       ├── Kandydat → zgłoszenie       [Sequence, pamięć]
    │       │   ├── Dojazd do kandydata     replanowanie, śledzenie, porzucenie przy zaniku
    │       │   ├── Skan weryfikujący       (bez nadgarstka: ustalenie się kamery przedniej)
    │       │   ├── P_rep ≥ τ_n?
    │       │   ├── Zasięg                  [Selector, pamięć]
    │       │   │   ├── ŝ ∈ R(q)?
    │       │   │   └── Podjazd w zasięg
    │       │   └── Zgłoszenie              [Sequence, pamięć]; wejście: zgłoszenie w przygotowaniu
    │       │       ├── Przed chwytem       pomiar z bliska, do 2 korekt (mediana pomiarów)
    │       │       ├── Chwyt i podniesienie
    │       │       └── Publikacja zgłoszenia
    │       │       (bez nadgarstka: tylko publikacja)
    │       ├── Podejście z innej strony    najwyżej max_verify_attempts razy
    │       └── Porzucenie kandydata
    └── Przeszukiwanie                      [Sequence]
        ├── Wynik obserwacji                pozę osiągnięto / wykluczono
        └── Planowanie                      wybór akcji strategii (histereza)

Przed chwytem i chwyt nie przerywają zgłoszenia - wynik trafia do flag pregrasp_reached
i grasped (detekcja pozostaje ważna).
"""
from typing import TYPE_CHECKING

from ig_search.application.ports import Status
import py_trees
from py_trees.common import Status as BT

if TYPE_CHECKING:                               # pragma: no cover
    from ig_search.application.mission import Mission


class MissionBehaviour(py_trees.behaviour.Behaviour):
    """Węzeł z dostępem do kontekstu misji."""

    def __init__(self, name: str, m: 'Mission'):
        super().__init__(name)
        self.m = m


class Condition(MissionBehaviour):
    """Warunek: SUCCESS / FAILURE z predykatu na kontekście misji."""

    def __init__(self, name: str, m: 'Mission', pred):
        super().__init__(name, m)
        self.pred = pred

    def update(self) -> BT:
        return BT.SUCCESS if self.pred(self.m) else BT.FAILURE


class Timed(MissionBehaviour):
    """Węzeł z czasem wejścia (czas misji, nie zegar ścienny)."""

    def initialise(self):
        self.t0 = self.m.now

    @property
    def dt(self) -> float:
        return self.m.now - self.t0

    def restart(self):
        self.t0 = self.m.now


def arm_moving(m: 'Mission', t0: float, timeout_attr: str) -> bool:
    """Ruch ramienia trwa (ACTIVE i nie minął limit z WristParams)."""
    return (m.manip.status() == Status.ACTIVE
            and m.now - t0 < getattr(m.p.wrist, timeout_attr))


# ---------------------------------------------------------------- koniec misji
class MissionOver(MissionBehaviour):
    """Koniec misji (M zgłoszeń, brak sensownych akcji) albo przekroczony T_max."""

    def update(self) -> BT:
        m = self.m
        if not m.finished and m.elapsed > m.p.mission.t_max:
            m.timeout()
        return BT.SUCCESS if m.finished else BT.FAILURE


# ---------------------------------------------------------------- przeszukiwanie
class ObservationResult(MissionBehaviour):
    """
    Koniec jazdy do pozy obserwacji: osiągnięta (wstrzymanie pozy) albo wykluczona.

    RUNNING, póki jazda nie ruszyła (składanie ramienia, odłożony cel) - bez planowania.
    """

    def update(self) -> BT:
        m = self.m
        if m.drive_blocked():
            return BT.RUNNING
        nav = m.watchdog(m.nav.status())
        if m.action is not None and nav in (Status.SUCCEEDED, Status.FAILED):
            m.observation_done(nav == Status.SUCCEEDED)
        return BT.SUCCESS


class Plan(MissionBehaviour):
    """Replanowanie co replan_period albo po nowym dowodzie; akcja trwa (RUNNING)."""

    def update(self) -> BT:
        self.m.plan_if_due()
        return BT.RUNNING


# ---------------------------------------------------------------- kandydat
class DriveToCandidate(MissionBehaviour):
    """
    Jazda do pozy dojazdu (z ramieniem do chwili, gdy ŝ ∈ R(q) z zapasem).

    Ramię jest najpierw składane do pozycji transportowej. W trakcie jazdy replanowanie
    śledzi kandydata: zanik albo P_k < p_abandon - porzucenie, lepsza akcja strategii -
    wywłaszczenie (drzewo zmienia gałąź w następnym tyku). FAILURE - nawigacja nieudana.
    """

    def initialise(self):
        self.m.state_approach()

    def update(self) -> BT:
        m = self.m
        if m.drive_blocked():
            return BT.RUNNING
        nav = m.watchdog(m.nav.status())
        if m.action is not None and nav == Status.SUCCEEDED:
            return BT.SUCCESS
        if m.action is not None and nav == Status.FAILED:
            m.recovery = 'nawigacja do pozy dojazdu nieudana'
            return BT.FAILURE
        m.plan_if_due()
        return BT.RUNNING


class WristScan(Timed):
    """
    Skan kamerą nadgarstka nad ŝ, postój kamery i poza próbki z nieruchomej kamery.

    Gdy ramię nie dojdzie do pozy skanu w scan_timeout, ocena z klatek zebranych w ruchu.
    Zawsze SUCCESS - o zgłoszeniu decyduje reguła zgłoszenia.
    """

    def initialise(self):
        super().initialise()
        self.dwell = False
        self.m.enter_verify()
        self.m.start_scan()

    def update(self) -> BT:
        m = self.m
        if not self.dwell:
            if arm_moving(m, self.t0, 'scan_timeout'):
                return BT.RUNNING
            if m.manip.status() != Status.SUCCEEDED:
                m.event('skan', 'ramię nie dotarło do pozy skanu - ocena z dostępnych klatek')
                return BT.SUCCESS
            self.dwell = True
            self.restart()
        if self.dt < m.SCAN_DWELL:
            return BT.RUNNING
        m.measure_scan(self.t0)
        return BT.SUCCESS


class FrontSettle(Timed):
    """Bez nadgarstka: kilka klatek kamery przedniej z nieruchomego łazika."""

    def initialise(self):
        super().initialise()
        self.m.enter_verify()

    def update(self) -> BT:
        return BT.RUNNING if self.dt < self.m.SETTLE_TIME else BT.SUCCESS


class MoveIntoReach(MissionBehaviour):
    """Krótki podjazd, który wprowadza ŝ w R(q); FAILURE, gdy się nie da."""

    def initialise(self):
        self.started = self.m.start_reach_move()

    def update(self) -> BT:
        m = self.m
        if not self.started:
            return BT.FAILURE
        if m.drive_blocked():
            return BT.RUNNING
        nav = m.watchdog(m.nav.status())
        if nav == Status.ACTIVE:
            return BT.RUNNING
        if nav != Status.SUCCEEDED:
            m.recovery = 'podjazd w zasięg nieudany'
            return BT.FAILURE
        if not m.target_in_reach():
            m.recovery = 'po podjeździe ŝ wciąż poza R(q)'
            return BT.FAILURE
        return BT.SUCCESS


class Pregrasp(Timed):
    """
    Chwytak w pozycji przed chwytem; z chwytem - pomiar z bliska i najwyżej jedna korekta.

    Gdy pomiar przesunie cel (medianę pomiarów z bliska) o > REFINE_TOL, pozycja przed chwytem
    jest powtarzana dla nowego celu, najwyżej MAX_REFINE_CORRECTIONS razy. Zawsze SUCCESS -
    wynik w Report.pregrasp_reached.
    """

    def __init__(self, name: str, m: 'Mission', refine: bool):
        super().__init__(name, m)
        self.refine = refine

    def initialise(self):
        super().initialise()
        self.stage = 'move'
        self.corrections = 0
        if self.m.pending_report.sample_pose is None:   # brak pozy z nadgarstka
            self.stage = 'done'
        else:
            self.m.start_pregrasp()

    def update(self) -> BT:
        m = self.m
        r = m.pending_report
        if self.stage == 'move':
            if arm_moving(m, self.t0, 'pregrasp_timeout'):
                return BT.RUNNING
            r.pregrasp_reached = m.manip.status() == Status.SUCCEEDED
            if not (r.pregrasp_reached and self.refine):
                return BT.SUCCESS
            self.stage = 'refine'
            self.restart()
        if self.stage == 'refine':
            if self.dt < m.REFINE_DWELL:
                return BT.RUNNING
            m.refine_pose(self.t0)
            if m.refine_shifted and self.corrections < m.MAX_REFINE_CORRECTIONS:
                self.corrections += 1
                self.stage = 'move'
                self.restart()
                m.start_pregrasp()
                return BT.RUNNING
        return BT.SUCCESS


class Grasp(Timed):
    """Opuszczenie, zamknięcie, podniesienie; tylko po osiągniętej pozycji przed chwytem."""

    def initialise(self):
        super().initialise()
        self.skip = not self.m.pending_report.pregrasp_reached
        if not self.skip:
            self.m.start_grasp()

    def update(self) -> BT:
        m = self.m
        if self.skip:
            return BT.SUCCESS
        if arm_moving(m, self.t0, 'grasp_timeout'):
            return BT.RUNNING
        m.pending_report.grasped = m.manip.status() == Status.SUCCEEDED
        if not m.pending_report.grasped:
            m.event('chwyt', 'nieudany - zgłoszenie bez próbki w chwytaku')
        return BT.SUCCESS


class Reapproach(MissionBehaviour):
    """Podejście z innej strony; FAILURE po wyczerpaniu podejść albo bez innej pozy."""

    def update(self) -> BT:
        return BT.SUCCESS if self.m.try_reapproach() else BT.FAILURE


class Abandon(MissionBehaviour):

    def update(self) -> BT:
        self.m.abandon(self.m.recovery or 'nieudana obsługa kandydata')
        self.m.recovery = None
        return BT.SUCCESS


class Publish(MissionBehaviour):

    def update(self) -> BT:
        self.m.publish_report()
        return BT.SUCCESS


class ReportSequence(py_trees.composites.Sequence):
    """Sekwencja zgłoszenia; przy wejściu - zgłoszenie w przygotowaniu (faza REPORT)."""

    def __init__(self, name: str, m: 'Mission', children):
        super().__init__(name, True, children)
        self.m = m

    def initialise(self):
        self.m.enter_report()


# ---------------------------------------------------------------- budowa drzewa
def _seq(name, *children, memory=True):
    return py_trees.composites.Sequence(name, memory, list(children))


def _sel(name, *children, memory=False):
    return py_trees.composites.Selector(name, memory, list(children))


def _report(m: 'Mission'):
    children = []
    if m.manip is not None:
        children.append(Pregrasp('Przed chwytem', m, refine=m.p.wrist.grasp))
        if m.p.wrist.grasp:
            children.append(Grasp('Chwyt i podniesienie', m))
    children.append(Publish('Publikacja zgłoszenia', m))
    return ReportSequence('Zgłoszenie', m, children)


def build_tree(m: 'Mission') -> py_trees.behaviour.Behaviour:
    verify = (WristScan('Skan weryfikujący', m) if m.manip is not None
              else FrontSettle('Ustalenie się kamery przedniej', m))
    pursue = _seq('Kandydat → zgłoszenie',
                  DriveToCandidate('Dojazd do kandydata', m),
                  verify,
                  Condition('P_rep ≥ τ_n?', m, lambda m: m.report_certain()),
                  _sel('Zasięg',
                       Condition('ŝ ∈ R(q)?', m, lambda m: m.in_reach),
                       MoveIntoReach('Podjazd w zasięg', m),
                       memory=True),
                  _report(m))
    candidate = _seq(
        'Kandydat',
        Condition('Śledzony kandydat?', m, lambda m: m.target is not None),
        _sel('Obsługa kandydata',
             pursue,
             Reapproach('Podejście z innej strony', m),
             Abandon('Porzucenie kandydata', m)),
        memory=False)
    search = _seq('Przeszukiwanie',
                  ObservationResult('Wynik obserwacji', m),
                  Plan('Planowanie', m),
                  memory=False)
    return _sel('Misja', MissionOver('Koniec misji?', m), candidate, search)


def tree_ascii(root, show_status: bool = True) -> str:
    """Drzewo jako tekst (do logów i dokumentacji)."""
    return py_trees.display.unicode_tree(root, show_status=show_status)
