"""
Logika misji: przeszukiwanie, dojazd, weryfikacja, zgłoszenie (DESIGN §3.3–3.5, PDR §4).

Przepływ sterowania jest drzewem zachowań (`mission_tree`, py_trees); ta klasa to kontekst
misji - belief, śledzony kandydat, zgłoszenia - i operacje, które wołają węzły drzewa.
Fazy (numeracja jak w ig_search_interfaces/MissionStatus, pole `state`):

  SEARCH   - strategia wybiera akcję: obserwacja z pozy albo dojazd do kandydata;
  APPROACH - jazda do pozy dojazdu (z nadgarstkiem kończona, gdy tylko ŝ jest w obszarze
             chwytu - bez ustawiania łazika centralnie); kandydat śledzony między replanami,
             porzucany, gdy P_k spadnie poniżej p_abandon; strategia może go wywłaszczyć;
  VERIFY   - skan kamerą na nadgarstku (albo krótkie ustalenie się kamery przedniej),
             potem reguła zgłoszenia P_rep ≥ τ_n; przy niepewności - podejście z innej strony;
  REPORT   - chwytak w pozycji przed chwytem, chwyt i podniesienie próbki (wrist.grasp),
             publikacja zgłoszenia, wyzerowanie beliefu w otoczeniu ŝ; próbka trafia
             z ramieniem do pozycji transportowej; po M zgłoszeniach DONE;
  DONE / TIMEOUT - koniec (M zgłoszeń, brak sensownych akcji albo T_max).

Wszystkie wywołania portów są nieblokujące; `step()` jest wołane cyklicznie (np. 5 Hz)
i tyka drzewo raz, `on_front_frame` / `on_wrist_frame` - dla każdej klatki detektora
(także pustej).
"""
from dataclasses import dataclass, field, replace
from enum import IntEnum
import math
from typing import Callable, List, Optional

from ig_search.application.mission_tree import build_tree, tree_ascii
from ig_search.application.ports import (
    Frame, Manipulator, Navigator, ReachGoal, Report, Reporter, SamplePose, Status)
from ig_search.application.world import SearchWorld
from ig_search.domain.candidates import (
    attach_approach_poses, Candidate, find_candidates, match_candidate)
from ig_search.domain.report_rule import (
    check_report, DZ_FLAT, in_workspace, reach_ring_width, threshold, workspace_polar,
    workspace_r_max)
from ig_search.domain.strategies.base import (
    Action, APPROACH, OBSERVE, PlanningContext, Strategy)
import numpy as np


class State(IntEnum):
    SEARCH = 0
    APPROACH = 1
    VERIFY = 2
    REPORT = 3
    DONE = 4
    TIMEOUT = 5


@dataclass
class Event:
    t: float                # [s] od startu misji
    kind: str
    detail: str


@dataclass
class Target:
    """Śledzony kandydat podczas dojazdu i weryfikacji."""

    x: float
    y: float
    mass: float
    app_yaw: float = 0.0            # kierunek bieżącego podejścia
    goal: tuple = (0.0, 0.0, 0.0)   # wysłana poza dojazdu
    goal_for: tuple = (0.0, 0.0)    # ŝ, dla której ją wyliczono
    excluded: List[float] = field(default_factory=list)
    attempts: int = 0
    reach_move: bool = False        # pewna, ale poza zasięgiem - podjazd bliżej bez weryfikacji
    raw: bool = False               # surowa detekcja (chase): do pierwszej weryfikacji nie
    #                                 porzucana przez belief


TRACK_RADIUS = 1.0          # [m] kandydat po replanie = ten sam cel
RAW_MERGE = 0.5             # [m] surowe detekcje bliżej niż tyle - jedna (średnia położeń)
RETARGET_DISTANCE = 0.3     # [m] przesunięcie ŝ wymuszające nową pozę dojazdu
SETTLE_TIME = 1.5           # [s] bez nadgarstka: czas na klatki kamery przedniej po dojeździe
SCAN_DWELL = 2.0            # [s] kamera nadgarstka nieruchomo nad ŝ po dojściu ramienia do pozy
#  skanu (gdy próbka wchodzi w kadr dopiero na końcu ruchu, bez tego brak pozy z nadgarstka
#  i zgłoszenie bez pozycji przed chwytem)
BLOCK_RADIUS = 0.5          # [m] porzucony kandydat - ignorowany w tym promieniu
UNBLOCK_MARGIN = 0.2        # ... chyba że P_k wzrośnie o tyle (nowe dowody), najwyżej do 0,99
STOW_TIMEOUT = 20.0         # [s]
# [m] podjazd w zasięg krótszy niż tyle (radialnie, ŝ w kącie R(q)) - ramię zostaje w pozie
# skanu nad próbką i stamtąd idzie od razu do pozycji przed chwytem (bez składania do pozycji
# transportowej i zamykania chwytaka); dłuższy - ramię złożone na czas jazdy
ARM_OUT_REACH_MOVE = 0.5
NAV_TIMEOUT_FACTOR = 3.0    # watchdog: cel trwa dłużej niż 3·T_przewidywany + 60 s
NAV_TIMEOUT_MARGIN = 60.0   # (Nav2 potrafi krążyć tuż za tolerancją celu bez porażki)
NAV_NEAR_ENOUGH = 1.0       # [m] obserwacja przerwana tak blisko celu = osiągnięta
STUCK_WINDOW = 30.0         # [s] utknięcie: w tym oknie droga estymowanej pozy ≥ STUCK_PATH,
STUCK_PATH = 3.0            # [m]   a przesunięcie netto ≤ STUCK_NET - koła buksują (odometria
STUCK_NET = 0.5             # [m]   liczy ruch, GNSS ściąga pozę z powrotem); w Gazebo łazik
#                             zawisł na skale i 26 min „jechał” w miejscu; okno 20 s / 2 m
#                             dawało fałszywe alarmy przy jeździe slalomem po trudnym terenie
#                             (poza z opóźnioną korektą GNSS zaniża przesunięcie netto o ~1 m)
STUCK_BACKUP = 0.8          # [m] cofnięcie po utknięciu
# brak postępu: cel nawigacji aktywny od STALL_WINDOW s, a przesunięcie netto w tym oknie
# ≤ STALL_NET - też utknięcie. Z lokalizacją z Gazebo (prawdziwa poza) droga z odometrii nie
# rośnie, gdy koła buksują: w świecie z kamieniami łazik osiadł podwoziem na kamieniu i przez
# 16 min dostawał kolejne cele (przejeżdżał ~2 m na 100 s), a powyższe kryterium milczało
STALL_WINDOW = 60.0         # [s]
STALL_NET = 0.5             # [m]
# kolejne nieudane nawigacje bez ruchu łazika (np. Nav2 „Start occupied” - łazik przy brzegu
# warstwy statycznej): po tylu w oknie FAIL_WINDOW, z przesunięciem ≤ FAIL_NET, odzyskanie jak
# po utknięciu. W Gazebo misja wybrała 2615 kolejnych celów w 590 s, łazik stał 0,15 m od brzegu
FAILS_IN_PLACE = 3
FAIL_WINDOW = 30.0          # [s]
FAIL_NET = 0.3              # [m]
STUCK_BLOCK_RADIUS = 0.8    # [m] miejsce utknięcia wyłączone z grafu przejezdności
VIEW_COOLDOWN = 60.0        # [s] osiągnięta poza obserwacji nie jest wybierana ponownie
VIEW_COOLDOWN_RADIUS = 0.6  # [m] ... ani pozy o tym samym kursie w tym promieniu
#  (Nav2 kończy cel z tolerancją 0,4 m / 0,5 rad - poza tuż obok łazika jest „osiągnięta”
#  bez ruchu, więc przewidywana obserwacja nigdy nie następuje)
WRIST_POSE_MAX_AGE = 5.0    # [s]
REFINE_DWELL = 1.5          # [s] kamera nadgarstka nad próbką w pozycji przed chwytem
REFINE_TOL = 0.01           # [m] nowy pomiar dalej niż tyle -> poprawiona pozycja przed chwytem
MAX_REFINE_CORRECTIONS = 2  # korekty pozycji przed chwytem; cel - mediana wszystkich pomiarów
#  z bliska (pojedyncze pomiary różnią się o 2-4 cm, Δz ±1,6 cm - korekta do ostatniego
#  pomiaru chybiała w Gazebo o 2,5 cm)
#  (szczęki rozwarte na 6,5 cm obejmują próbkę ⌀3 cm przy błędzie w poprzek do ±1,75 cm;
#  pomiar ze skanu z 0,5 m myli się o kilka cm, z pozycji przed chwytem - z ~0,3 m)


class Mission:

    SETTLE_TIME = SETTLE_TIME
    SCAN_DWELL = SCAN_DWELL
    REFINE_DWELL = REFINE_DWELL
    MAX_REFINE_CORRECTIONS = MAX_REFINE_CORRECTIONS

    def __init__(self, world: SearchWorld, strategy: Strategy, nav: Navigator,
                 reporter: Reporter, manip: Optional[Manipulator] = None,
                 log: Optional[Callable[[str], None]] = None, prior_weights=None):
        self.w = world
        self.p = world.params
        self.strategy = strategy
        self.nav = nav
        self.reporter = reporter
        self.manip = manip if self.p.mission.use_wrist else None
        self._log = log or (lambda msg: None)
        self._prior = world.prior(prior_weights)
        self.belief = world.new_belief(prior_weights)
        self.state = State.SEARCH
        self.action: Optional[Action] = None
        self.target: Optional[Target] = None
        self.n_reported = 0
        self.reports: List[Report] = []
        self.events: List[Event] = []
        self.blocked: List[tuple] = []              # (x, y, P_k przy porzuceniu, do kiedy)
        self.t0 = None
        self.now = 0.0
        self.q = (0.0, 0.0, 0.0)
        self.detail = ''
        self.recovery: Optional[str] = None         # przyczyna nieudanej obsługi kandydata
        #                                             (do zdarzeń ponownego podejścia i porzucenia)
        self._in_reach = False                      # ŝ ∈ R(q) z ostatniej reguły zgłoszenia
        self.pending_report: Optional[Report] = None
        self.refine_shifted = False                 # pomiar z bliska przesunął próbkę > REFINE_TOL
        self._close_poses: List[SamplePose] = []    # pomiary z bliska bieżącego zgłoszenia
        self._replan = True
        self._last_plan = -math.inf
        self._stowing_since = None
        self._arm_out = False                       # ramię poza pozycją transportową
        self._carrying = False                      # chwyt i przenoszenie próbki: kamery widzą
        #                                             ją w chwytaku - klatki pomijane
        self._wrist_poses: List[tuple] = []         # (stamp, SamplePose)
        self._still_sp = None                       # poza próbki z nieruchomej kamery (skan)
        self._report_decision = None                # (x, y, P_rep, τ) z reguły zgłoszenia
        self._pending_goal = None
        self._nav_t0 = 0.0
        self._track: List[tuple] = []               # (t, x, y, droga) z ostatnich STALL_WINDOW s
        self._backing = False                       # cofanie po utknięciu (cele odkładane)
        self.stuck_points: List[tuple] = []         # (x, y) miejsc utknięcia
        self._fails: List[tuple] = []               # (czas, x, y) nieudanych nawigacji
        self._odo = 0.0                             # [m] droga łazika (z kolejnych q)
        self._odo_goal = 0.0                        # ... w chwili wysłania celu nawigacji
        self.frames = {'front': 0, 'wrist': 0}
        self._view_until = np.full(len(world.views), -np.inf)
        self._raw: List[list] = []                  # [x, y, n] surowych detekcji (chase)
        self.tree = build_tree(self)
        self.tick: Callable[[], None] = self.tree.tick_once    # infrastruktura może podmienić
        #                                                        (np. py_trees_ros z migawkami)

    # ------------------------------------------------------------ pomocnicze
    @property
    def elapsed(self) -> float:
        return 0.0 if self.t0 is None else self.now - self.t0

    @property
    def finished(self) -> bool:
        return self.state in (State.DONE, State.TIMEOUT)

    def event(self, kind: str, detail: str):
        self.events.append(Event(self.elapsed, kind, detail))
        self.detail = detail
        self._log(f'[{self.elapsed:7.1f} s] {kind}: {detail}')

    def _finish(self, state: State, reason: str):
        self.state = state
        self.nav.cancel()
        self.event('koniec', f'{reason} (zgłoszenia: {self.n_reported})')

    def timeout(self):
        self._finish(State.TIMEOUT, 'przekroczono T_max')

    def tree_ascii(self, show_status: bool = True) -> str:
        """Drzewo zachowań ze statusami z ostatniego tyku."""
        return tree_ascii(self.tree, show_status)

    # ------------------------------------------------------------ klatki
    def _known(self, dets: np.ndarray, cam_xy, model) -> np.ndarray:
        """
        Maska detekcji przypisanych już zgłoszonym próbkom (próbka zostaje na miejscu).

        Promień ε' + 2σ(r): szum detekcji z daleka nie może odbudować beliefu tuż za
        obszarem wyłączonym po zgłoszeniu (inaczej - ponowne zgłoszenie tej samej próbki).
        """
        if not self.reports or len(dets) == 0:
            return np.zeros(len(dets), bool)
        rep = np.array([[r.x, r.y] for r in self.reports])
        r_det = np.hypot(dets[:, 0] - cam_xy[0], dets[:, 1] - cam_xy[1])
        radius = self.p.report.clear_radius + 2.0 * model.sigma(r_det)
        d = np.hypot(dets[:, None, 0] - rep[None, :, 0], dets[:, None, 1] - rep[None, :, 1])
        return (d <= radius[:, None]).any(axis=1)

    def _apply_frame(self, f: Frame, vis, model) -> float:
        dets = np.asarray(f.detections, float).reshape(-1, 3)[:, :2]
        dets = dets[~self._known(dets, f.camera.position[:2], model)]
        v = vis.visible(f.camera)
        if len(dets):
            radius = float(model.sigma(vis.cam.r_max)) * self.p.belief.detection_window_sigmas
            v = v.union(vis.cells_near(f.camera, dets, radius))
        res = self.belief.update(v, dets, f.camera.position[:2], model=model)
        if res.max_ell_gain > 1.0:
            self._replan = True                 # nowy dowód - przelicz plan
        self._last_dets = dets
        return res.max_ell_gain

    def on_front_frame(self, f: Frame):
        if self.finished or self._carrying:
            return
        self.frames['front'] += 1
        self._apply_frame(f, self.w.vis, self.w.model)
        if self.strategy.uses_raw_detections:
            self._note_raw(self._last_dets)

    def _note_raw(self, dets: np.ndarray):
        """Surowe detekcje (poza zgłoszonymi i chwilowo porzuconymi miejscami) do odwiedzenia."""
        for x, y in dets:
            if any(math.hypot(x - b[0], y - b[1]) < BLOCK_RADIUS
                   for b in self.blocked if b[3] > self.now):
                continue
            near = [r for r in self._raw if math.hypot(x - r[0], y - r[1]) < RAW_MERGE]
            if near:
                r = near[0]
                r[0] = (r[0] * r[2] + x) / (r[2] + 1)
                r[1] = (r[1] * r[2] + y) / (r[2] + 1)
                r[2] += 1
            else:
                self._raw.append([float(x), float(y), 1])
                self._replan = True

    def _drop_raw(self, x: float, y: float):
        """Detekcje obsłużonego miejsca (zgłoszenie albo porzucenie) - już nie do odwiedzenia."""
        self._raw = [r for r in self._raw if math.hypot(r[0] - x, r[1] - y) >= TRACK_RADIUS]

    def _raw_candidates(self) -> List[Candidate]:
        out = []
        p, g = self.belief.p, self.w.grid
        for x, y, _ in self._raw:
            i = int(g.flat_index(x, y))
            c = Candidate(np.empty(0, np.int64), float(p[i]) if i >= 0 else 0.0, x, y,
                          float(p[i]) if i >= 0 else 0.0, raw=True)
            out.append(attach_approach_poses(c, self.w.travel, self.p.candidates,
                                             distance=self._approach_distance(),
                                             ring=self._ring(c) if self.manip else None))
        return out

    def on_wrist_frame(self, f: Frame):
        # tylko gdy ramię celuje kamerą w próbkę (skan, pozycja przed chwytem); w pozycji
        # transportowej i w ruchu do niej kamera widzi łazik i chwytak - same fałszywe detekcje
        if (self.finished or self._carrying
                or self.state not in (State.VERIFY, State.REPORT) or self._stowing_since):
            return
        self.frames['wrist'] += 1
        # z pozycji przed chwytem (~0,3 m) tylko pomiar pozycji: kadr ±0,13 m, P_d nadgarstka
        # z tej odległości nieskalibrowane - puste klatki zbijały belief prawdziwej próbki
        # (w Gazebo P_rep 1,00 -> 0,20 i porzucenie próbki mimo detekcji przy skanie)
        if self.state == State.VERIFY:
            self._apply_frame(f, self.w.wrist_vis, self.w.wrist_model)
        self._wrist_poses = [(f.stamp, sp) for sp in f.poses] + [
            wp for wp in self._wrist_poses if f.stamp - wp[0] <= WRIST_POSE_MAX_AGE]

    # ------------------------------------------------------------ planowanie
    def _candidates(self) -> List[Candidate]:
        cands = find_candidates(self.belief, self.p.candidates)
        self.blocked = [b for b in self.blocked if b[3] > self.now]
        out = []
        for c in cands:
            blk = [b for b in self.blocked if math.hypot(c.x - b[0], c.y - b[1]) < BLOCK_RADIUS]
            if blk and c.mass < min(max(b[2] for b in blk) + UNBLOCK_MARGIN, 0.99):
                continue
            out.append(attach_approach_poses(c, self.w.travel, self.p.candidates,
                                             distance=self._approach_distance(),
                                             ring=self._ring(c) if self.manip else None))
        return out

    def _ring(self, c: Candidate):
        """Szerokość pierścienia R(q) z zapasem dla ŝ kandydata z póz (xs, ys), mapa wysokości."""
        hm = self.w.heightmap

        def width(xs, ys):
            dz = np.asarray(hm.height_at(c.x, c.y) - hm.height_at(xs, ys), float) + DZ_FLAT
            return reach_ring_width(self.p.report, np.where(np.isfinite(dz), dz, DZ_FLAT))
        return width

    def _approach_distance(self, into_reach: bool = False) -> float:
        """
        Odległość base_link -> ŝ przy dojeździe.

        Z nadgarstkiem - od razu w zasięg ramienia (skan z bliska). Bez nadgarstka - najpierw
        z odległości, z której kamera przednia widzi ŝ, potem podjazd w zasięg.
        """
        cp = self.p.candidates
        return cp.approach_distance if self.manip is not None or into_reach \
            else cp.front_verify_distance

    def _context(self) -> PlanningContext:
        drive = self.w.travel.drive_times(self.q[0], self.q[1])
        return PlanningContext(self.q, self.belief, self.w.views, self.w.travel, drive,
                               self._candidates(), self.p, self.elapsed, self.n_reported,
                               [(b[0], b[1]) for b in self.blocked],
                               self._view_until > self.now,
                               self._raw_candidates() if self.strategy.uses_raw_detections
                               else [])

    def _cool_down(self, a: Action):
        v = self.w.views
        near = ((np.hypot(v.xs - a.x, v.ys - a.y) <= VIEW_COOLDOWN_RADIUS)
                & (np.abs((v.yaws - a.yaw + math.pi) % (2 * math.pi) - math.pi) < 0.1))
        if a.view >= 0:
            near[a.view] = True
        self._view_until[near] = self.now + VIEW_COOLDOWN

    @staticmethod
    def _same(a: Action, b: Action) -> bool:
        if a.kind != b.kind:
            return False
        if a.kind == OBSERVE:
            return a.view == b.view if a.view >= 0 else (a.x, a.y) == (b.x, b.y)
        return (a.candidate is not None and b.candidate is not None
                and a.candidate.distance_to(b.candidate.x, b.candidate.y) < TRACK_RADIUS)

    def _preempts(self, best: Action, ctx: PlanningContext) -> bool:
        if self.action is None:
            return True
        if self._same(best, self.action):
            return False
        cur = self.strategy.value_of(ctx, self.action)
        return cur <= 0 or best.value > cur * (1.0 + self.p.planner.hysteresis)

    def plan_if_due(self):
        """Replanowanie po nowym dowodzie, bez akcji albo co replan_period."""
        if (self._replan or self.action is None
                or self.now - self._last_plan >= self.p.planner.replan_period):
            self._plan()

    def _plan(self):
        self._replan = False
        self._last_plan = self.now
        ctx = self._context()
        if self.target is not None:
            self._track_target(ctx)
            if self.target is None:
                return
        ctx = self._approachable(ctx)
        best = self.strategy.choose(ctx)
        if best is not None and not (math.isfinite(best.time) and best.value > 0):
            best = None                     # nieosiągalna albo bezwartościowa (T = inf, S = 0)
        if best is None:
            if self.action is None and self.target is None:
                self._finish(State.DONE, 'brak sensownych akcji '
                             f'(masa beliefu {self.belief.mass():.3f})')
            return
        if not self._preempts(best, ctx):
            return
        self._execute(best)

    def _approachable(self, ctx: PlanningContext) -> PlanningContext:
        """Do wyboru akcji: kandydaci z P_k ≥ p_approach_min (oraz bieżący cel dojazdu)."""
        t, p_min = self.target, self.p.candidates.p_approach_min
        cands = [c for c in ctx.candidates if c.mass >= p_min or (
            t is not None and c.distance_to(t.x, t.y) < TRACK_RADIUS)]
        return ctx if len(cands) == len(ctx.candidates) else replace(ctx, candidates=cands)

    def _execute(self, a: Action):
        self.action = a
        if a.kind == APPROACH:
            c = a.candidate
            self.target = Target(c.x, c.y, c.mass, a.yaw, (a.x, a.y, a.yaw), (c.x, c.y),
                                 raw=c.raw)
            self.state = State.APPROACH
        else:
            self.target = None
            self.state = State.SEARCH
        reach = (self._reach_goal(a.candidate.x, a.candidate.y, base_xy=(a.x, a.y))
                 if a.kind == APPROACH else None)
        self._go(a.x, a.y, a.yaw, precise=a.kind == APPROACH, reach=reach)
        self.event('akcja', a.detail)

    def observation_done(self, reached: bool):
        """Koniec jazdy do pozy obserwacji: wstrzymanie pozy albo jej wykluczenie."""
        a = self.action
        if not reached:
            self.strategy.on_failed(a)
            if a.view >= 0:
                self.w.views.ok[a.view] = False
            self.event('nawigacja', 'nieudana - poza wykluczona')
            self._fails = [f for f in self._fails if self.now - f[0] <= FAIL_WINDOW] + [
                (self.now, self.q[0], self.q[1])]
            if len(self._fails) >= FAILS_IN_PLACE and math.hypot(
                    self._fails[-1][1] - self._fails[0][1],
                    self._fails[-1][2] - self._fails[0][2]) <= FAIL_NET:
                self._recover(self.q[0], self.q[1],
                              f'{len(self._fails)} nieudane nawigacje w '
                              f'{self.now - self._fails[0][0]:.0f} s bez ruchu')
                self._fails = []
        else:
            self.strategy.on_reached(a)
            self._cool_down(a)
        self.action = None
        self._replan = True

    def watchdog(self, nav: Status) -> Status:
        """Przerywa cel nawigacji trwający znacznie dłużej niż przewiduje model ruchu."""
        a = self.action
        if a is None or nav != Status.ACTIVE:
            return nav
        if self._stuck():
            return Status.FAILED
        limit = NAV_TIMEOUT_FACTOR * a.time + NAV_TIMEOUT_MARGIN
        if self.now - self._nav_t0 <= limit:
            return nav
        self.nav.cancel()
        dist = math.hypot(self.q[0] - a.x, self.q[1] - a.y)
        ok = a.kind == OBSERVE and dist <= NAV_NEAR_ENOUGH
        moved = self._odo - self._odo_goal
        self.event('nawigacja', f'przerwana po {self.now - self._nav_t0:.0f} s '
                   f'(limit {limit:.0f} s), {dist:.2f} m od celu, przejechane {moved:.1f} m '
                   f'(przewidywany czas {a.time:.0f} s) - '
                   f'{"uznana za osiągniętą" if ok else "nieudana"}')
        return Status.SUCCEEDED if ok else Status.FAILED

    def _stuck(self) -> bool:
        """
        Łazik utknął (koła buksują): cofnięcie, miejsce wyłączone z grafu, cel nieudany.

        Okno liczone od wysłania celu - zatrzymanie na początku jazdy nie jest utknięciem.
        """
        tr = [p for p in self._track if p[0] >= self._nav_t0]
        if not tr or tr[-1][0] - tr[0][0] < STUCK_WINDOW:
            return False
        path = tr[-1][3] - tr[0][3]
        net = math.hypot(tr[-1][1] - tr[0][1], tr[-1][2] - tr[0][2])
        if path < STUCK_PATH or net > STUCK_NET:
            return self._stalled(tr)
        x = float(np.median([p[1] for p in tr]))
        y = float(np.median([p[2] for p in tr]))
        self._recover(x, y, f'droga {path:.1f} m, przesunięcie {net:.2f} m w '
                            f'{STUCK_WINDOW:.0f} s')
        return True

    def _stalled(self, tr) -> bool:
        """Brak postępu: przez STALL_WINDOW s jazdy przesunięcie netto ≤ STALL_NET."""
        if self.now - self._nav_t0 < STALL_WINDOW:
            return False
        w = [p for p in tr if p[0] >= tr[-1][0] - STALL_WINDOW]
        if not w or w[-1][0] - w[0][0] < 0.9 * STALL_WINDOW:
            return False
        xs, ys = [p[1] for p in w], [p[2] for p in w]
        if max(math.hypot(x - xs[0], y - ys[0]) for x, y in zip(xs, ys)) > STALL_NET:
            return False
        self._recover(float(np.median(xs)), float(np.median(ys)),
                      f'brak postępu: przesunięcie ≤ {STALL_NET} m w {STALL_WINDOW:.0f} s jazdy')
        return True

    def _recover(self, x, y, why: str):
        """Cofnięcie, miejsce wyłączone z grafu przejezdności i z póz obserwacji."""
        n = self.w.travel.block(x, y, STUCK_BLOCK_RADIUS)
        v = self.w.views
        self.w.views.ok[np.hypot(v.xs - x, v.ys - y) <= STUCK_BLOCK_RADIUS] = False
        self.stuck_points.append((x, y))
        self.nav.cancel()
        self.nav.back_up(STUCK_BACKUP)
        self._backing = True
        self._track = []
        self.event('utknięcie', f'w ({x:.2f}, {y:.2f}): {why} - cofanie {STUCK_BACKUP} m, '
                   f'{n} komórek wyłączonych z grafu przejezdności')

    # ------------------------------------------------------------ dojazd
    def state_approach(self):
        self.state = State.APPROACH

    def _track_target(self, ctx: PlanningContext):
        t = self.target
        c = match_candidate(ctx.candidates, t.x, t.y, TRACK_RADIUS)
        if t.raw and (c is None or c.mass < self.p.candidates.p_abandon):
            return                          # surowa detekcja: o porzuceniu decyduje weryfikacja
        if c is None or c.mass < self.p.candidates.p_abandon:
            self.abandon('kandydat zanikł' if c is None else f'P_k spadło do {c.mass:.2f}')
            return
        t.x, t.y, t.mass = c.x, c.y, c.mass
        self.action.candidate = c
        if math.hypot(c.x - t.goal_for[0], c.y - t.goal_for[1]) > RETARGET_DISTANCE:
            self._send_approach(c, ctx)

    def _send_approach(self, c: Candidate, ctx: PlanningContext = None) -> bool:
        t = self.target
        reach = self.manip is not None or t.reach_move
        attach_approach_poses(c, self.w.travel, self.p.candidates, t.excluded,
                              distance=self._approach_distance(t.reach_move),
                              ring=self._ring(c) if reach else None)
        ctx = ctx or self._context()
        a = ctx.approach_action(c, self.action.value if self.action else 0.0)
        if a is None:
            return False
        t.app_yaw, t.goal, t.goal_for = a.yaw, (a.x, a.y, a.yaw), (c.x, c.y)
        self.action = a
        self._go(a.x, a.y, a.yaw, precise=True,
                 reach=self._reach_goal(c.x, c.y, stop_early=t.attempts == 0,
                                        base_xy=(a.x, a.y)))
        self.event('dojazd', f'cel ({a.x:.2f}, {a.y:.2f}, {math.degrees(a.yaw):.0f}°) '
                   f'do ŝ=({c.x:.2f}, {c.y:.2f}) P_k={c.mass:.2f}')
        return True

    def try_reapproach(self) -> bool:
        """
        Nowe podejście z innej strony (bramka różnorodności doceni nowy punkt widzenia).

        False - wyczerpano podejścia albo brak innej pozy dojazdu (przyczyna w `recovery`).
        """
        t = self.target
        why = self.recovery or 'nieudana obsługa kandydata'
        t.attempts += 1
        if t.attempts > self.p.report.max_verify_attempts:
            self.recovery = f'{why}; wyczerpano podejścia'
            return False
        t.excluded.append(t.app_yaw)
        t.reach_move = False
        # skupisko z beliefu (pozy dojazdu omijają jego komórki), bez niego - sam punkt ŝ
        c = match_candidate(find_candidates(self.belief, self.p.candidates), t.x, t.y,
                            TRACK_RADIUS) or Candidate(np.empty(0, np.int64), t.mass, t.x, t.y,
                                                       t.mass)
        c.x, c.y = t.x, t.y
        self.state = State.APPROACH
        self._stow()
        if not self._send_approach(c):
            self.recovery = f'{why}; brak innej pozy dojazdu'
            return False
        self.event('ponowne podejście', why)
        self.recovery = None
        return True

    def start_reach_move(self) -> bool:
        """
        Próbka pewna, ale poza R(q): najmniejszy ruch, który wprowadza ŝ w R(q).

        Z nadgarstkiem bez nowej pozy dojazdu - nawigator koryguje tylko położenie ŝ względem
        podstawy ramienia (zwykle krótki odcinek prosto). Bez nadgarstka - poza dojazdu
        w odległości chwytu. False - brak takiej pozy (przyczyna w `recovery`).
        """
        t = self.target
        why = self.recovery
        t.reach_move = True
        self.state = State.APPROACH
        if self.manip is None or self._reach_shift(t.x, t.y) > ARM_OUT_REACH_MOVE:
            self._stow()
        else:
            why = f'{why}; ramię zostaje nad próbką'
        if self.manip is not None:
            t.goal_for = (t.x, t.y)
            reach = ReachGoal(t.x, t.y, self.p.report, True, dz=self._dz(t.x, t.y))
            self._go(*self.q, precise=True, reach=reach)
        elif not self._send_approach(Candidate(np.empty(0, np.int64), t.mass, t.x, t.y,
                                               t.mass)):
            self.recovery = f'{why}; brak pozy w zasięgu ramienia'
            return False
        self.event('podjazd w zasięg', why)
        return True

    def _reach_shift(self, x, y) -> float:
        """Przesunięcie wzdłuż promienia, po którym (x, y) wejdzie w R(q); inf poza kątem."""
        rp = self.p.report
        r, bearing = (float(v[0]) for v in workspace_polar(self.q, [x], [y], rp))
        if abs(bearing) > math.radians(rp.workspace_half_angle_deg - rp.reach_angle_margin_deg):
            return math.inf
        r_max = float(workspace_r_max(rp, self._dz(x, y))) - rp.reach_margin
        return max(r - r_max, rp.workspace_r_min + rp.reach_margin - r, 0.0)

    def target_in_reach(self) -> bool:
        """ŝ w R(q) z bieżącej pozy (po podjeździe w zasięg)."""
        t = self.target
        return bool(in_workspace(self.q, [t.x], [t.y], self.p.report, dz=self._dz(t.x, t.y))[0])

    def abandon(self, why: str):
        t = self.target
        if t is not None:
            self.blocked.append((t.x, t.y, t.mass, self.now + self.p.candidates.block_time))
            self._drop_raw(t.x, t.y)
        self.event('porzucenie', why)
        self.target = None
        self.action = None
        self.state = State.SEARCH
        self.nav.cancel()
        self._pending_goal = None
        self._stow()
        self._replan = True

    # ------------------------------------------------------------ nawigacja i ramię
    def _go(self, x, y, yaw, precise=False, reach=None):
        """Cel nawigacji; podczas składania ramienia i cofania odkładany do ich zakończenia."""
        if self._stowing_since is not None or self._backing:
            self._pending_goal = (x, y, yaw, precise, reach)
        else:
            self._pending_goal = None
            self._nav_t0 = self.now
            self._odo_goal = self._odo
            self.nav.go_to(x, y, yaw, precise, self._via(x, y), reach)

    def drive_blocked(self) -> bool:
        """
        Jazda jeszcze nie ruszyła.

        Ramię jedzie do pozycji transportowej albo w tym kroku wysłano odłożony na ten czas
        cel nawigacji (status nawigacji dotyczy jeszcze poprzedniego celu).
        """
        if self.arm_busy():
            return True
        if self._backing:
            if self.nav.status() == Status.ACTIVE:
                return True
            self._backing = False
        if self._pending_goal is None:
            return False
        x, y, yaw, precise, reach = self._pending_goal
        self.nav.go_to(x, y, yaw, precise, self._via(x, y), reach)
        self._pending_goal = None
        self._nav_t0 = self.now
        self._odo_goal = self._odo
        return True

    def _dz(self, x, y, base_xy=None) -> float:
        """Wysokość środka próbki w (x, y) względem base_link łazika w base_xy (mapa wysokości)."""
        bx, by = base_xy if base_xy is not None else self.q[:2]
        hm = self.w.heightmap
        dz = float(hm.height_at(x, y) - hm.height_at(bx, by)) + DZ_FLAT
        return dz if math.isfinite(dz) else DZ_FLAT

    def _reach_goal(self, x, y, stop_early=True, base_xy=None) -> Optional[ReachGoal]:
        """
        Dojazd kończy się z ŝ w obszarze chwytu (tylko z ramieniem - zasięg ma sens).

        Ponowne podejście (stop_early=False) jedzie do nowej pozy: ma dać inny punkt
        widzenia, a ŝ zwykle już jest w R(q). Zasięg liczony dla łazika w base_xy
        (poza dojazdu; domyślnie bieżąca).
        """
        t = self.target
        if self.manip is None and not (t is not None and t.reach_move):
            return None
        return ReachGoal(x, y, self.p.report, stop_early=stop_early,
                         dz=self._dz(x, y, base_xy))

    def _via(self, x, y):
        """Punkty pośrednie po grafie przejezdności (Nav2 nie zna nachylenia terenu)."""
        return self.w.travel.waypoints(self.q[0], self.q[1], x, y)

    def _stow(self):
        if self.manip is not None and self._arm_out:
            self.manip.stow()
            self._stowing_since = self.now
            self._arm_out = False

    def arm_busy(self) -> bool:
        """Ramię w drodze do pozycji transportowej (najwyżej STOW_TIMEOUT)."""
        if self._stowing_since is None:
            return False
        if self.manip.status() == Status.ACTIVE and self.now - self._stowing_since < STOW_TIMEOUT:
            return True
        self._stowing_since = None
        self._carrying = False
        return False

    def _sample_pose_near(self, x, y) -> Optional[SamplePose]:
        best, d_best = None, self.p.wrist.match_radius
        for _, sp in self._wrist_poses:
            d = math.hypot(sp.position[0] - x, sp.position[1] - y)
            if d <= d_best:
                best, d_best = sp, d
        return best

    def _still_pose(self, x, y, since: float):
        """
        Poza próbki z klatek nadgarstka od chwili `since` (ramię nieruchomo) w pobliżu (x, y).

        Mediana pozycji (odporna na pojedyncze złe detekcje); oś z detekcji najbliższej
        medianie. Klatki z ruchu ramienia są pomijane - poza kamery z TF jest wtedy mniej
        dokładna (w Gazebo błędy do 6 cm). Zwraca (SamplePose albo None, liczba detekcji).
        """
        near = [p for t, p in self._wrist_poses if t >= since and math.hypot(
            p.position[0] - x, p.position[1] - y) <= self.p.wrist.match_radius]
        if not near:
            return None, 0
        med = np.median([p.position for p in near], axis=0)
        best = min(near, key=lambda p: float(np.linalg.norm(p.position - med)))
        local = None
        if all(p.local_position is not None for p in near):
            local = np.median([p.local_position for p in near], axis=0)
        return SamplePose(med, best.axis, local, best.local_axis), len(near)

    # ------------------------------------------------------------ weryfikacja
    def enter_verify(self):
        self.state = State.VERIFY
        self._still_sp = None
        t = self.target
        t.raw = False                       # po weryfikacji na miejscu - jak każdy kandydat
        if self.manip is None:
            self.event('weryfikacja', f'kamera przednia ŝ=({t.x:.2f}, {t.y:.2f})')

    def start_scan(self):
        t = self.target
        hm = self.w.heightmap
        dz = float(hm.height_at(t.x, t.y) - hm.height_at(self.q[0], self.q[1]))
        self.manip.scan(t.x, t.y, dz if math.isfinite(dz) else 0.0)
        self._arm_out = True
        self.event('weryfikacja', f'skan nadgarstkiem ŝ=({t.x:.2f}, {t.y:.2f})')

    def measure_scan(self, since: float):
        """Po postoju kamery nad ŝ: poza próbki z nieruchomego nadgarstka."""
        t = self.target
        self._still_sp, n = self._still_pose(t.x, t.y, since)
        self.event('skan', f'{n} detekcji z nieruchomej kamery nadgarstka')

    def report_certain(self) -> bool:
        """
        Reguła zgłoszenia, część probabilistyczna: P_rep ≥ τ_n (DESIGN §3.5).

        Doprecyzowuje ŝ (z nadgarstka, gdy jest poza z nieruchomej kamery) i zapamiętuje, czy
        ŝ ∈ R(q) - zasięg sprawdza osobny warunek drzewa. Przy False przyczyna w `recovery`.
        """
        t = self.target
        chk = check_report(self.belief, self.q, t.x, t.y, self.p.report, dz=self._dz(t.x, t.y))
        tau = threshold(self.n_reported, self.p.report)
        t.x, t.y = chk.x, chk.y
        self._in_reach = chk.in_reach
        sp = self._still_sp or self._sample_pose_near(chk.x, chk.y)
        if sp is not None:                  # chwyt celuje w pozycję z nadgarstka (σ ≈ 2 cm)
            t.x, t.y = float(sp.position[0]), float(sp.position[1])
            self._in_reach = self.target_in_reach()
        self._report_decision = (chk.x, chk.y, chk.probability, tau)
        if chk.probability < tau:
            self.recovery = f'P_rep={chk.probability:.2f} < τ={tau:.2f}'
            return False
        if self.manip is not None and sp is None:
            # weryfikacja z bliska nic nie widziała: P_rep tylko z kamery przedniej, której
            # fałszywe alarmy (ten sam kamień z kilku stron) nie są niezależne - bez zgłoszenia
            self.recovery = (f'P_rep={chk.probability:.2f}, ale brak detekcji z nadgarstka '
                             f'przy ŝ')
            return False
        if not self._in_reach:
            r = math.hypot(t.x - self.q[0], t.y - self.q[1])
            b = math.degrees((math.atan2(t.y - self.q[1], t.x - self.q[0]) - self.q[2]
                              + math.pi) % (2 * math.pi) - math.pi)
            self.recovery = (f'ŝ poza zasięgiem ramienia: {r:.2f} m, {b:.0f}° '
                             f'(P_rep={chk.probability:.2f})')
        return True

    @property
    def in_reach(self) -> bool:
        """ŝ ∈ R(q) według ostatniej reguły zgłoszenia."""
        return self._in_reach

    # ------------------------------------------------------------ zgłoszenie i chwyt
    def enter_report(self):
        self.state = State.REPORT
        x, y, prob, tau = self._report_decision
        sp = self._still_sp or self._sample_pose_near(x, y)
        if sp is not None:                  # nadgarstek: dokładniejsza pozycja (σ ≈ 2 cm)
            x, y, z = (float(v) for v in sp.position)
        else:
            z = float(self.w.heightmap.height_at(x, y)) + self.p.camera.target_height
        self.pending_report = Report(self.n_reported + 1, x, y, z, prob, tau, self.q,
                                     self.now, sp, False)
        self.refine_shifted = False
        self._close_poses = []

    def _with_ground(self, sp: SamplePose) -> SamplePose:
        """Poza próbki z wysokością gruntu pod nią względem gruntu pod łazikiem (do chwytu)."""
        hm = self.w.heightmap
        dz = float(hm.height_at(sp.position[0], sp.position[1])
                   - hm.height_at(self.q[0], self.q[1]))
        return replace(sp, ground_dz=dz) if math.isfinite(dz) else sp

    def start_pregrasp(self):
        self.manip.pregrasp(self._with_ground(self.pending_report.sample_pose))
        self._arm_out = True

    def refine_pose(self, since: float):
        """
        Pomiar z bliska z pozycji przed chwytem; `refine_shifted` - potrzebna korekta.

        Cel chwytu to mediana wszystkich pomiarów z bliska tego zgłoszenia (każdy pomiar to
        już mediana detekcji z nieruchomej kamery), a nie ostatni pomiar.
        """
        r = self.pending_report
        old = r.sample_pose.position
        meas, n = self._still_pose(old[0], old[1], since)
        self.refine_shifted = False
        if meas is None:
            self.event('chwyt', 'pomiar z bliska: brak detekcji - chwyt z poprzedniej pozycji')
            return
        self._close_poses.append(meas)
        new = self._fuse(self._close_poses)
        prev = r.sample_pose
        if new.local_position is not None and prev.local_position is not None:
            # ramka odometrii: porównanie bez skoków korekty GNSS między pomiarami
            shift = float(np.linalg.norm(new.local_position[:2] - prev.local_position[:2]))
        else:
            shift = float(np.linalg.norm(new.position[:2] - old[:2]))
        r.sample_pose = new
        r.x, r.y, r.z = (float(v) for v in new.position)
        dz = 100 * (new.position[2] - old[2])
        self.event('chwyt', f'pomiar z bliska #{len(self._close_poses)}: {n} detekcji, '
                            f'cel (mediana pomiarów) przesunięty o {100 * shift:.1f} cm, '
                            f'Δz {dz:+.1f} cm')
        self.refine_shifted = shift > REFINE_TOL

    @staticmethod
    def _fuse(poses: List[SamplePose]) -> SamplePose:
        """Mediana pozycji (i pozycji w ramce odometrii); oś z pomiaru najbliższego medianie."""
        med = np.median([p.position for p in poses], axis=0)
        best = min(poses, key=lambda p: float(np.linalg.norm(p.position - med)))
        local = None
        if all(p.local_position is not None for p in poses):
            local = np.median([p.local_position for p in poses], axis=0)
        return SamplePose(med, best.axis, local, best.local_axis)

    def start_grasp(self):
        self._carrying = True
        self.manip.grasp(self._with_ground(self.pending_report.sample_pose))

    def publish_report(self):
        r = self.pending_report
        r.stamp = self.now
        self.pending_report = None
        self.reporter.report(r)
        self.reports.append(r)
        self.n_reported += 1
        # resztki skupiska tej próbki poza ε' -> a priori (inaczej wyglądają jak kolejna próbka)
        blob = match_candidate(find_candidates(self.belief, self.p.candidates), r.x, r.y,
                               TRACK_RADIUS)
        if blob is not None:
            self.belief.reset_to_prior(blob.cells, self._prior)
        self.belief.clear(r.x, r.y, self.p.report.clear_radius)
        self._drop_raw(r.x, r.y)
        self.event('zgłoszenie', f'#{r.index} ŝ=({r.x:.2f}, {r.y:.2f}) P_rep={r.probability:.2f}'
                   f' τ={r.threshold:.2f} przed chwytem={r.pregrasp_reached}'
                   f' chwycona={r.grasped}')
        self.target = None
        self.action = None
        self._stow()
        if self.n_reported >= self.p.belief.n_samples:
            self._finish(State.DONE, 'zgłoszono wszystkie próbki')
            return
        self.state = State.SEARCH
        self._replan = True

    # ------------------------------------------------------------ pętla
    def step(self, now: float, q: tuple):
        """Jeden tyk drzewa misji; q = (x, y, psi) łazika w ramce map."""
        self.now = now
        self._odo += math.hypot(q[0] - self.q[0], q[1] - self.q[1]) if self.t0 is not None else 0.0
        self.q = q
        keep = max(STUCK_WINDOW, STALL_WINDOW)
        self._track = [p for p in self._track if now - p[0] <= keep] + [
            (now, q[0], q[1], self._odo)]
        if self.t0 is None:
            self.t0 = now
            self.event('start', f'strategia {self.strategy.name}, '
                       f'masa beliefu {self.belief.mass():.2f}')
            if self.manip is not None:      # ramię do pozycji transportowej przed jazdą
                self._arm_out = True
                self._stow()
        self.tick()

    def status(self) -> dict:
        return {'state': int(self.state), 'strategy': self.strategy.name,
                'reports': self.n_reported, 'elapsed': self.elapsed,
                'belief_mass': self.belief.mass(), 'belief_entropy': self.belief.entropy(),
                'detail': self.detail}
