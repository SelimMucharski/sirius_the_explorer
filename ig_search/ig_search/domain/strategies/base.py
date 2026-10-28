"""
Wspólny port strategii wyboru akcji i kontekst planowania (DESIGN §3.4, §4.2). Bez ROS.

Wszystkie strategie dzielą belief, model detektora, regułę zgłoszenia i logikę misji;
różnią się tylko wyborem następnej akcji: obserwacji z pozy albo dojazdu do kandydata.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from ig_search.domain.belief import Belief
from ig_search.domain.candidates import Candidate
from ig_search.domain.params import SearchParams
from ig_search.domain.travel import TravelModel
from ig_search.domain.views import ViewSet
import numpy as np

OBSERVE = 'observe'
APPROACH = 'approach'


@dataclass
class Action:
    kind: str                       # OBSERVE | APPROACH
    x: float
    y: float
    yaw: float
    value: float                    # ocena w jednostkach strategii (do histerezy)
    time: float                     # przewidywany czas dotarcia [s]
    view: int = -1                  # indeks pozy w ViewSet (OBSERVE)
    candidate: Optional[Candidate] = None
    detail: str = ''
    plan: tuple = ()                # zaplanowana sekwencja póz (x, y, yaw), od tej akcji


@dataclass
class PlanningContext:
    q: tuple                        # (x, y, psi) łazika
    belief: Belief
    views: ViewSet
    travel: TravelModel
    drive: np.ndarray               # czas jazdy z q do każdej komórki
    candidates: List[Candidate]     # z dołączonymi pozami dojazdu
    params: SearchParams
    elapsed: float = 0.0            # [s] od startu misji
    n_reported: int = 0
    blocked_candidates: List[tuple] = field(default_factory=list)  # (x, y) porzucone
    view_blocked: np.ndarray = None  # pozy chwilowo wyłączone (np. właśnie osiągnięte)
    # surowe detekcje kamery przedniej jako kandydaci z pozami dojazdu (raw = True); tylko
    # dla strategii z uses_raw_detections
    raw_candidates: List[Candidate] = field(default_factory=list)

    def view_times(self) -> np.ndarray:
        """Czas dotarcia do każdej pozy obserwacji; inf dla wykluczonych i nieosiągalnych."""
        v = self.views
        T = self.travel.pose_times(self.q, v.xs, v.ys, v.yaws, self.drive, v.cells)
        T[~v.ok | ~np.isfinite(T)] = np.inf
        if self.view_blocked is not None:
            T[self.view_blocked] = np.inf
        return T

    def approach_time(self, cand: Candidate):
        """(indeks najlepszej pozy dojazdu, czas) albo (-1, inf)."""
        if len(cand.app_cell) == 0:
            return -1, np.inf
        T = self.travel.pose_times(self.q, cand.app_x, cand.app_y, cand.app_yaw, self.drive,
                                   cand.app_cell)
        k = int(np.argmin(T))
        return k, float(T[k])

    def approach_action(self, cand: Candidate, value: float) -> Optional[Action]:
        k, T = self.approach_time(cand)
        if k < 0 or not np.isfinite(T):
            return None
        return Action(APPROACH, float(cand.app_x[k]), float(cand.app_y[k]),
                      float(cand.app_yaw[k]), value, T, candidate=cand,
                      detail=f'dojazd do ŝ=({cand.x:.2f}, {cand.y:.2f}) P_k={cand.mass:.2f}')

    def observe_action(self, v: int, value: float, T: float, detail: str = '') -> Action:
        vs = self.views
        return Action(OBSERVE, float(vs.xs[v]), float(vs.ys[v]), float(vs.yaws[v]), value,
                      float(T), view=v, detail=detail)


class Strategy(ABC):
    name = 'base'
    uses_raw_detections = False     # misja podaje surowe detekcje w ctx.raw_candidates

    def __init__(self, params: SearchParams):
        self.params = params

    @abstractmethod
    def choose(self, ctx: PlanningContext) -> Optional[Action]:
        """Najlepsza akcja w bieżącym stanie albo None (brak sensownych akcji)."""

    def value_of(self, ctx: PlanningContext, action: Action) -> float:
        """Aktualna ocena wykonywanej akcji (domyślnie bez zmian - brak wywłaszczania)."""
        return action.value

    def on_reached(self, action: Action):
        """Powiadomienie: łazik dotarł do celu akcji."""

    def on_failed(self, action: Action):
        """Powiadomienie: nawigacja do celu akcji się nie powiodła."""


def threshold_approach(ctx: PlanningContext, threshold: float) -> Optional[Action]:
    """Reguła dojazdu strategii odniesienia: najbliższy kandydat z P_k > progu."""
    best = None
    for c in ctx.candidates:
        if c.mass < threshold:
            continue
        a = ctx.approach_action(c, value=np.inf)
        if a is not None and (best is None or a.time < best.time):
            best = a
    return best
