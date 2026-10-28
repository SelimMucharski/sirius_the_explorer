"""
Strategia odniesienia: ścieżka meandrowa (boustrophedon), DESIGN §4.2.

Pasy wzdłuż dłuższej osi obszaru przejezdnego, rozstaw z szerokości pola widzenia
na odległości pełnej skuteczności detektora: 2·r_full·sin(FOV_h/2)·(1 − zakładka).
Punkty pasa w kolejności „koszenia trawnika”; nieprzejezdne pomijane. Po ostatnim punkcie
kolejne przejście (do `coverage_passes`, 0 - bez limitu): pasy przesunięte o pół rozstawu
w przejściach parzystych, przejazd od końca (zaczyna przy miejscu, w którym skończyło się
poprzednie). Dojazd do kandydata - gdy P_k przekroczy próg (ta sama reguła zgłoszenia co
w innych strategiach).
"""
import math
from typing import Optional

from ig_search.domain.params import SearchParams
from ig_search.domain.strategies.base import (
    Action, OBSERVE, PlanningContext, Strategy, threshold_approach)
from ig_search.domain.travel import TravelModel
import numpy as np


def lane_spacing(params: SearchParams) -> float:
    pl = params.planner
    return (2 * params.detector.r_full * math.sin(params.hfov / 2)
            * (1.0 - pl.meander_overlap))


def build_waypoints(travel: TravelModel, params: SearchParams, shift: float = 0.0):
    """(xs, ys, yaws) punktów meandra w kolejności przejazdu; pasy przesunięte o `shift` [m]."""
    trav = travel.traversable
    xs, ys = travel.cx[trav], travel.cy[trav]
    along_x = (xs.max() - xs.min()) >= (ys.max() - ys.min())
    a, b = (xs, ys) if along_x else (ys, xs)        # a - wzdłuż pasa, b - w poprzek
    spacing = lane_spacing(params)
    step = params.planner.meander_waypoint_step
    out = []
    lanes = np.arange(b.min() + spacing / 2 + shift, b.max(), spacing)
    for n, lb in enumerate(lanes):
        pts = np.arange(a.min(), a.max() + step, step)
        if n % 2:
            pts = pts[::-1]
        heading = 0.0 if n % 2 == 0 else math.pi
        for pa in pts:
            px, py = (pa, lb) if along_x else (lb, pa)
            c = int(travel.grid.flat_index(px, py))
            if (c >= 0 and travel.traversable[c]
                    and travel.clearance[c] >= params.views.edge_clearance):
                yaw = heading if along_x else heading + math.pi / 2
                out.append((px, py, yaw))
    arr = np.array(out) if out else np.empty((0, 3))
    return arr[:, 0], arr[:, 1], arr[:, 2]


class Meander(Strategy):
    name = 'meander'

    def __init__(self, params: SearchParams):
        super().__init__(params)
        self.wp = None
        self.next = 0
        self.coverage_pass = 1

    def _skip_reached(self, ctx: PlanningContext):
        xs, ys, _ = self.wp
        r = self.params.planner.waypoint_reached
        while self.next < len(xs):
            c = int(ctx.travel.grid.flat_index(xs[self.next], ys[self.next]))
            near = math.hypot(xs[self.next] - ctx.q[0], ys[self.next] - ctx.q[1]) <= r
            if near or c < 0 or not np.isfinite(ctx.drive[c]):
                self.next += 1
            else:
                break

    def choose(self, ctx: PlanningContext) -> Optional[Action]:
        a = threshold_approach(ctx, self.params.planner.approach_threshold)
        if a is not None:
            return a
        if self.wp is None:
            self.wp = build_waypoints(ctx.travel, self.params)
        self._skip_reached(ctx)
        passes = self.params.planner.coverage_passes
        while self.next >= len(self.wp[0]):
            if passes > 0 and self.coverage_pass >= passes:
                return None
            self.coverage_pass += 1
            even = self.coverage_pass % 2 == 0
            xs, ys, yaws = build_waypoints(
                ctx.travel, self.params, lane_spacing(self.params) / 2 if even else 0.0)
            if len(xs) == 0:
                return None
            if even:                                # od końca, przodem w drugą stronę
                xs, ys, yaws = xs[::-1], ys[::-1], np.mod(yaws[::-1] + math.pi, 2 * math.pi)
            self.wp, self.next = (xs, ys, yaws), 0
            self._skip_reached(ctx)
        xs, ys, yaws = self.wp
        i = self.next
        c = int(ctx.travel.grid.flat_index(xs[i], ys[i]))
        T = ctx.travel.pose_times(ctx.q, [xs[i]], [ys[i]], [yaws[i]], ctx.drive, [c])[0]
        return Action(OBSERVE, float(xs[i]), float(ys[i]), float(yaws[i]), 1.0, float(T),
                      detail=f'meander przejście {self.coverage_pass} punkt {i + 1}/{len(xs)}')

    def on_reached(self, action: Action):
        if action.kind == OBSERVE and self.wp is not None:
            self.next += 1

    def on_failed(self, action: Action):
        self.on_reached(action)
