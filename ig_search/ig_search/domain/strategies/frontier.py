"""
Strategia odniesienia: najbliższy nieobejrzany obszar (czyste pokrycie), DESIGN §4.2.

Wybiera najszybciej osiągalną pozę, z której widać co najmniej `frontier_min_gain`
(Σ P_d) komórek jeszcze nieobejrzanych w bieżącym przejściu; nie korzysta z wartości p.
Przejście k obejmuje komórki obejrzane z mniej niż k różnych punktów widzenia (kubełków
bramki różnorodności), liczone tylko z nowych dla nich punktów widzenia. Po pokryciu terenu
zaczyna się kolejne przejście (do `coverage_passes`, 0 - bez limitu). Dojazd do kandydata -
gdy P_k przekroczy próg.

`chase` - „jedź do wszystkiego, co wykryto”: dojazd do najbliższej (czasem dojazdu) surowej
detekcji kamery przedniej, niezależnie od beliefu (detekcje bliżej niż 0,5 m łączone, miejsca
zgłoszone i porzucone pomijane - Mission._note_raw); na miejscu ta sama weryfikacja i reguła
zgłoszenia co w innych strategiach. Bez detekcji do odwiedzenia - eksploracja jak frontier.
Kandydaci z beliefu nie są używani.
"""
from typing import Optional

from ig_search.domain.strategies.base import (
    Action, PlanningContext, Strategy, threshold_approach)
import numpy as np
from scipy.sparse import csr_matrix


class Frontier(Strategy):
    name = 'frontier'

    def __init__(self, params):
        super().__init__(params)
        self.coverage_pass = 1

    def choose(self, ctx: PlanningContext) -> Optional[Action]:
        a = threshold_approach(ctx, self.params.planner.approach_threshold)
        return a if a is not None else self.explore(ctx)

    def explore(self, ctx: PlanningContext) -> Optional[Action]:
        pl = self.params.planner
        vs = ctx.views
        novel = ctx.belief.view_novelty(vs)
        V = csr_matrix((vs.V.data * novel, vs.indices, vs.indptr), shape=vs.V.shape)
        counts = ctx.belief.view_counts()
        T = ctx.view_times()
        reachable = np.isfinite(T)
        # bez limitu przejść: koniec, gdy żadna poza nie da nowego punktu widzenia
        if not ((V @ ctx.belief.active.astype(np.float32) >= pl.frontier_min_gain)
                & reachable).any():
            return None
        while pl.coverage_passes <= 0 or self.coverage_pass <= pl.coverage_passes:
            unseen = (ctx.belief.active & (counts < self.coverage_pass)).astype(np.float32)
            gain = V @ unseen
            ok = (gain >= pl.frontier_min_gain) & reachable
            if ok.any():
                v = int(np.argmin(np.where(ok, T, np.inf)))
                return ctx.observe_action(
                    v, 1.0, T[v],
                    f'frontier przejście {self.coverage_pass} nowe={gain[v]:.1f} T={T[v]:.1f}s')
            self.coverage_pass += 1
        return None


class Chase(Frontier):
    name = 'chase'
    uses_raw_detections = True

    def choose(self, ctx: PlanningContext) -> Optional[Action]:
        best = None
        for c in ctx.raw_candidates:
            a = ctx.approach_action(c, value=np.inf)
            if a is not None and (best is None or a.time < best.time):
                best = a
        if best is not None:
            best.detail = best.detail.replace('dojazd do', 'dojazd do surowej detekcji')
            return best
        return self.explore(ctx)
