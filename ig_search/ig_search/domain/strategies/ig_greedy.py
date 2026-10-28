"""
Strategia proponowana, zachłanna (D = 1): oczekiwana liczba zgłoszeń na sekundę (DESIGN §3.4).

  S_obs(v) = Σ_i V[v,i]·p̃_i / (T(v) + T̂_app(v))
  S_app(k) = P_k / (T_app(k) + t_verify)

p̃ - belief poza skupiskami kandydatów: obserwacja ma przynieść NOWE detekcje, a znani
kandydaci są obsługiwani akcją dojazdu (inaczej widok na kandydata liczyłby go dwa razy).
V liczone tylko dla par (v, i), w których punkt widzenia v jest dla komórki i nowy -
spójnie z bramką różnorodności (obserwacja z „zużytego” punktu widzenia nic nie daje).
T̂_app(v) - oczekiwany czas od obserwacji do zgłoszenia: dojazd z v do wykrytej komórki
(średnia odległość ważona V·p̃) i weryfikacja. Dzięki temu obie akcje mierzą to samo:
oczekiwaną liczbę poprawnych zgłoszeń na sekundę.
"""
from typing import Optional

from ig_search.domain.strategies.base import Action, APPROACH, PlanningContext, Strategy
import numpy as np
from scipy.sparse import csr_matrix


class IGGreedy(Strategy):
    name = 'ig_greedy'

    def _observe_scores(self, ctx: PlanningContext):
        vs = ctx.views
        p = ctx.belief.p.copy()
        for c in ctx.candidates:
            p[c.cells] = 0.0
        novel = ctx.belief.view_novelty(vs)
        V = csr_matrix((vs.V.data * novel, vs.indices, vs.indptr), shape=vs.V.shape)
        RV = csr_matrix((vs.RV.data * novel, vs.indices, vs.indptr), shape=vs.V.shape)
        exp_det = V @ p
        rng = RV @ p
        r_hat = rng / np.maximum(exp_det, 1e-12)
        cp, tp = self.params.candidates, self.params.travel
        t_follow = (np.maximum(r_hat - cp.approach_distance, 0.0) / tp.v_max
                    + tp.t_overhead + cp.t_verify)
        T = ctx.view_times()
        return exp_det / (T + t_follow), exp_det, T

    def _approach_score(self, ctx: PlanningContext, cand) -> tuple:
        k, T = ctx.approach_time(cand)
        return cand.mass / (T + self.params.candidates.t_verify), T

    def choose(self, ctx: PlanningContext) -> Optional[Action]:
        S, gain, T = self._observe_scores(ctx)
        best = None
        if np.isfinite(S).any():
            v = int(np.argmax(np.where(np.isfinite(S), S, -np.inf)))
            if gain[v] >= self.params.planner.min_gain and S[v] > 0 and np.isfinite(T[v]):
                best = ctx.observe_action(
                    v, float(S[v]), T[v],
                    f'obserwacja zysk={gain[v]:.3f} T={T[v]:.1f}s S={S[v]:.2e}/s')
        for c in ctx.candidates:
            s, _ = self._approach_score(ctx, c)
            if np.isfinite(s) and (best is None or s > best.value):
                a = ctx.approach_action(c, float(s))
                if a is not None:
                    a.detail += f' S={s:.2e}/s'
                    best = a
        return best

    def value_of(self, ctx: PlanningContext, action: Action) -> float:
        if action.kind == APPROACH:
            if action.candidate is None:
                return 0.0
            s, _ = self._approach_score(ctx, action.candidate)
            return float(s) if np.isfinite(s) else 0.0
        S, _, _ = self._observe_scores(ctx)
        s = S[action.view]
        return float(s) if np.isfinite(s) else 0.0
