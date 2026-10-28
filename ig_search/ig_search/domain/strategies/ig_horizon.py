"""
Strategia z horyzontem (D > 1): sekwencja obserwacji, zgłoszenia na sekundę (DESIGN §3.6).

Przeszukiwanie wiązkowe po sekwencjach póz σ = (v_1, …, v_D):
  - poziom 1: pozy oceniane z bieżącego położenia jak w ig_greedy; z każdego węzła wiązki
    rozszerzenie o `beam_candidates` najlepszych póz według oceny zachłannej z jego miejsca,
    zostaje `beam_width` najlepszych sekwencji;
  - między obserwacjami belief jest aktualizowany jak przy braku detekcji (najbardziej
    prawdopodobny wynik przy małych p): p_i ← p_i(1 − P_d)/(1 − p_i·P_d), więc kolejne pozy
    nie liczą drugi raz obszaru obejrzanego przez wcześniejsze;
  - wartość sekwencji - oczekiwana liczba zgłoszeń na sekundę (decyzja autora 2026-10-26,
    bez kary za czas f(t)):
        J(σ) = Σ_k g_k / Σ_k (T_k + t_follow,k),
    g_k = Σ_i V[v_k,i]·p_i⁽ᵏ⁾, T_k - czas dojazdu do v_k od poprzedniej pozy, t_follow,k -
    czas od obserwacji do zgłoszenia (jak w ig_greedy), liczony przy każdej obserwacji jak
    w ig_greedy - inaczej sekwencja rozkładałaby go na kilka obserwacji i jej ocena byłaby
    zawyżona względem oceny dojazdu do kandydata.
Dla D = 1 J(σ) jest oceną obserwacji z ig_greedy. Wykonywana jest tylko pierwsza akcja
najlepszej sekwencji (horyzont przesuwny). Dojazd do kandydata - tylko jako pierwsza akcja,
porównywany z J w tej samej skali; jego wyniku drzewo nie symuluje.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

from ig_search.domain.strategies.base import Action, APPROACH, PlanningContext
from ig_search.domain.strategies.ig_greedy import IGGreedy
import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class _Node:
    views: Tuple[int, ...]
    p: np.ndarray                   # belief po symulowanych obserwacjach
    pose: tuple                     # (x, y, psi) po ostatniej obserwacji
    gain: float                     # Σ g_k
    time: float                     # Σ T_k
    follow: float                   # Σ t_follow,k

    @property
    def value(self) -> float:
        if self.gain <= 0:
            return 0.0
        return self.gain / (self.time + self.follow)


class IGHorizon(IGGreedy):
    name = 'ig_horizon'

    def __init__(self, params):
        super().__init__(params)
        self._drive_cache = {}          # (graf, komórka) -> czasy jazdy z tej komórki

    # ------------------------------------------------------------ elementy
    def _matrices(self, ctx: PlanningContext):
        vs = ctx.views
        novel = ctx.belief.view_novelty(vs)
        V = csr_matrix((vs.V.data * novel, vs.indices, vs.indptr), shape=vs.V.shape)
        RV = csr_matrix((vs.RV.data * novel, vs.indices, vs.indptr), shape=vs.V.shape)
        return V, RV

    def _drive_from(self, ctx: PlanningContext, x: float, y: float) -> np.ndarray:
        tr = ctx.travel
        key = (id(tr.G), tr.nearest_traversable(x, y))
        d = self._drive_cache.get(key)
        if d is None:
            if len(self._drive_cache) > 512:
                self._drive_cache.clear()
            d = self._drive_cache[key] = tr.drive_times(x, y)
        return d

    def _times(self, ctx: PlanningContext, pose, drive) -> np.ndarray:
        vs = ctx.views
        T = ctx.travel.pose_times(pose, vs.xs, vs.ys, vs.yaws, drive, vs.cells)
        T[~vs.ok | ~np.isfinite(T)] = np.inf
        if ctx.view_blocked is not None:
            T[ctx.view_blocked] = np.inf
        return T

    def _expand(self, ctx, V, RV, node: _Node, T: np.ndarray, n: int, only=None):
        """Najlepsze rozszerzenia węzła (według oceny zachłannej z jego miejsca)."""
        exp_det = V @ node.p
        r_hat = (RV @ node.p) / np.maximum(exp_det, 1e-12)
        cp, tp = self.params.candidates, self.params.travel
        t_follow = (np.maximum(r_hat - cp.approach_distance, 0.0) / tp.v_max
                    + tp.t_overhead + cp.t_verify)
        S = exp_det / (T + t_follow)
        S[list(node.views)] = -np.inf
        S[~np.isfinite(S)] = -np.inf
        if only is not None:
            picks = [only] if np.isfinite(S[only]) else []
        else:
            k = min(n, len(S))
            picks = np.argpartition(-S, k - 1)[:k] if k > 0 else []
        vs = ctx.views
        out = []
        for v in picks:
            v = int(v)
            if not np.isfinite(S[v]) or exp_det[v] <= 0:
                continue
            p = node.p.copy()
            a, b = vs.indptr[v], vs.indptr[v + 1]
            idx, d = vs.indices[a:b], V.data[a:b]
            p[idx] = p[idx] * (1.0 - d) / np.maximum(1.0 - p[idx] * d, 1e-12)
            g = float(exp_det[v])
            out.append(_Node(node.views + (v,), p,
                             (float(vs.xs[v]), float(vs.ys[v]), float(vs.yaws[v])),
                             node.gain + g, node.time + float(T[v]),
                             node.follow + float(t_follow[v])))
        return out

    def _plan(self, ctx: PlanningContext, first: Optional[int] = None) -> Optional[_Node]:
        """Najlepsza sekwencja (z pierwszą pozą `first`, gdy podana) albo None."""
        pl = self.params.planner
        V, RV = self._matrices(ctx)
        p = ctx.belief.p.copy()
        for c in ctx.candidates:                    # p̃ - jak w ig_greedy
            p[c.cells] = 0.0
        root = _Node((), p, ctx.q, 0.0, 0.0, 0.0)
        beam = self._expand(ctx, V, RV, root, ctx.view_times(), pl.beam_candidates, first)
        beam = sorted(beam, key=lambda n: -n.value)[:max(pl.beam_width, 1)]
        if not beam:
            return None
        if beam[0].gain < pl.min_gain and first is None:
            return None
        for _ in range(1, max(pl.horizon_depth, 1)):
            nxt = []
            for node in beam:
                drive = self._drive_from(ctx, node.pose[0], node.pose[1])
                nxt += self._expand(ctx, V, RV, node, self._times(ctx, node.pose, drive),
                                    pl.beam_candidates)
            if not nxt:
                break
            beam = sorted(nxt, key=lambda n: -n.value)[:pl.beam_width]
        return beam[0]

    # ------------------------------------------------------------ port
    def choose(self, ctx: PlanningContext) -> Optional[Action]:
        best = None
        node = self._plan(ctx)
        if node is not None and node.value > 0:
            v = node.views[0]
            T = ctx.view_times()[v]
            if np.isfinite(T):
                seq = ' → '.join(str(i) for i in node.views)
                best = ctx.observe_action(
                    v, float(node.value), T,
                    f'obserwacja (sekwencja {len(node.views)}: {seq}) zysk={node.gain:.3f} '
                    f'T={node.time:.1f}s J={node.value:.2e}/s')
                vs = ctx.views
                best.plan = tuple((float(vs.xs[i]), float(vs.ys[i]), float(vs.yaws[i]))
                                  for i in node.views)
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
            return super().value_of(ctx, action)
        node = self._plan(ctx, first=action.view)
        return float(node.value) if node is not None else 0.0
