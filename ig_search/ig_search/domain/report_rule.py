"""
Reguła zgłoszenia próbki (DESIGN §3.5). Bez ROS.

W pozie q z estymatą ŝ zgłaszamy, gdy ŝ ∈ R(q) oraz
    P_rep = 1 − Π_{i: c_i ∈ B(ŝ, ε)} (1 − p_i)  ≥  τ_n
(prawdopodobieństwo i osiągalność liczone osobno - przy niepewnej, ale nieosiągalnej
próbce łazik podjeżdża bliżej zamiast ponownie weryfikować).
n - numer zgłoszenia. Fałszywe trzecie zgłoszenie kończy misję, więc τ_3 > τ_1.
R(q) - obszar chwytu pionowego: pierścień wokół podstawy ramienia, przed łazikiem.
"""
from dataclasses import dataclass
import math

from ig_search.domain.belief import Belief, presence
from ig_search.domain.params import ReportParams
import numpy as np


def workspace_polar(q, xs, ys, params: ReportParams):
    """Odległość i kierunek (względem kursu łazika) punktów od podstawy ramienia."""
    x, y, psi = q
    rx = x + params.workspace_root_x * math.cos(psi)
    ry = y + params.workspace_root_x * math.sin(psi)
    dx, dy = np.asarray(xs, float) - rx, np.asarray(ys, float) - ry
    bearing = (np.arctan2(dy, dx) - psi + math.pi) % (2 * math.pi) - math.pi
    return np.hypot(dx, dy), bearing


DZ_FLAT = 0.015         # [m] środek leżącej próbki nad gruntem pod łazikiem (płaski teren)


def workspace_r_max(params: ReportParams, dz=DZ_FLAT):
    """
    Zewnętrzny promień R(q) dla wysokości środka próbki względem base_link (z IK).

    Poniżej tabeli liniowo (próbka niżej - zasięg dalej maleje, ~2 cm na 1 cm wysokości, przy
    dz ≈ -7 cm R(q) jest pusty), powyżej - wartość brzegowa.
    """
    dzs = np.asarray(params.workspace_dz, float)
    rs = np.asarray(params.workspace_r_max_dz, float)
    dz = np.asarray(dz, float)
    slope = (rs[1] - rs[0]) / (dzs[1] - dzs[0])
    return np.where(dz < dzs[0], rs[0] + slope * (dz - dzs[0]), np.interp(dz, dzs, rs))


def reach_ring_width(params: ReportParams, dz=DZ_FLAT):
    """Szerokość pierścienia R(q) z zapasem (warunek końca dojazdu); ≤ 0 - nieosiągalny."""
    return (workspace_r_max(params, dz) - params.reach_margin
            - (params.workspace_r_min + params.reach_margin))


def in_workspace(q, xs, ys, params: ReportParams, margin: bool = False,
                 dz=DZ_FLAT) -> np.ndarray:
    """Maska punktów w R(q); margin - z zapasem (warunek zakończenia dojazdu)."""
    r, bearing = workspace_polar(q, xs, ys, params)
    m = params.reach_margin if margin else 0.0
    a = params.workspace_half_angle_deg - (params.reach_angle_margin_deg if margin else 0.0)
    return ((r >= params.workspace_r_min + m) & (r <= workspace_r_max(params, dz) - m)
            & (np.abs(bearing) <= math.radians(a)))


def threshold(n_reported: int, params: ReportParams) -> float:
    """τ_n dla kolejnego zgłoszenia (n_reported - dotychczasowe zgłoszenia)."""
    t = params.thresholds
    return t[min(n_reported, len(t) - 1)]


@dataclass(frozen=True)
class ReportCheck:
    x: float                # ŝ po doprecyzowaniu
    y: float
    probability: float      # P_rep
    in_reach: bool          # ŝ ∈ R(q)


def check_report(belief: Belief, q, x0: float, y0: float, params: ReportParams,
                 search_radius: float = 0.5, dz=DZ_FLAT) -> ReportCheck:
    """Doprecyzowuje ŝ wokół (x0, y0), liczy P_rep i sprawdza, czy ŝ ∈ R(q)."""
    p = belief.p
    near = belief.active & (np.hypot(belief.cx - x0, belief.cy - y0) <= search_radius)
    if not near.any() or p[near].max() <= 0:
        return ReportCheck(x0, y0, 0.0, bool(in_workspace(q, [x0], [y0], params, dz=dz)[0]))
    ids = np.flatnonzero(near)
    peak = ids[np.argmax(p[ids])]
    local = ids[np.hypot(belief.cx[ids] - belief.cx[peak], belief.cy[ids] - belief.cy[peak])
                <= params.epsilon + belief.grid.res / 2]
    w = p[local]
    x = float((w * belief.cx[local]).sum() / w.sum())
    y = float((w * belief.cy[local]).sum() / w.sum())
    # komórki, które przecinają kulę B(ŝ, ε): środek bliżej niż ε + a/2
    ball = ids[np.hypot(belief.cx[ids] - x, belief.cy[ids] - y)
               <= params.epsilon + belief.grid.res / 2]
    prob = presence(p[ball])
    return ReportCheck(x, y, prob, bool(in_workspace(q, [x], [y], params, dz=dz)[0]))
