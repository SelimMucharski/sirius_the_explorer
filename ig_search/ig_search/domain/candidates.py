"""
Kandydaci na próbki: skupiska komórek o podwyższonym p (DESIGN §3.3). Bez ROS.

  kandydat k = spójna składowa komórek z p_i > p_seed,
  P_k = 1 − Π (1 − p_i)        (≥ 1 próbka w skupisku),
  ŝ_k = Σ p_i c_i / Σ p_i,
  pozy dojazdu: łazik w odległości d (domyślnie `approach_distance`) od ŝ_k, przodem do niej,
  z `n_approach_dirs` kierunków; dopuszczalne tylko w komórkach przejezdnych.
"""
from dataclasses import dataclass, field
import math
from typing import List

from ig_search.domain.belief import Belief, presence
from ig_search.domain.params import CandidateParams
from ig_search.domain.travel import TravelModel
import numpy as np
from scipy.ndimage import label


@dataclass
class Candidate:
    cells: np.ndarray
    mass: float                     # P_k
    x: float                        # ŝ_k
    y: float
    p_max: float
    cell_p: np.ndarray = field(default_factory=lambda: np.empty(0))   # p komórek skupiska
    app_x: np.ndarray = field(default_factory=lambda: np.empty(0))
    app_y: np.ndarray = field(default_factory=lambda: np.empty(0))
    app_yaw: np.ndarray = field(default_factory=lambda: np.empty(0))
    app_cell: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    raw: bool = False               # surowa detekcja (strategia chase), nie skupisko beliefu

    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)


def find_candidates(belief: Belief, params: CandidateParams) -> List[Candidate]:
    p = belief.p
    seed = (p > params.p_seed) & belief.active
    if not seed.any():
        return []
    lab, n = label(seed.reshape(belief.grid.shape), structure=np.ones((3, 3), int))
    lab = lab.ravel()
    out = []
    idx = np.flatnonzero(lab)
    order = np.argsort(lab[idx], kind='stable')
    idx = idx[order]
    splits = np.flatnonzero(np.diff(lab[idx])) + 1
    for cells in np.split(idx, splits):
        pc = p[cells]
        w = pc.sum()
        out.append(Candidate(cells=cells, mass=presence(pc),
                             x=float((pc * belief.cx[cells]).sum() / w),
                             y=float((pc * belief.cy[cells]).sum() / w),
                             p_max=float(pc.max()), cell_p=pc))
    out.sort(key=lambda c: -c.mass)
    return out


def attach_approach_poses(cand: Candidate, travel: TravelModel, params: CandidateParams,
                          exclude_dirs=(), distance: float = None, ring=None) -> Candidate:
    """
    Wylicza dopuszczalne pozy dojazdu (komórki przejezdne), z pominięciem kierunków.

    Poza nie stoi na skupisku kandydata (max_mass_under_rover); ring(xs, ys) - szerokość
    pierścienia R(q) z zapasem dla ŝ z pozy (dojazd w zasięg ramienia): pozy z pustym
    pierścieniem (próbka za nisko względem łazika) odpadają. Gdy filtr odrzuci wszystkie
    pozy przejezdne, zostaje najlepsza z nich.
    """
    th = np.arange(params.n_approach_dirs) * 2 * math.pi / params.n_approach_dirs
    keep = np.ones(len(th), bool)
    half_sector = math.pi / params.n_approach_dirs
    for d in exclude_dirs:
        keep &= np.abs((th - d + math.pi) % (2 * math.pi) - math.pi) > half_sector
    th = th[keep]
    d = params.approach_distance if distance is None else distance
    xs = cand.x - d * np.cos(th)
    ys = cand.y - d * np.sin(th)
    cells = travel.grid.flat_index(xs, ys)
    ok = cells >= 0
    ok[ok] &= travel.traversable[cells[ok]]
    under = mass_under_rover(cand, travel, xs, ys)
    ok = _filter(ok, under <= params.max_mass_under_rover * max(cand.mass, 1e-9), -under)
    if ring is not None:
        width = np.asarray(ring(xs, ys), float)
        ok = _filter(ok, width > 0.0, width)
    cand.app_x, cand.app_y, cand.app_yaw, cand.app_cell = xs[ok], ys[ok], th[ok], cells[ok]
    return cand


def _filter(ok, good, score):
    """Maska ok & good, a gdy nic nie zostaje - tylko najlepsze (max score) z ok."""
    if (ok & good).any():
        return ok & good
    if ok.any():
        return ok & (score == score[ok].max())
    return ok


def mass_under_rover(cand: Candidate, travel: TravelModel, xs, ys) -> np.ndarray:
    """P(co najmniej jedna próbka skupiska pod łazikiem) dla póz (xs, ys), promień łazika."""
    xs, ys = np.atleast_1d(xs), np.atleast_1d(ys)
    if len(cand.cells) == 0 or len(cand.cell_p) != len(cand.cells):
        return np.zeros(len(xs))
    g = travel.grid
    cx = g.origin_x + (cand.cells % g.width + 0.5) * g.res
    cy = g.origin_y + (cand.cells // g.width + 0.5) * g.res
    d2 = (cx[None, :] - xs[:, None]) ** 2 + (cy[None, :] - ys[:, None]) ** 2
    inside = d2 <= travel.p.robot_radius ** 2
    return 1.0 - np.prod(np.where(inside, 1.0 - cand.cell_p[None, :], 1.0), axis=1)


def match_candidate(cands: List[Candidate], x: float, y: float, radius: float):
    """Kandydat najbliższy punktowi (x, y) w promieniu - śledzenie celu między replanami."""
    best = min(cands, key=lambda c: c.distance_to(x, y), default=None)
    if best is None or best.distance_to(x, y) > radius:
        return None
    return best
