"""
Czas ruchu po terenie (DESIGN §3.7). Bez ROS.

Graf 8-sąsiedztwa po komórkach przejezdnych, waga krawędzi = czas:
    L / v_max · (1 + c_up·s⁺ + c_down·s⁻),   s = Δh / L,
krawędzie o |s| > tg(slope_max) usunięte. Do czasu jazdy dochodzą obroty (napęd
różnicowy, obrót w miejscu) i stały narzut na cel.
"""
import math

from ig_search.domain.params import TravelParams
from ig_search.domain.terrain import Grid, HeightMap
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra


def wrap(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


MAIN_COMPONENT_FRACTION = 0.05     # składowa grafu mniejsza niż tyle największej - wysepka


class TravelModel:

    def __init__(self, heightmap: HeightMap, grid: Grid, params: TravelParams, allowed=None):
        """Parametr `allowed`: opcjonalna maska komórek siatki (mapa przejezdności, DESIGN §6)."""
        self.grid = grid
        self.p = params
        self.cx, self.cy = grid.centers()

        def coarse(fine_values, how):
            masked = np.where(heightmap.valid, fine_values, np.nan)
            return HeightMap(masked, heightmap.grid).resample(grid, how).ravel()

        self.h = coarse(heightmap.smoothed(params.robot_radius), 'mean')
        rough = coarse(heightmap.roughness(params.robot_radius), 'max')
        gx, gy = heightmap.gradient(params.robot_radius)
        slope = coarse(np.hypot(gx, gy), 'max')
        s_max = math.tan(math.radians(params.slope_max_deg))
        bump = coarse(heightmap.bump_height(params.robot_radius), 'max')
        ok = (np.isfinite(self.h) & (np.nan_to_num(slope, nan=np.inf) <= s_max)
              & (np.nan_to_num(rough, nan=np.inf) <= params.roughness_max)
              & (np.nan_to_num(bump, nan=np.inf) <= params.bump_max))
        if allowed is not None:
            ok &= np.asarray(allowed, bool).ravel()
        # Odstęp od komórek nieprzejezdnych (promień łazika) i od krawędzi mapy wysokości
        clear = distance_transform_edt(ok.reshape(grid.shape)) * grid.res
        # brzeg siatki też jest krawędzią mapy (ramka nieważnych komórek)
        valid = np.pad(np.isfinite(self.h).reshape(grid.shape), 1, constant_values=False)
        edge = distance_transform_edt(valid)[1:-1, 1:-1] * grid.res
        self.traversable = ((clear.ravel() >= params.robot_radius)
                            & (edge.ravel() >= params.edge_margin) & ok)
        # odległość od komórek nieprzejezdnych i brzegu mapy [m] (pozy obserwacji z odstępem)
        self.clearance = np.minimum(clear.ravel(), edge.ravel())
        self.slope = slope
        self._build_graph()

    def _build_graph(self):
        g = self.grid
        H, W = g.shape
        trav = self.traversable.reshape(H, W)
        ys, xs = np.nonzero(trav)
        rows, cols, w = [], [], []
        s_max = math.tan(math.radians(self.p.slope_max_deg))
        h = self.h.reshape(H, W)
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)):
            ny, nx = ys + dy, xs + dx
            ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
            ay, ax, ny, nx = ys[ok], xs[ok], ny[ok], nx[ok]
            ok = trav[ny, nx]
            if dy and dx:                       # bez ścinania narożników
                ok &= trav[ay, nx] & trav[ny, ax]
            ay, ax, ny, nx = ay[ok], ax[ok], ny[ok], nx[ok]
            length = math.hypot(dx, dy) * g.res
            s = (h[ny, nx] - h[ay, ax]) / length
            keep = np.abs(s) <= s_max
            ay, ax, ny, nx, s = ay[keep], ax[keep], ny[keep], nx[keep], s[keep]
            t = length / self.p.v_max * (1.0 + self.p.uphill_cost * np.maximum(s, 0)
                                         + self.p.downhill_cost * np.maximum(-s, 0))
            rows.append(ay * W + ax)
            cols.append(ny * W + nx)
            w.append(t)
        n = g.size
        self.G = csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                            shape=(n, n))
        # spójne składowe: start trasy z wysepki (np. po wyłączeniu miejsca utknięcia) -
        # od najbliższej komórki dużej składowej, inaczej wszystkie czasy byłyby nieskończone
        _, lab = connected_components(self.G, directed=False)
        sizes = np.bincount(lab[self.traversable], minlength=lab.max() + 1)
        big = sizes >= MAIN_COMPONENT_FRACTION * max(int(sizes.max()), 1)
        self._main = self.traversable & big[lab]

    # ------------------------------------------------------------ zapytania
    def block(self, x: float, y: float, radius: float) -> int:
        """Wyłącza komórki w promieniu (miejsce, gdzie łazik utknął); liczba wyłączonych."""
        m = ((self.cx - x) ** 2 + (self.cy - y) ** 2 <= radius ** 2) & self.traversable
        if m.any():
            self.traversable[m] = False
            self._build_graph()
        return int(m.sum())

    def nearest_traversable(self, x: float, y: float) -> int:
        """Najbliższa komórka przejezdna w dużej składowej grafu (nie na odciętej wysepce)."""
        idx = np.flatnonzero(self._main if self._main.any() else self.traversable)
        if len(idx) == 0:
            raise RuntimeError('brak komórek przejezdnych')
        d2 = (self.cx[idx] - x) ** 2 + (self.cy[idx] - y) ** 2
        return int(idx[np.argmin(d2)])

    def drive_times(self, x: float, y: float) -> np.ndarray:
        """Czas jazdy [s] z (x, y) do każdej komórki (inf = nieosiągalna)."""
        src = self.nearest_traversable(x, y)
        return dijkstra(self.G, directed=True, indices=src)

    def waypoints(self, x0: float, y0: float, x1: float, y1: float,
                  spacing: float = 1.5) -> list:
        """
        Punkty pośrednie najszybszej trasy po grafie przejezdności (bez punktu końcowego).

        Nav2 nie zna nachylenia (costmapa tylko z przeszkód kamery głębi) i potrafi
        poprowadzić łazik przez zbocze; trasa przez gęste punkty pośrednie trzyma go na
        terenie przejezdnym według tego modelu. Pusta lista: cel blisko albo nieosiągalny.
        """
        src = self.nearest_traversable(x0, y0)
        dst = self.nearest_traversable(x1, y1)
        _, pred = dijkstra(self.G, directed=True, indices=src, return_predecessors=True)
        if dst != src and pred[dst] < 0:
            return []
        cells = [dst]
        while cells[-1] != src:
            cells.append(int(pred[cells[-1]]))
        cells.reverse()
        out, acc = [], 0.0
        for a, b in zip(cells[:-1], cells[1:]):
            acc += float(np.hypot(self.cx[b] - self.cx[a], self.cy[b] - self.cy[a]))
            if acc >= spacing:
                out.append((float(self.cx[b]), float(self.cy[b])))
                acc = 0.0
        # ostatni punkt pośredni nie bliżej niż spacing/2 od celu
        while out and np.hypot(out[-1][0] - x1, out[-1][1] - y1) < spacing / 2:
            out.pop()
        return out

    def pose_times(self, q, xs, ys, yaws, drive: np.ndarray, cells) -> np.ndarray:
        """Czas dotarcia z q = (x, y, psi) do póz (xs, ys, yaws) leżących w komórkach `cells`."""
        x0, y0, psi0 = q
        d = drive[cells]
        yaws = np.asarray(yaws)
        bearing = np.arctan2(np.asarray(ys) - y0, np.asarray(xs) - x0)
        far = d > 2.0 * self.grid.res / self.p.v_max
        rot = np.where(far, np.abs(wrap(bearing - psi0)) + np.abs(wrap(yaws - bearing)),
                       np.abs(wrap(yaws - psi0)))
        return d + rot / self.p.w_max + self.p.t_overhead
