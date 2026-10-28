"""
Widoczność 2,5D: które komórki beliefu widzi kamera w danej pozie (DESIGN §3.1). Bez ROS.

Ramka kamery w tym module: x w przód (oś optyczna), y w lewo, z w górę (jak base_link,
nie ramka optyczna). Komórka i ∈ V(q), jeśli punkt (c_i, h(c_i) + δ):
  1. rzutuje się w prostokąt kadru (kamera perspektywiczna, FOV poziomy i pionowy),
  2. jest w odległości [r_min, r_max] od kamery,
  3. odcinek kamera -> punkt nie przechodzi pod terenem (marsz po odcinku z marginesem).
"""
from dataclasses import dataclass
import math

from ig_search.domain.params import CameraParams
from ig_search.domain.terrain import Grid, HeightMap
import numpy as np


@dataclass(frozen=True)
class CameraPose:
    """Położenie kamery w ramce map i macierz obrotu R (kolumny = osie x, y, z kamery)."""

    position: np.ndarray
    R: np.ndarray

    @property
    def forward(self) -> np.ndarray:
        return self.R[:, 0]


@dataclass(frozen=True)
class VisibleSet:
    """Komórki widoczne w jednej klatce: indeksy płaskie, odległość r, kąt od osi α."""

    ids: np.ndarray
    r: np.ndarray
    alpha: np.ndarray

    @staticmethod
    def empty() -> 'VisibleSet':
        return VisibleSet(np.empty(0, np.int64), np.empty(0), np.empty(0))

    def __len__(self):
        return len(self.ids)

    def union(self, other: 'VisibleSet') -> 'VisibleSet':
        """Suma zbiorów; przy powtórzeniu komórki zostaje wpis z self."""
        extra = ~np.isin(other.ids, self.ids)
        return VisibleSet(np.concatenate([self.ids, other.ids[extra]]),
                          np.concatenate([self.r, other.r[extra]]),
                          np.concatenate([self.alpha, other.alpha[extra]]))


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def base_rotation(psi: float, grad_x: float, grad_y: float) -> np.ndarray:
    """Obrót base_link stojącego na zboczu o gradiencie (gx, gy), z kursem psi."""
    c, s = math.cos(psi), math.sin(psi)
    fwd = np.array([c, s, grad_x * c + grad_y * s])
    left = np.array([-s, c, -grad_x * s + grad_y * c])
    fwd /= np.linalg.norm(fwd)
    left -= fwd * (left @ fwd)
    left /= np.linalg.norm(left)
    return np.column_stack([fwd, left, np.cross(fwd, left)])


class Visibility:
    """Liczy V(q) dla kamery na terenie; komórki beliefu na siatce `grid`."""

    def __init__(self, heightmap: HeightMap, grid: Grid, cam: CameraParams,
                 hfov: float, vfov: float, base_radius: float = 0.5):
        self.hm = heightmap
        self.grid = grid
        self.cam = cam
        self.hfov, self.vfov = hfov, vfov
        self.cx, self.cy = grid.centers()
        h_cell = heightmap.height_at(self.cx, self.cy)
        self.cell_valid = heightmap.is_valid_at(self.cx, self.cy)
        self.cz = h_cell + cam.target_height
        # Gradient wygładzony w promieniu łazika (poza na zboczu), na siatce mapy wysokości
        self._gx, self._gy = heightmap.gradient(base_radius)

    # ------------------------------------------------------------ poza kamery
    def gradient_at(self, x: float, y: float):
        i, j = self.hm.grid.cell_of(x, y)
        i = int(np.clip(i, 0, self.hm.grid.height - 1))
        j = int(np.clip(j, 0, self.hm.grid.width - 1))
        return float(self._gx[i, j]), float(self._gy[i, j])

    def camera_pose(self, x: float, y: float, psi: float) -> CameraPose:
        """Poza kamery przedniej dla pozy łazika q = (x, y, psi) stojącego na terenie."""
        gx, gy = self.gradient_at(x, y)
        R_base = base_rotation(psi, gx, gy)
        base = np.array([x, y, float(self.hm.height_at(x, y))])
        pos = base + R_base @ np.array([self.cam.mount_x, 0.0, self.cam.mount_z])
        R_cam = R_base @ _rot_y(math.radians(self.cam.mount_pitch_deg))
        return CameraPose(pos, R_cam)

    # ------------------------------------------------------------ V(q)
    def _cells_in_radius(self, x: float, y: float, radius: float) -> np.ndarray:
        g = self.grid
        i0, j0 = g.cell_of(x - radius, y - radius)
        i1, j1 = g.cell_of(x + radius, y + radius)
        i0, j0 = max(int(i0), 0), max(int(j0), 0)
        i1, j1 = min(int(i1), g.height - 1), min(int(j1), g.width - 1)
        if i0 > i1 or j0 > j1:
            return np.empty(0, np.int64)
        ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing='ij')
        ids = (ii * g.width + jj).ravel()
        d2 = (self.cx[ids] - x) ** 2 + (self.cy[ids] - y) ** 2
        return ids[(d2 <= radius * radius) & self.cell_valid[ids]]

    def _geometry(self, pose: CameraPose, ids: np.ndarray):
        """Wektory do komórek w ramce kamery -> (r, α, vc); α = kąt od osi optycznej."""
        v = np.stack([self.cx[ids], self.cy[ids], self.cz[ids]], axis=1) - pose.position
        vc = v @ pose.R                      # współrzędne w ramce kamery
        r = np.linalg.norm(vc, axis=1)
        alpha = np.arccos(np.clip(vc[:, 0] / np.maximum(r, 1e-9), -1.0, 1.0))
        return r, alpha, vc

    def _in_frame(self, vc: np.ndarray) -> np.ndarray:
        """Rzut perspektywiczny: punkt przed kamerą i w kadrze pomniejszonym o margines."""
        x = np.maximum(vc[:, 0], 1e-9)
        m = math.radians(self.cam.frame_margin_deg)
        th = math.tan(max(self.hfov / 2 - m, 0.0))
        tv = math.tan(max(self.vfov / 2 - m, 0.0))
        return (vc[:, 0] > 0) & (np.abs(vc[:, 1]) <= x * th) & (np.abs(vc[:, 2]) <= x * tv)

    def _unoccluded(self, pose: CameraPose, ids: np.ndarray, r: np.ndarray) -> np.ndarray:
        cam = self.cam
        if len(ids) == 0:
            return np.zeros(0, bool)
        n_s = max(1, int(math.ceil(float(r.max()) / cam.occlusion_step)))
        d = np.arange(1, n_s + 1) * cam.occlusion_step          # odległość od kamery
        frac = d[None, :] / r[:, None]                          # (n, S)
        use = d[None, :] < (r[:, None] - cam.occlusion_skip_end)
        p0 = pose.position
        tx = p0[0] + frac * (self.cx[ids] - p0[0])[:, None]
        ty = p0[1] + frac * (self.cy[ids] - p0[1])[:, None]
        tz = p0[2] + frac * (self.cz[ids] - p0[2])[:, None]
        under = tz < self.hm.height_at(tx, ty) + cam.occlusion_margin
        return ~np.any(under & use, axis=1)

    def visible(self, pose: CameraPose) -> VisibleSet:
        """Zbiór V(q) dla pozy kamery (z przesłanianiem przez teren)."""
        cam = self.cam
        ids = self._cells_in_radius(pose.position[0], pose.position[1], cam.r_max)
        if len(ids) == 0:
            return VisibleSet.empty()
        r, alpha, vc = self._geometry(pose, ids)
        ok = self._in_frame(vc) & (r >= cam.r_min) & (r <= cam.r_max)
        ids, r, alpha = ids[ok], r[ok], alpha[ok]
        ok = self._unoccluded(pose, ids, r)
        return VisibleSet(ids[ok], r[ok], alpha[ok])

    def visible_from(self, x: float, y: float, psi: float) -> VisibleSet:
        return self.visible(self.camera_pose(x, y, psi))

    def cells_near(self, pose: CameraPose, points_xy: np.ndarray, radius: float) -> VisibleSet:
        """
        Komórki w promieniu od punktów (detekcji), bez testu przesłonięcia i pola widzenia.

        Detekcja w komórce, którą model widoczności uznał za zasłoniętą, to błąd modelu
        (np. mapy h), nie dowód braku próbki - te komórki też dostają aktualizację.
        """
        if len(points_xy) == 0:
            return VisibleSet.empty()
        ids = np.unique(np.concatenate(
            [self._cells_in_radius(px, py, radius) for px, py in points_xy]))
        if len(ids) == 0:
            return VisibleSet.empty()
        r, alpha, _ = self._geometry(pose, ids)
        return VisibleSet(ids, r, alpha)
