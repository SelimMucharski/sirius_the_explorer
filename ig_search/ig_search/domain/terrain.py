"""
Siatka i mapa wysokości terenu h: W -> R (DESIGN §3.1). Bez ROS.

Konwencja: komórka (i, j) = (wiersz, kolumna); wiersz rośnie z y, kolumna z x.
`origin` to narożnik (x, y) komórki (0, 0). Komórki płaskie (indeks i·W + j) służą
jako indeksy beliefu.
"""
from dataclasses import dataclass
import os
import warnings

import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter, uniform_filter
import yaml


@dataclass(frozen=True)
class Grid:
    """Geometria regularnej siatki: rozdzielczość, narożnik, rozmiar (H, W)."""

    res: float
    origin_x: float
    origin_y: float
    height: int
    width: int

    @property
    def shape(self):
        return (self.height, self.width)

    @property
    def size(self) -> int:
        return self.height * self.width

    def centers(self):
        """Środki wszystkich komórek jako płaskie tablice (cx, cy)."""
        jj, ii = np.meshgrid(np.arange(self.width), np.arange(self.height))
        cx = self.origin_x + (jj.ravel() + 0.5) * self.res
        cy = self.origin_y + (ii.ravel() + 0.5) * self.res
        return cx, cy

    def cell_of(self, x, y):
        """(i, j) komórki zawierającej punkt; bez obcinania do zakresu."""
        j = np.floor((np.asarray(x) - self.origin_x) / self.res).astype(int)
        i = np.floor((np.asarray(y) - self.origin_y) / self.res).astype(int)
        return i, j

    def inside(self, i, j):
        return (i >= 0) & (i < self.height) & (j >= 0) & (j < self.width)

    def flat_index(self, x, y):
        """Płaski indeks komórki albo -1 poza siatką."""
        i, j = self.cell_of(x, y)
        return np.where(self.inside(i, j), i * self.width + j, -1)

    def coarsen(self, k: int) -> 'Grid':
        """Siatka k razy rzadsza o tym samym narożniku."""
        return Grid(self.res * k, self.origin_x, self.origin_y,
                    -(-self.height // k), -(-self.width // k))


class HeightMap:
    """Mapa wysokości na siatce; NaN = brak danych (poza terenem)."""

    def __init__(self, h: np.ndarray, grid: Grid):
        if h.shape != grid.shape:
            raise ValueError(f'kształt h {h.shape} != siatka {grid.shape}')
        self.grid = grid
        self.h = h.astype(np.float64)
        self.valid = np.isfinite(self.h)
        # Braki wypełnione najbliższą znaną wysokością: brzeg mapy nie tworzy sztucznego zbocza
        if self.valid.all() or not self.valid.any():
            self._hf = np.nan_to_num(self.h)
        else:
            _, (ii, jj) = distance_transform_edt(~self.valid, return_indices=True)
            self._hf = self.h[ii, jj]

    # ------------------------------------------------------------ I/O
    @staticmethod
    def load(yaml_path: str) -> 'HeightMap':
        """Wczytuje plik .yaml {image: plik.npy, resolution, origin: [x, y]}."""
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        npy = meta['image']
        if not os.path.isabs(npy):
            npy = os.path.join(os.path.dirname(yaml_path), npy)
        h = np.load(npy)
        g = Grid(float(meta['resolution']), float(meta['origin'][0]), float(meta['origin'][1]),
                 h.shape[0], h.shape[1])
        return HeightMap(h, g)

    def save(self, yaml_path: str):
        base = os.path.splitext(yaml_path)[0]
        np.save(base + '.npy', self.h.astype(np.float32))
        meta = {'image': os.path.basename(base) + '.npy', 'resolution': self.grid.res,
                'origin': [self.grid.origin_x, self.grid.origin_y]}
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(meta, f)

    # ------------------------------------------------------------ zapytania
    def height_at(self, x, y):
        """Interpolacja dwuliniowa h(x, y) (wektorowo); poza siatką - najbliższy brzeg."""
        g = self.grid
        fx = (np.asarray(x, float) - g.origin_x) / g.res - 0.5
        fy = (np.asarray(y, float) - g.origin_y) / g.res - 0.5
        fx = np.clip(fx, 0.0, g.width - 1.0)
        fy = np.clip(fy, 0.0, g.height - 1.0)
        j0 = np.minimum(fx.astype(int), g.width - 2) if g.width > 1 else np.zeros_like(fx, int)
        i0 = np.minimum(fy.astype(int), g.height - 2) if g.height > 1 else np.zeros_like(fy, int)
        tx, ty = fx - j0, fy - i0
        j1 = np.minimum(j0 + 1, g.width - 1)
        i1 = np.minimum(i0 + 1, g.height - 1)
        hf = self._hf
        return ((1 - ty) * ((1 - tx) * hf[i0, j0] + tx * hf[i0, j1])
                + ty * ((1 - tx) * hf[i1, j0] + tx * hf[i1, j1]))

    def is_valid_at(self, x, y):
        g = self.grid
        i, j = g.cell_of(np.atleast_1d(x), np.atleast_1d(y))
        ok = g.inside(i, j)
        out = np.zeros(ok.shape, bool)
        out[ok] = self.valid[i[ok], j[ok]]
        return out

    def smoothed(self, radius: float) -> np.ndarray:
        """Wysokość uśredniona w kwadracie o boku ~2·radius (gradient pod łazikiem)."""
        n = max(1, int(round(2 * radius / self.grid.res)) | 1)
        return uniform_filter(self._hf, size=n, mode='nearest')

    def gradient(self, radius: float = 0.0):
        """(dh/dx, dh/dy) na siatce; z wygładzeniem przy radius > 0."""
        h = self.smoothed(radius) if radius > 0 else self._hf
        gy, gx = np.gradient(h, self.grid.res)
        return gx, gy

    def roughness(self, radius: float) -> np.ndarray:
        """
        Odchylenie std. wysokości od lokalnej średniej (kamienie, uskoki, nie samo zbocze).

        Średnia ruchoma znosi trend liniowy, więc równe zbocze ma chropowatość ~0.
        """
        n = max(1, int(round(2 * radius / self.grid.res)) | 1)
        resid = self._hf - uniform_filter(self._hf, size=n, mode='nearest')
        return np.sqrt(np.maximum(uniform_filter(resid ** 2, size=n, mode='nearest'), 0.0))

    def bump_height(self, radius: float) -> np.ndarray:
        """
        Najwyższy garb w promieniu: max wysokości ponad lokalną średnią (bez trendu zbocza).

        Chropowatość (odch. std.) pojedynczej skały w oknie łazika bywa mała, a skała wyższa
        niż prześwit zawiesza łazik na podwoziu.
        """
        n = max(1, int(round(2 * radius / self.grid.res)) | 1)
        resid = self._hf - uniform_filter(self._hf, size=n, mode='nearest')
        return maximum_filter(resid, size=n, mode='nearest')

    def resample(self, grid: Grid, how: str = 'mean') -> np.ndarray:
        """Wartości h na innej siatce o rozdzielczości k razy większej ('mean' | 'max')."""
        k = int(round(grid.res / self.grid.res))
        if k < 1 or abs(k * self.grid.res - grid.res) > 1e-9 or (
                grid.origin_x, grid.origin_y) != (self.grid.origin_x, self.grid.origin_y):
            raise ValueError('resample: siatka docelowa musi być k-krotnie rzadsza, '
                             'z tym samym narożnikiem')
        H, W = grid.shape
        pad = np.full((H * k, W * k), np.nan)
        pad[:self.grid.height, :self.grid.width] = self.h
        blocks = pad.reshape(H, k, W, k)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # bloki całe z NaN -> NaN
            if how == 'max':
                return np.nanmax(blocks, axis=(1, 3))
            return np.nanmean(blocks, axis=(1, 3))
