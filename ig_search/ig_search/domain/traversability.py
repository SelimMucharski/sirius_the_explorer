"""
Mapa przejezdności: binarna maska terenu, po którym łazik może jeździć (DESIGN §6). Bez ROS.

Format jak mapy kalman_robot (kalman_nav2/maps) i nav2_map_server: obraz PNG (biały -
przejezdny, czarny - nie) + .yaml {image, resolution, origin [x, y, yaw], negate,
occupied_thresh, free_thresh, mode}. Wiersz 0 obrazu to górny brzeg (największe y), więc przy
zapisie i odczycie maska jest odwracana do konwencji siatki (wiersz rośnie z y).

Maska opisuje sam teren (nachylenie, chropowatość, garby, odstęp od krawędzi mapy wysokości,
wzniesienia ponad otoczenie - kopce o łagodnych zboczach, na których nie ma próbek),
bez promienia łazika: Nav2 dodaje obrys, a TravelModel - własny odstęp `robot_radius`.
Z wygładzenia (domknięcie i otwarcie kołem, zalanie małych dziur) zostaje jedna spójna
strefa z punktem startu - jak ręcznie rysowane mapy kalman.
"""
import math
import os

from ig_search.domain.params import TravelParams
from ig_search.domain.terrain import Grid, HeightMap
import numpy as np
from PIL import Image
from scipy import ndimage
import yaml


def _disk(radius: float, res: float) -> np.ndarray:
    n = max(1, int(round(radius / res)))
    y, x = np.ogrid[-n:n + 1, -n:n + 1]
    return x * x + y * y <= n * n


def terrain_ok(hm: HeightMap, p: TravelParams) -> np.ndarray:
    """Komórki mapy wysokości przejezdne według kryteriów TravelModel (bez odstępu łazika)."""
    gx, gy = hm.gradient(p.robot_radius)
    ok = (hm.valid & (np.hypot(gx, gy) <= math.tan(math.radians(p.slope_max_deg)))
          & (hm.roughness(p.robot_radius) <= p.roughness_max)
          & (hm.bump_height(p.robot_radius) <= p.bump_max))
    valid = np.pad(hm.valid, 1, constant_values=False)     # brzeg siatki też jest krawędzią
    edge = ndimage.distance_transform_edt(valid)[1:-1, 1:-1] * hm.grid.res
    return ok & (edge >= p.edge_margin)


def hills(hm: HeightMap, height: float = 0.25, window: float = 6.0,
          close_radius: float = 0.5) -> np.ndarray:
    """
    Wzniesienia: teren wyżej niż `height` ponad medianę wysokości w kwadracie `window`.

    Kopiec o łagodnych zboczach przechodzi kryteria nachylenia i garbów; dziury w wykrytym
    obszarze (płaski wierzchołek bywa bliżej mediany okna) są zamykane i zalewane.
    """
    n = max(3, int(round(window / hm.grid.res)) | 1)
    rel = hm._hf - ndimage.median_filter(hm._hf, size=n, mode='nearest')
    m = ndimage.binary_closing(rel > height, _disk(close_radius, hm.grid.res))
    return ndimage.binary_fill_holes(m) & hm.valid


def polygon_mask(grid: Grid, polygon) -> np.ndarray:
    """Komórki siatki, których środki leżą w wielokącie [(x, y), ...] (parzystość przecięć)."""
    cx, cy = (c.reshape(grid.shape) for c in grid.centers())
    inside = np.zeros(grid.shape, bool)
    pts = np.asarray(polygon, float)
    for (x0, y0), (x1, y1) in zip(pts, np.roll(pts, -1, axis=0)):
        if y0 == y1:
            continue
        cross = (y0 > cy) != (y1 > cy)
        x_at = x0 + (cy - y0) * (x1 - x0) / (y1 - y0)
        inside ^= cross & (cx < x_at)
    return inside


def load_exclusions(yaml_path: str):
    """Wielokąty stref wyłączonych z pliku .yaml {zones: [{name, polygon: [[x, y], ...]}]}."""
    with open(yaml_path) as f:
        meta = yaml.safe_load(f) or {}
    return [z['polygon'] for z in meta.get('zones', [])]


def buried(visual: HeightMap, collision: HeightMap, depth: float = 0.015,
           window: float = 0.5) -> np.ndarray:
    """
    Komórki mapy wysokości (wizualnej), w których próbka zapada się pod widoczny teren.

    Gazebo liczy kontakt z uproszczonym meshem kolizyjnym; tam, gdzie widoczny teren leży
    wyżej niż kolizyjny o więcej niż `depth` (w kwadracie `window`, bo próbka po upadku
    staczała się ~10 cm i ma 0,2 m długości), leżąca próbka ⌀3 cm znika w wizualnym podłożu.
    """
    g = visual.grid
    cx, cy = g.centers()
    hc = np.where(collision.is_valid_at(cx, cy), collision.height_at(cx, cy), np.nan)
    d = visual.h - hc.reshape(g.shape)
    n = max(1, int(round(window / g.res)) | 1)
    return ndimage.maximum_filter(np.nan_to_num(d, nan=np.inf), size=n) > depth


def smooth_mask(ok: np.ndarray, grid: Grid, start=(0.0, 0.0), radius: float = 0.3,
                min_hole: float = 2.0) -> np.ndarray:
    """
    Wygładzona maska: domknięcie i otwarcie kołem `radius`, dziury < `min_hole` [m²] zalane.

    Zostaje składowa z punktem `start` (miejsce startu łazika), a gdy start leży poza
    maską - największa składowa.
    """
    k = _disk(radius, grid.res)
    m = ndimage.binary_opening(ndimage.binary_closing(ok, k), k)
    holes, n = ndimage.label(~m)
    if n:
        sizes = np.bincount(holes.ravel(), minlength=n + 1) * grid.res ** 2
        small = sizes < min_hole
        small[0] = False
        m |= small[holes]
    lab, n = ndimage.label(m)
    if not n:
        return m
    i, j = grid.cell_of(start[0], start[1])
    keep = lab[i, j] if grid.inside(i, j) else 0
    if not keep:
        keep = 1 + int(np.argmax(np.bincount(lab.ravel())[1:]))
    return lab == keep


class TraversabilityMap:
    """Maska przejezdności na siatce (wiersz rośnie z y); poza siatką - nieprzejezdne."""

    def __init__(self, mask: np.ndarray, grid: Grid):
        if mask.shape != grid.shape:
            raise ValueError(f'kształt maski {mask.shape} != siatka {grid.shape}')
        self.mask = mask.astype(bool)
        self.grid = grid

    @staticmethod
    def from_heightmap(hm: HeightMap, p: TravelParams, start=(0.0, 0.0), radius: float = 0.3,
                       min_hole: float = 2.0, upsample: int = 2, hill_height: float = 0.25,
                       hill_window: float = 6.0, exclude=()) -> 'TraversabilityMap':
        """
        Maska z kryteriów terenu, wygładzona; `upsample` - gładszy brzeg na gęstszej siatce.

        `hill_height` <= 0 - bez wycinania wzniesień; `exclude` - wielokąty stref wyłączonych
        ręcznie (ramka map).
        """
        ok = terrain_ok(hm, p)
        if hill_height > 0:
            ok &= ~hills(hm, hill_height, hill_window)
        for poly in exclude:
            ok &= ~polygon_mask(hm.grid, poly)
        m = smooth_mask(ok, hm.grid, start, radius, min_hole)
        g = hm.grid
        if upsample > 1:
            fine = ndimage.zoom(m.astype(float), upsample, order=1)
            m = ndimage.gaussian_filter(fine, 0.75 * upsample) > 0.5
            g = Grid(g.res / upsample, g.origin_x, g.origin_y, m.shape[0], m.shape[1])
        return TraversabilityMap(m, g)

    # ------------------------------------------------------------ I/O
    @staticmethod
    def load(yaml_path: str) -> 'TraversabilityMap':
        """Wczytuje mapę w formacie nav2_map_server (biały = przejezdny przy negate 0)."""
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        img = meta['image']
        if not os.path.isabs(img):
            img = os.path.join(os.path.dirname(yaml_path), img)
        gray = np.asarray(Image.open(img).convert('L'), float) / 255.0
        occ = gray if meta.get('negate', 0) else 1.0 - gray
        mask = (occ < float(meta.get('free_thresh', 0.25)))[::-1]
        g = Grid(float(meta['resolution']), float(meta['origin'][0]), float(meta['origin'][1]),
                 mask.shape[0], mask.shape[1])
        return TraversabilityMap(mask, g)

    def save(self, yaml_path: str):
        base = os.path.splitext(yaml_path)[0]
        img = np.where(self.mask[::-1], 255, 0).astype(np.uint8)
        Image.fromarray(img, mode='L').save(base + '.png')
        meta = {'image': os.path.basename(base) + '.png', 'mode': 'trinary',
                'resolution': self.grid.res,
                'origin': [self.grid.origin_x, self.grid.origin_y, 0.0],
                'negate': 0, 'occupied_thresh': 0.65, 'free_thresh': 0.25}
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(meta, f, sort_keys=False)

    # ------------------------------------------------------------ zapytania
    def allowed_at(self, x, y) -> np.ndarray:
        """Czy punkty (x, y) leżą na przejezdnym terenie (wektorowo)."""
        g = self.grid
        i, j = g.cell_of(np.atleast_1d(x), np.atleast_1d(y))
        ok = g.inside(i, j)
        out = np.zeros(ok.shape, bool)
        out[ok] = self.mask[i[ok], j[ok]]
        return out

    def spawn_map(self, visual: HeightMap, collision: HeightMap, depth: float = 0.015,
                  window: float = 0.5) -> 'TraversabilityMap':
        """Miejsca na próbki: ta mapa bez komórek, w których próbka znika w podłożu."""
        g = visual.grid
        visible = ~buried(visual, collision, depth, window)
        ok = self.allowed_at(*g.centers()).reshape(g.shape) & visible
        return TraversabilityMap(ok, g)

    def area(self) -> float:
        return float(self.mask.sum()) * self.grid.res ** 2
