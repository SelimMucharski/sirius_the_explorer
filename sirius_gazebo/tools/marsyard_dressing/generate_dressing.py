#!/usr/bin/env python3
"""
Realistyczne wyposażenie Mars Yardu ERC 2025: kamienie 3D, żwir w teksturze, światło.

Teren z erc_import ma prawdziwą rzeźbę i ortofoto toru jako teksturę (~1,3 cm/px), ale
kamienie są na nim tylko namalowane. Na zdjęciach z zawodów (2026-10-27) widać:
  - „ogródki” płyt kamiennych (piaskowiec, 20-60 cm, 2-6 cm grubości), gęsto ułożonych;
  - głazy 30-80 cm (różowo-szary granit) w skupiskach przy brzegach i przy płytach;
  - pojedyncze kamienie 8-25 cm na całym torze;
  - jasne, wydłużone odłamki i listewki 10-30 cm (najbardziej podobne do próbki);
  - żwir 2-6 cm na większości powierzchni, płaty brązowego piasku;
  - światło dnia z chmurami (miękkie cienie).
Skrypt (stałe ziarno - jeden tor, jak na zawodach) generuje:
  models/erc2025_dressing/       model statyczny: płyty, głazy, kamienie, odłamki (OBJ + MTL
                                 + tekstury), kolizje tylko głazów ≥ 0,25 m (w kafelkach
                                 4 × 4 m; płyty, kamienie i odłamki - tylko wizualne);
  models/erc2025_marsyard_hd/    teren z erc2025_marsyard z teksturą 3× gęstszą i nałożonym
                                 żwirem (kamyki ~3 cm zamiast plamy 2-3 px);
  worlds/erc2025_real.sdf        świat erc2025.sdf z tymi modelami i światłem pochmurnym;
  <ig_search>/config/maps/erc2025_dressing_exclude.yaml
                                 obrysy głazów ≥ 0,3 m (strefy wyłączone z mapy przejezdności).
Obszary - ręcznie z ortofoto w ramce świata (OBSZARY niżej).

  python3 generate_dressing.py [--seed 2025] [--tex-scale 3]
"""
import argparse
import math
import os
import re
import shutil

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, '..', '..'))
TERRAIN = os.path.join(PKG, 'models', 'erc2025_marsyard')
IG_MAPS = os.path.normpath(os.path.join(PKG, '..', 'ig_search', 'config', 'maps'))

# ---------------------------------------------------------------- obszary (ramka świata, m)
# elipsy (cx, cy, rx, ry) i prostokąty (x0, x1, y0, y1) z ortofoto (ortho_world.png)
SLAB_GARDENS = [(7.0, -4.2, 4.0, 3.1), (15.7, -3.4, 3.2, 2.9)]
BOULDER_PILES = [            # pasy i usypiska kamieni: (x0, x1, y0, y1, gęstość / m²)
    (0.5, 9.5, 4.4, 5.8, 1.7),       # rząd kamieni przy grzbiecie
    (-2.3, -0.6, -9.0, 10.0, 1.3),   # lewy brzeg
    (-1.5, 2.5, -16.0, -10.5, 1.4),  # lewy dolny róg
    (8.0, 11.0, -12.6, -9.4, 2.0),   # usypisko na dole
    (7.5, 10.0, 9.0, 11.6, 1.4),     # usypisko u góry
    (-1.0, 3.0, 9.0, 12.0, 1.4),     # lewy górny róg
]
STICK_MOUNDS = [(9.6, 1.6, 2.6, 1.6), (18.0, 3.4, 2.2, 1.6), (24.6, -7.0, 1.6, 1.2)]
YARD = (-0.5, 32.0, -12.0, 9.5)      # zasięg pojedynczych kamieni i odłamków
STONE_DENSITY = 0.8                  # [1/m²] pojedyncze kamienie 7-30 cm na całym torze
PEBBLE_DENSITY = 7.0                 # [1/m²] kamyki 3-8 cm (żwir widoczny z bliska jako bryły)
START_CLEAR = 2.5                    # [m] wolne koło wokół startu łazika (0, 0)
EXCLUDE_MIN = 0.25                   # [m] głazy od tej średnicy - z kolizją i do mapy przejezdności
#  (mniejsze kamienie tylko wizualne: łazik w symulacji osiadał na 18-cm kamieniu podwoziem,
#  a koła buksowały - 2026-10-27; widzą je kamery, detektor i warstwa przeszkód Nav2)

# ---------------------------------------------------------------- kolory (sRGB 0-255)
GRANITE = [(184, 146, 138), (160, 140, 136), (196, 160, 146), (140, 128, 126)]
SANDSTONE = [(132, 112, 96), (112, 102, 96), (150, 128, 106), (98, 90, 86), (140, 132, 124),
             (146, 118, 96), (88, 84, 82), (124, 110, 98), (168, 150, 128), (116, 112, 110)]
SHARD = [(176, 162, 136), (160, 146, 120), (150, 142, 132), (184, 170, 146)]


# ================================================================ geometria
ASSETS = os.path.join(HERE, 'assets')
ROCK_TEX = {'namaqualand_stones_01': 'rock_grey', 'stone_01': 'rock_pink'}


def load_rocks():
    """Fotoskany kamieni Poly Haven (assets/rocks, CC0): [(mesh z UV, materiał)]."""
    out = []
    d = os.path.join(ASSETS, 'rocks')
    for f in sorted(os.listdir(d)):
        if f.endswith('.obj'):
            m = trimesh.load(os.path.join(d, f), force='mesh', process=False)
            m.apply_translation(-m.bounds.mean(axis=0))
            out.append((m, ROCK_TEX[f.rsplit('_', 1)[0]]))
    return out


def rock_instance(rng, rocks, size, pink=0.55):
    """Kopia fotoskanu przeskalowana do największego wymiaru `size` (lekko anizotropowo)."""
    pool = [r for r in rocks if (r[1] == 'rock_pink') == (rng.random() < pink)] or rocks
    m, mat = pool[rng.integers(len(pool))]
    m = m.copy()
    m.apply_transform(trimesh.transformations.random_rotation_matrix(rng.random(3)))
    m.apply_scale(size / max(m.extents) * rng.uniform(0.9, 1.1, 3))
    m.apply_translation(-m.bounds.mean(axis=0))
    return m, mat

def rock_mesh(rng, size, flat=(0.6, 0.9), n=26, jitter=0.12):
    """Kanciasty kamień: otoczka wypukła punktów z elipsoidy (wymiary size × [1, a, b])."""
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    a, b = rng.uniform(*flat), rng.uniform(flat[0] * 0.7, flat[0])
    p = d * np.array([1.0, a, b]) * (0.5 * size) * rng.uniform(1 - jitter, 1 + jitter, (n, 1))
    m = trimesh.Trimesh(p, ConvexHull(p).simplices)
    m.fix_normals()
    return m


def slab_mesh(rng, length, thick):
    """Płyta: nieregularny wielokąt wypukły wyciągnięty na grubość."""
    k = rng.integers(6, 10)
    th = np.sort(rng.uniform(0, 2 * np.pi, k))
    w = rng.uniform(0.5, 0.85)
    r = rng.uniform(0.75, 1.0, k)
    pts = np.c_[0.5 * length * r * np.cos(th), 0.5 * length * w * r * np.sin(th)]
    pts = pts[ConvexHull(pts).vertices]
    p = np.r_[np.c_[pts, np.full(len(pts), -thick / 2)],
              np.c_[pts * rng.uniform(0.9, 0.98), np.full(len(pts), thick / 2)]]
    m = trimesh.Trimesh(p, ConvexHull(p).simplices)
    m.fix_normals()
    return m


def shard_mesh(rng, length):
    """Odłamek / listewka: wydłużony, lekko nieregularny prostopadłościan."""
    w = rng.uniform(0.02, 0.045)
    t = rng.uniform(0.008, 0.02)
    p = np.array([[sx * length / 2, sy * w / 2, sz * t / 2]
                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    p += rng.normal(0, 0.12, p.shape) * np.array([length * 0.1, w, t])
    m = trimesh.Trimesh(p, ConvexHull(p).simplices)
    m.fix_normals()
    return m


def box_uv(m, scale):
    """UV z rzutu na płaszczyznę prostopadłą do dominującej osi normalnej (bez szwów w ścianie)."""
    v = m.vertices[m.faces].reshape(-1, 3)
    n = np.repeat(m.face_normals, 3, axis=0)
    ax = np.argmax(np.abs(n), axis=1)
    uv = np.where(ax[:, None] == 0, v[:, [1, 2]], np.where(ax[:, None] == 1, v[:, [0, 2]],
                                                            v[:, [0, 1]]))
    return v, uv / scale


class Obj:
    """Zbiera trójkąty (wierzchołki nie dzielone - płaskie cieniowanie ścian) według materiałów."""

    def __init__(self):
        self.parts = {}

    def add(self, mesh, T, material, uv_scale, window=None):
        """UV: z meshu (fotoskan), inaczej rzut / uv_scale; window = (u0, v0, du, dv)."""
        m = mesh.copy()
        m.apply_transform(T)
        uv0 = getattr(m.visual, 'uv', None)
        if uv0 is not None and len(uv0) == len(m.vertices):
            v = m.vertices[m.faces].reshape(-1, 3)
            uv = np.asarray(uv0)[m.faces].reshape(-1, 2)
        else:
            v, uv = box_uv(m, uv_scale)
            if window is not None:
                uv = window[:2] + (uv - uv.min(0)) / max(np.ptp(uv, 0).max(), 1e-9) * window[2:]
        n = np.repeat(m.face_normals, 3, axis=0)
        self.parts.setdefault(material, []).append((v, uv, n))

    def write(self, path, mtl_name):
        with open(path, 'w') as f:
            if mtl_name:
                f.write(f'mtllib {mtl_name}\n')
            base = 1
            for mat, chunks in self.parts.items():
                v = np.concatenate([c[0] for c in chunks])
                uv = np.concatenate([c[1] for c in chunks])
                n = np.concatenate([c[2] for c in chunks])
                f.write(''.join(f'v {a:.4f} {b:.4f} {c:.4f}\n' for a, b, c in v))
                f.write(''.join(f'vt {a:.4f} {b:.4f}\n' for a, b in uv))
                f.write(''.join(f'vn {a:.4f} {b:.4f} {c:.4f}\n' for a, b, c in n))
                if mtl_name:
                    f.write(f'usemtl {mat}\n')
                idx = np.arange(len(v)).reshape(-1, 3) + base
                f.write(''.join(f'f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n' for a, b, c in idx))
                base += len(v)


# ================================================================ tekstury
def noise(rng, shape, scales=(4, 16, 64), weights=(0.5, 0.3, 0.2)):
    """Szum wartości (suma oktaw), średnia 0, odchylenie ~1."""
    out = np.zeros(shape)
    for s, w in zip(scales, weights):
        g = rng.normal(size=(shape[0] // s + 2, shape[1] // s + 2))
        img = np.asarray(Image.fromarray(g.astype(np.float32)).resize(
            (shape[1] + 2 * s, shape[0] + 2 * s), Image.BICUBIC))[s:s + shape[0], s:s + shape[1]]
        out += w * img
    return out / out.std()


def stone_texture(rng, kind, size=512):
    """Albedo w odcieniach szarości (kolor nadaje Kd materiału): ziarno granitu albo warstwy."""
    n = noise(rng, (size, size), (2, 8, 32, 128), (0.35, 0.3, 0.2, 0.15))
    if kind == 'granite':
        speck = rng.random((size, size))
        g = 0.85 + 0.07 * n
        g[speck < 0.06] *= 0.55                     # ciemne ziarna (biotyt)
        g[speck > 0.96] *= 1.15                     # jasne (skaleń)
    elif kind == 'sandstone':
        y = np.arange(size)[:, None] / size
        strata = np.sin(2 * np.pi * (y * 9 + 0.3 * noise(rng, (size, size), (64, 128), (0.6, 0.4))))
        g = 0.85 + 0.06 * n + 0.05 * strata
    else:                                           # odłamek / listewka - drobne włókna
        x = np.arange(size)[None, :] / size
        grain = noise(rng, (size, size), (1, 4), (0.5, 0.5))
        g = 0.9 + 0.04 * n + 0.03 * np.sin(2 * np.pi * (x * 40 + grain))
    return Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8)).convert('RGB')


def gravel_detail(rng, h, w, px_per_m, row0=0):
    """
    Mnożnik jasności żwiru (średnia ~1): kamyki jako komórki Voronoia (~2-4 cm) cieniowane jak
    wypukłe kopułki oświetlone z jednego kierunku, z cieniem w szczelinie po stronie
    przeciwnej do światła, z jasnością i odcieniem (szary - różowy granit) kamyka.
    """
    cell = 0.028 * px_per_m                          # [px] rozstaw kamyków
    gy, gx = int(h / cell) + 3, int(w / cell) + 3
    srng = np.random.default_rng(rng)
    jit = srng.uniform(0.1, 0.9, (gy, gx, 2))
    size = srng.uniform(0.75, 1.25, (gy, gx))       # promień kopułki względem komórki
    bright = srng.normal(1.0, 0.13, (gy, gx))
    warm = srng.normal(0, 1.0, (gy, gx))[..., None] * np.array([0.035, 0.0, -0.035])
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = ys / cell, xs / cell
    iy, ix = np.floor(cy).astype(int), np.floor(cx).astype(int)
    best = np.full((h, w), np.inf, np.float32)
    second = np.full((h, w), np.inf, np.float32)
    oy = np.zeros((h, w), int)
    ox = np.zeros((h, w), int)
    vy = np.zeros((h, w), np.float32)
    vx = np.zeros((h, w), np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ny, nx = np.clip(iy + dy, 0, gy - 1), np.clip(ix + dx, 0, gx - 1)
            sy, sx = ny + jit[ny, nx, 0], nx + jit[ny, nx, 1]
            d = ((sy - cy) ** 2 + (sx - cx) ** 2) / size[ny, nx] ** 2
            closer = d < best
            second = np.where(closer, best, np.minimum(second, d))
            best = np.where(closer, d, best)
            oy, ox = np.where(closer, ny, oy), np.where(closer, nx, ox)
            vy = np.where(closer, (cy - sy) / size[ny, nx], vy)
            vx = np.where(closer, (cx - sx) / size[ny, nx], vx)
    # kopułka: normalna z położenia względem środka kamyka (promień ~0,6 komórki)
    nx_, ny_ = vx / 0.62, vy / 0.62
    nz = np.sqrt(np.clip(1.0 - nx_ ** 2 - ny_ ** 2, 0.0, 1.0))
    light = np.array([-0.45, -0.35, 0.82])
    lam = np.clip(light[0] * nx_ + light[1] * ny_ + light[2] * nz, 0.0, 1.0)
    edge = np.sqrt(second) - np.sqrt(best)
    gap = np.clip(edge / 0.08, 0, 1)
    shade = 0.55 + 0.55 * lam
    shade = shade * (0.45 + 0.55 * gap)               # szczeliny między kamykami
    speck = 1.0 + 0.05 * srng.normal(size=(h, w)).astype(np.float32)
    lum = shade * bright[oy, ox] * speck
    lum /= lum.mean()
    return lum[..., None] * (1.0 + warm[oy, ox])


TEX_PX_PER_M = 3856 / 51.12                         # ortofoto: 3856 px na 1/0,019562 m


def photo_detail(path, tile_m, px_per_m, shape, r0, sigma_m=0.15):
    """
    Detal z fotograficznej tekstury (kafelek tile_m × tile_m): jasność / jej rozmycie
    (średnia ~1, bez koloru i bez niskich częstotliwości) dla wierszy r0.. obrazu `shape`.
    """
    from scipy.ndimage import gaussian_filter
    key = (path, tile_m, px_per_m)
    if key not in _DETAIL:
        n = max(8, int(round(tile_m * px_per_m)))
        t = np.asarray(Image.open(path).convert('L').resize((n, n), Image.LANCZOS), np.float32)
        low = gaussian_filter(t, sigma_m * px_per_m, mode='wrap')
        _DETAIL[key] = t / np.maximum(low, 1.0)
    d = _DETAIL[key]
    n = d.shape[0]
    rows = (np.arange(r0, r0 + shape[0]) % n)[:, None]
    cols = (np.arange(shape[1]) % n)[None, :]
    return d[rows, cols]


_DETAIL = {}


def hd_texture(src, out, scale, seed, src_px_per_m=TEX_PX_PER_M):
    """
    Ortofoto ×scale z fotograficznym detalem: żwir (Poly Haven „Gravel Floor 02”, kamyki
    ~3 cm) na jasnych, mało zielonych miejscach, piasek („Brown Mud Dry”) na ciemnych,
    roślinność bez zmian. Kolor zostaje z ortofoto.
    """
    im = Image.open(src).convert('RGB')
    W, H = im.size
    px_per_m = src_px_per_m * scale
    big = im.resize((W * scale, H * scale), Image.BICUBIC)
    a = np.asarray(im, np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = a.mean(-1)
    green = g - 0.5 * (r + b)
    plant = np.clip(green / 0.06, 0.0, 1.0)
    gravel = np.clip((lum - 0.36) / 0.12, 0.0, 1.0) * (1 - plant)
    sand = (1 - gravel) * (1 - plant)

    def up(x):
        return np.asarray(Image.fromarray((x * 255).astype(np.uint8)).resize(
            big.size, Image.BILINEAR), np.float32) / 255.0
    gravel, sand = up(gravel), up(sand)
    res = np.asarray(big, np.float32) / 255.0
    rows = 512
    for r0 in range(0, res.shape[0], rows):
        r1 = min(r0 + rows, res.shape[0])
        sh = (r1 - r0, res.shape[1])
        dg = photo_detail(os.path.join(ASSETS, 'gravel_floor_02_diff.jpg'), 3.0, px_per_m, sh,
                          r0)
        ds = photo_detail(os.path.join(ASSETS, 'brown_mud_dry_diff.jpg'), 2.0, px_per_m, sh, r0)
        wg, ws = 1.0 * gravel[r0:r1], 0.6 * sand[r0:r1]
        m = 1.0 + wg * (dg - 1.0) + ws * (ds - 1.0)
        res[r0:r1] *= m[..., None]
    Image.fromarray((np.clip(res, 0, 1) * 255).astype(np.uint8)).save(out, quality=92)


# ================================================================ teren i rozmieszczenie
class Ground:
    """Wysokość i normalna mesha wizualnego terenu (promień z góry)."""

    CELL = 0.5                                      # [m] siatka indeksu trójkątów

    def __init__(self, obj):
        m = trimesh.load(obj, force='mesh', process=False)
        self.tri = m.vertices[m.faces]                 # (F, 3, 3)
        self.normals = m.face_normals
        lo = self.tri[:, :, :2].min(1)
        hi = self.tri[:, :, :2].max(1)
        self.origin = lo.min(0)
        self.index = {}
        a = np.floor((lo - self.origin) / self.CELL).astype(int)
        b = np.floor((hi - self.origin) / self.CELL).astype(int)
        for f in range(len(self.tri)):
            for i in range(a[f, 0], b[f, 0] + 1):
                for j in range(a[f, 1], b[f, 1] + 1):
                    self.index.setdefault((i, j), []).append(f)

    def at(self, xy):
        """Wysokość i normalna najwyższego trójkąta nad (x, y); nan poza meshem."""
        xy = np.atleast_2d(xy)
        z = np.full(len(xy), np.nan)
        n = np.tile([0.0, 0.0, 1.0], (len(xy), 1))
        for k, (x, y) in enumerate(xy):
            key = tuple(np.floor((np.array([x, y]) - self.origin) / self.CELL).astype(int))
            fs = np.array(self.index.get(key, []), int)
            if not len(fs):
                continue
            t = self.tri[fs]
            v0, v1, v2 = t[:, 0, :2], t[:, 1, :2], t[:, 2, :2]
            d = (v1[:, 1] - v2[:, 1]) * (v0[:, 0] - v2[:, 0]) + \
                (v2[:, 0] - v1[:, 0]) * (v0[:, 1] - v2[:, 1])
            d = np.where(np.abs(d) < 1e-12, 1e-12, d)
            l0 = ((v1[:, 1] - v2[:, 1]) * (x - v2[:, 0]) + (v2[:, 0] - v1[:, 0]) * (y - v2[:, 1])) / d
            l1 = ((v2[:, 1] - v0[:, 1]) * (x - v2[:, 0]) + (v0[:, 0] - v2[:, 0]) * (y - v2[:, 1])) / d
            l2 = 1 - l0 - l1
            ok = (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
            if not ok.any():
                continue
            zz = l0 * t[:, 0, 2] + l1 * t[:, 1, 2] + l2 * t[:, 2, 2]
            i = np.flatnonzero(ok)[np.argmax(zz[ok])]
            z[k] = zz[i]
            nn = self.normals[fs[i]]
            n[k] = nn if nn[2] > 0 else -nn
        return z, n


def in_ellipse(p, e):
    return ((p[0] - e[0]) / e[2]) ** 2 + ((p[1] - e[1]) / e[3]) ** 2 <= 1.0


def pose_on_ground(xyz, normal, yaw, tilt_rng=None, tilt=0.0):
    """Macierz: obrót wokół z, ułożenie na normalnej terenu (+ losowe przechylenie)."""
    z = normal / np.linalg.norm(normal)
    if tilt_rng is not None and tilt > 0:
        z = z + tilt_rng.normal(0, tilt, 3) * np.array([1, 1, 0])
        z /= np.linalg.norm(z)
    T_align = trimesh.geometry.align_vectors([0, 0, 1], z)
    T = T_align @ trimesh.transformations.rotation_matrix(yaw, [0, 0, 1])
    T[:3, 3] = xyz
    return T


def place(args):
    rng = np.random.default_rng(args.seed)
    ground = Ground(os.path.join(TERRAIN, 'meshes', 'erc2025_marsyard.obj'))
    items = []                       # (rodzaj, mesh, T, materiał, kolizja, (x, y, r), okno UV)
    occupied = []                                    # (x, y, r) - bez nakładania dużych
    rocks = load_rocks()

    def rock(size):
        """Kamień od 10 cm - fotoskan; mniejszy - proceduralny (taniej, i tak 2-3 px z daleka)."""
        if size >= 0.1:
            return rock_instance(rng, rocks, size)
        return rock_mesh(rng, size, n=16), f'granite{rng.integers(len(GRANITE))}'

    def free(x, y, r, slack=0.0):
        if math.hypot(x, y) < START_CLEAR + r:
            return False
        return all(math.hypot(x - a, y - b) >= (r + c) * (1 - slack) for a, b, c in occupied)

    def put(kind, mesh, x, y, sink, tilt, mat, collide, r=None, slack=0.0, window=None):
        z, n = ground.at([x, y])
        if not np.isfinite(z[0]):
            return False
        if r is not None:
            if not free(x, y, r, slack):
                return False
            occupied.append((x, y, r))
        ext = mesh.bounds[1][2] - mesh.bounds[0][2]
        zc = z[0] - mesh.bounds[0][2] - sink * ext
        T = pose_on_ground([x, y, zc], n[0] if kind == 'slab' else np.array([0, 0, 1.0]),
                           rng.uniform(0, 2 * np.pi), rng, tilt)
        items.append((kind, mesh, T, mat, collide, (x, y, r or 0.0), window))
        return True

    def lognormal(mu, lo, hi, s=0.35):
        return float(np.clip(rng.lognormal(np.log(mu), s), lo, hi))

    # płyty w ogródkach (pokrycie ~45 %), między nimi głazy i odłamki
    for e in SLAB_GARDENS:
        area = math.pi * e[2] * e[3]
        target, covered, tries = 0.45 * area, 0.0, 0
        while covered < target and tries < 6000:
            tries += 1
            x, y = rng.uniform(e[0] - e[2], e[0] + e[2]), rng.uniform(e[1] - e[3], e[1] + e[3])
            if not in_ellipse((x, y), e):
                continue
            L = lognormal(0.38, 0.18, 0.7)
            m = slab_mesh(rng, L, rng.uniform(0.025, 0.06))
            # płyta: fragment fotoskanu kamienia (płaskie szaro-brązowe łaty) w odcieniu
            win = np.r_[rng.uniform(0.0, 0.86, 2), 0.14, 0.14]
            # bez kolizji: łazik w symulacji więzł między płytami na brzegu ogródka (2026-10-27),
            # a prawdziwy po nich przejeżdża; wystają 1-3 cm nad grunt
            if put('slab', m, x, y, 0.45, 0.04, f'slab{rng.integers(len(SANDSTONE))}', False,
                   r=0.38 * L, slack=0.25, window=win):
                covered += ConvexHull(m.vertices[:, :2]).volume
        for _ in range(int(0.35 * area)):
            x, y = rng.uniform(e[0] - e[2], e[0] + e[2]), rng.uniform(e[1] - e[3], e[1] + e[3])
            if in_ellipse((x, y), e):
                s = lognormal(0.32, 0.15, 0.7)
                m, mat = rock(s)
                put('boulder', m, x, y, 0.25, 0.0, mat, s >= EXCLUDE_MIN, r=0.4 * s)
        for _ in range(int(1.2 * area)):
            x, y = rng.uniform(e[0] - e[2], e[0] + e[2]), rng.uniform(e[1] - e[3], e[1] + e[3])
            if in_ellipse((x, y), e):
                put('shard', shard_mesh(rng, lognormal(0.18, 0.08, 0.35)), x, y, 0.2, 0.15,
                    f'shard{rng.integers(len(SHARD))}', False)
        # duże głazy na obrzeżu ogródka
        for k in range(rng.integers(4, 8)):
            th = rng.uniform(0, 2 * np.pi)
            x, y = e[0] + 1.05 * e[2] * math.cos(th), e[1] + 1.05 * e[3] * math.sin(th)
            s = rng.uniform(0.45, 0.85)
            m, mat = rock(s)
            put('boulder', m, x, y, 0.25, 0.0, mat, s >= EXCLUDE_MIN, r=0.45 * s)

    # usypiska kamieni
    for x0, x1, y0, y1, dens in BOULDER_PILES:
        for _ in range(int(dens * (x1 - x0) * (y1 - y0))):
            x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
            s = lognormal(0.28, 0.12, 0.75)
            m, mat = rock(s)
            put('boulder', m, x, y, 0.25, 0.0, mat, s >= EXCLUDE_MIN, r=0.4 * s, slack=0.3)

    # pojedyncze kamienie i odłamki na całym torze, odłamki gęściej na kopcach
    x0, x1, y0, y1 = YARD
    area = (x1 - x0) * (y1 - y0)
    for _ in range(int(STONE_DENSITY * area)):
        x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
        s = lognormal(0.14, 0.07, 0.3)
        m, mat = rock(s)
        put('stone', m, x, y, 0.3, 0.0, mat, False, r=0.4 * s)
    for _ in range(int(0.1 * area)):
        x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
        put('shard', shard_mesh(rng, lognormal(0.17, 0.08, 0.35)), x, y, 0.2, 0.15,
            f'shard{rng.integers(len(SHARD))}', False)
    # kamyki: proceduralne (kilkanaście trójkątów), bez sprawdzania nakładania - jak żwir
    for _ in range(int(PEBBLE_DENSITY * area)):
        x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
        m = rock_mesh(rng, lognormal(0.045, 0.03, 0.08), flat=(0.5, 0.85), n=10, jitter=0.2)
        put('pebble', m, x, y, 0.4, 0.0, f'granite{rng.integers(len(GRANITE))}', False)
    for e in STICK_MOUNDS:
        for _ in range(int(1.5 * math.pi * e[2] * e[3])):
            x, y = rng.uniform(e[0] - e[2], e[0] + e[2]), rng.uniform(e[1] - e[3], e[1] + e[3])
            if in_ellipse((x, y), e):
                put('shard', shard_mesh(rng, lognormal(0.2, 0.08, 0.35)), x, y, 0.2, 0.25,
                    f'shard{rng.integers(len(SHARD))}', False)
    return items


# ================================================================ zapis
def write_dressing(items, args):
    out = os.path.join(PKG, 'models', 'erc2025_dressing')
    meshes = os.path.join(out, 'meshes')
    if os.path.islink(meshes):       # siatki przeniesione na inny dysk: czyść cel dowiązania
        for f in os.listdir(meshes):
            os.remove(os.path.join(meshes, f))
        for f in os.listdir(out):
            if f != 'meshes':
                os.remove(os.path.join(out, f))
    else:
        if os.path.isdir(out):
            shutil.rmtree(out)
        os.makedirs(meshes)
    rng = np.random.default_rng(args.seed + 1)
    tex = {'granite': 'granite.png', 'sandstone': 'sandstone.png', 'shard': 'shard.png'}
    for kind, name in tex.items():
        stone_texture(rng, kind).save(os.path.join(meshes, name))
    for f in ('namaqualand_stones_01_diff.jpg', 'stone_01_diff.jpg'):
        shutil.copy(os.path.join(ASSETS, 'rocks', f), meshes)
    # płyty: atlas fotoskanu w skali szarości (ziarno kamienia), odcień nadaje Kd materiału
    gray = np.asarray(Image.open(os.path.join(ASSETS, 'rocks', 'namaqualand_stones_01_diff.jpg'))
                      .convert('L'), np.float32)
    gray = np.clip(0.8 + 0.2 * (gray - gray.mean()) / gray.std(), 0, 1)
    Image.fromarray((gray * 255).astype(np.uint8)).convert('RGB').save(
        os.path.join(meshes, 'slab_gray.jpg'), quality=90)
    with open(os.path.join(meshes, 'dressing.mtl'), 'w') as f:
        for base, cols in (('granite', GRANITE), ('shard', SHARD)):
            for i, c in enumerate(cols):
                kd = ' '.join(f'{v / 255:.3f}' for v in c)
                f.write(f'newmtl {base}{i}\nKa {kd}\nKd {kd}\nKs 0.03 0.03 0.03\nNs 5\n'
                        f'illum 1\nmap_Kd {tex[base]}\n\n')
        for i, c in enumerate(SANDSTONE):         # odcień płyty × ziarno z fotoskanu
            kd = ' '.join(f'{min(1.0, 1.15 * v / 255):.3f}' for v in c)
            f.write(f'newmtl slab{i}\nKa {kd}\nKd {kd}\nKs 0.03 0.03 0.03\nNs 5\n'
                    f'illum 1\nmap_Kd slab_gray.jpg\n\n')
        for mat, img in (('rock_grey', 'namaqualand_stones_01_diff.jpg'),
                         ('rock_pink', 'stone_01_diff.jpg')):
            f.write(f'newmtl {mat}\nKa 1 1 1\nKd 1 1 1\nKs 0.04 0.04 0.04\nNs 8\n'
                    f'illum 1\nmap_Kd {img}\n\n')
    visual = Obj()
    tiles = {}
    for kind, mesh, T, mat, collide, _, win in items:
        visual.add(mesh, T, mat, 0.25 if kind != 'shard' else 0.4, win)
        if collide:
            key = (int(math.floor(T[0, 3] / 4)), int(math.floor(T[1, 3] / 4)))
            tiles.setdefault(key, []).append((mesh.convex_hull, T))
    visual.write(os.path.join(meshes, 'dressing.obj'), 'dressing.mtl')
    cols = []
    for (i, j), ms in sorted(tiles.items()):
        name = f'collision_{i}_{j}.obj'.replace('-', 'm')
        col = Obj()          # z normalnymi i UV: DART (OdeMesh) czyta normalne z aiScene
        for mesh, T in ms:
            col.add(mesh, T, 'collision', 1.0)
        col.write(os.path.join(meshes, name), None)
        cols.append(f"""      <collision name="{name[:-4]}">
        <geometry><mesh><uri>meshes/{name}</uri></mesh></geometry>
        <surface><friction><ode><mu>0.9</mu><mu2>0.9</mu2></ode></friction></surface>
      </collision>""")
    with open(os.path.join(out, 'model.sdf'), 'w') as f:
        f.write(f"""<?xml version="1.0"?>
<!-- Generated by sirius_gazebo/tools/marsyard_dressing/generate_dressing.py
     (seed {args.seed}); do not edit by hand. -->
<sdf version="1.9">
  <model name="erc2025_dressing">
    <static>true</static>
    <link name="link">
{chr(10).join(cols)}
      <visual name="visual">
        <geometry><mesh><uri>meshes/dressing.obj</uri></mesh></geometry>
      </visual>
    </link>
  </model>
</sdf>
""")
    with open(os.path.join(out, 'model.config'), 'w') as f:
        f.write("""<?xml version="1.0"?>
<model>
  <name>ERC 2025 Marsyard dressing</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <description>Kamienie, płyty i odłamki Mars Yardu (generate_dressing.py).</description>
</model>
""")
    counts = {}
    for it in items:
        counts[it[0]] = counts.get(it[0], 0) + 1
    print('obiekty:', counts, '; kafelki kolizji:', len(tiles))
    # głazy ≥ EXCLUDE_MIN - strefy wyłączone z mapy przejezdności (ośmiokąty)
    zones = []
    for kind, mesh, T, _, _, (x, y, r), _ in items:
        ext = mesh.extents
        if kind == 'boulder' and max(ext) >= EXCLUDE_MIN:
            rr = 0.5 * max(ext[0], ext[1])
            poly = [[round(float(x + rr * math.cos(a)), 2), round(float(y + rr * math.sin(a)), 2)]
                    for a in np.arange(8) * np.pi / 4]
            zones.append(f'  - name: głaz {2 * rr:.2f} m\n    polygon: {poly}')
    with open(os.path.join(IG_MAPS, 'erc2025_dressing_exclude.yaml'), 'w') as f:
        f.write('# Głazy z sirius_gazebo/models/erc2025_dressing (generate_dressing.py) - strefy\n'
                '# wyłączone z mapy przejezdności (traversability_map --exclude).\nzones:\n'
                + '\n'.join(zones) + '\n')
    print('głazy do mapy przejezdności:', len(zones))
    # obrysy wszystkich kamieni i płyt - spawner nie wbija w nie pionowych próbek
    with open(os.path.join(out, 'footprints.yaml'), 'w') as f:
        f.write('# Koła (x, y, r) [m] kamieni i płyt (generate_dressing.py); probe_supervisor\n'
                '# spawner: upright_keep_out. shards: środki odłamków.\ndiscs:\n')
        for kind, mesh, T, _, _, _, _ in items:
            if kind not in ('shard', 'pebble'):
                b2 = mesh.copy().apply_transform(T).bounds
                f.write(f'  - [{(b2[0][0] + b2[1][0]) / 2:.3f}, {(b2[0][1] + b2[1][1]) / 2:.3f}, '
                        f'{0.5 * max(b2[1][0] - b2[0][0], b2[1][1] - b2[0][1]):.3f}]\n')
        # odłamki (x, y, z środka) - wydłużone, jasne: fałszywe cele dla zbiorów detektora
        f.write('shards:\n')
        for kind, mesh, T, _, _, _, _ in items:
            if kind == 'shard':
                c = mesh.copy().apply_transform(T).bounds.mean(axis=0)
                f.write(f'  - [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}]\n')


def write_terrain_hd(args):
    out = os.path.join(PKG, 'models', 'erc2025_marsyard_hd')
    meshes = os.path.join(out, 'meshes')
    os.makedirs(meshes, exist_ok=True)
    src = os.path.join(TERRAIN, 'meshes')
    for name in ('erc2025_marsyard.obj', 'erc2025_marsyard_collision.obj'):
        dst = os.path.join(meshes, name)
        if not os.path.exists(dst):
            os.symlink(os.path.relpath(os.path.join(src, name), meshes), dst)
    with open(os.path.join(meshes, 'erc2025_marsyard.mtl'), 'w') as f:
        f.write('newmtl terrain\nKa 1 1 1\nKd 1 1 1\nKs 0.03 0.03 0.03\nillum 1\n'
                'map_Kd hires-gravel.jpg\n')
    if not args.skip_texture:
        hd_texture(os.path.join(src, 'hires-local-rot.jpg'),
                   os.path.join(meshes, 'hires-gravel.jpg'), args.tex_scale, args.seed)
    sdf = open(os.path.join(TERRAIN, 'model.sdf')).read()
    sdf = sdf.replace('<model name="erc2025_marsyard">', '<model name="erc2025_marsyard_hd">')
    with open(os.path.join(out, 'model.sdf'), 'w') as f:
        f.write(sdf.replace('Generated by sirius_gazebo/tools/erc_import/unity_to_gazebo.py',
                            'Copy of erc2025_marsyard with the gravel texture '
                            '(marsyard_dressing/generate_dressing.py)'))
    with open(os.path.join(out, 'model.config'), 'w') as f:
        f.write(open(os.path.join(TERRAIN, 'model.config')).read().replace(
            'ERC 2025 Marsyard', 'ERC 2025 Marsyard HD'))


def write_world(args):
    src = open(os.path.join(PKG, 'worlds', 'erc2025.sdf')).read()
    head = ('<!-- ERC 2025 Marsyard, realistic variant: generated by\n'
            '     sirius_gazebo/tools/marsyard_dressing/generate_dressing.py from erc2025.sdf\n'
            '     (terrain with gravel texture, stones/slabs/shards, sky with clouds).\n'
            '     Do not edit by hand; regenerate instead. -->\n')
    w = re.sub(r'<!--.*?-->', head.strip(), src, count=1, flags=re.S)
    w = w.replace('<world name="erc2025">', '<world name="erc2025">')
    w = w.replace('<uri>model://erc2025_marsyard</uri>', '<uri>model://erc2025_marsyard_hd</uri>')
    w = re.sub(r'<scene>.*?</scene>', """<scene>
      <!-- dzień z chmurami: niebo proceduralne ogre2 (chmury nad horyzontem), słońce dość
           nisko - dłuższe cienie kamieni; rozproszone słabsze niż słońce, inaczej obraz płaski
           (porównanie wariantów 2026-10-27; GI VCT bez widocznej różnicy, +60 % czasu) -->
      <ambient>0.36 0.35 0.34 1</ambient>
      <background>0.72 0.74 0.77 1</background>
      <shadows>true</shadows>
      <sky></sky>
    </scene>""", w, count=1, flags=re.S)
    w = re.sub(r'(<light type="directional" name="sun">.*?<diffuse>)[^<]*(</diffuse>)',
               r'\g<1>0.95 0.92 0.86 1\g<2>', w, count=1, flags=re.S)
    w = re.sub(r'(<light type="directional" name="sun">.*?<direction>)[^<]*(</direction>)',
               r'\g<1>-0.5 0.35 -0.8\g<2>', w, count=1, flags=re.S)
    inc = """    <include>
      <uri>model://erc2025_dressing</uri>
      <name>dressing</name>
      <pose>0 0 0 0 0 0</pose>
    </include>

"""
    i = w.index('    <!-- Fallback ground')
    w = w[:i] + inc + w[i:]
    with open(os.path.join(PKG, 'worlds', 'erc2025_real.sdf'), 'w') as f:
        f.write(w)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--tex-scale', type=int, default=3)
    ap.add_argument('--skip-texture', action='store_true', help='bez (długiego) pieczenia żwiru')
    args = ap.parse_args()
    items = place(args)
    write_dressing(items, args)
    write_terrain_hd(args)
    write_world(args)


if __name__ == '__main__':
    main()
