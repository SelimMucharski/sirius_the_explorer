"""
Mapa wysokości z mesha terenu (OBJ, oś z w górę) przez rasteryzację trójkątów. Bez ROS.

Dla każdej komórki: maksimum z wysokości trójkątów pokrywających jej środek
(maksimum - przy nawisach i kamieniach liczy się górna powierzchnia).
"""
from ig_search.domain.terrain import Grid, HeightMap
import numpy as np


def read_obj(path: str):
    """Wierzchołki (N, 3) i trójkąty (M, 3); wielokąty triangulowane wachlarzem."""
    verts, tris = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                verts.append([float(v) for v in line.split()[1:4]])
            elif line.startswith('f '):
                idx = [int(tok.split('/')[0]) for tok in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                for k in range(1, len(idx) - 1):
                    tris.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(verts, float), np.asarray(tris, np.int64)


def rasterize(verts: np.ndarray, tris: np.ndarray, res: float, margin: float = 0.0) -> HeightMap:
    """Mapa wysokości o rozdzielczości `res` pokrywająca rzut mesha (+ margines)."""
    lo = verts[:, :2].min(axis=0) - margin
    hi = verts[:, :2].max(axis=0) + margin
    W = int(np.ceil((hi[0] - lo[0]) / res))
    H = int(np.ceil((hi[1] - lo[1]) / res))
    grid = Grid(res, float(lo[0]), float(lo[1]), H, W)
    h = np.full(H * W, -np.inf)

    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    tmin = np.minimum(np.minimum(a, b), c)
    tmax = np.maximum(np.maximum(a, b), c)
    j0 = np.floor((tmin[:, 0] - lo[0]) / res - 0.5).astype(int) + 1
    i0 = np.floor((tmin[:, 1] - lo[1]) / res - 0.5).astype(int) + 1
    j1 = np.floor((tmax[:, 0] - lo[0]) / res - 0.5).astype(int)
    i1 = np.floor((tmax[:, 1] - lo[1]) / res - 0.5).astype(int)
    det = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    good = np.abs(det) > 1e-12
    span_j = np.maximum(j1 - j0 + 1, 0)
    span_i = np.maximum(i1 - i0 + 1, 0)
    for dj in range(int(span_j.max(initial=0))):
        for di in range(int(span_i.max(initial=0))):
            m = good & (dj < span_j) & (di < span_i)
            if not m.any():
                continue
            jj, ii = j0[m] + dj, i0[m] + di
            px = lo[0] + (jj + 0.5) * res
            py = lo[1] + (ii + 0.5) * res
            am, bm, cm, dm = a[m], b[m], c[m], det[m]
            l1 = ((bm[:, 1] - cm[:, 1]) * (px - cm[:, 0])
                  + (cm[:, 0] - bm[:, 0]) * (py - cm[:, 1])) / dm
            l2 = ((cm[:, 1] - am[:, 1]) * (px - cm[:, 0])
                  + (am[:, 0] - cm[:, 0]) * (py - cm[:, 1])) / dm
            l3 = 1.0 - l1 - l2
            inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9) & grid.inside(ii, jj)
            z = l1 * am[:, 2] + l2 * bm[:, 2] + l3 * cm[:, 2]
            np.maximum.at(h, (ii * W + jj)[inside], z[inside])
    h[~np.isfinite(h)] = np.nan
    return HeightMap(h.reshape(H, W), grid)
