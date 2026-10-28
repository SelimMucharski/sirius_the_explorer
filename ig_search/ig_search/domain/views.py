"""
Kandydackie pozy obserwacji i prekomputowana macierz V[v, i] = P_d (DESIGN §3.1). Bez ROS.

Pozy: komórki przejezdne co `vp_step` × `n_yaw` kursów. Dla każdej pozy liczony jest
zbiór widocznych komórek z r i α; P_d wyznaczane z modelu detektora przy wczytaniu,
więc cache geometrii (plik .npz) nie zależy od kalibracji detektora.
"""
from dataclasses import asdict
import hashlib
import json
import math
import os
import time

from ig_search.domain.detector_model import DetectorModel
from ig_search.domain.params import CameraParams, ViewParams
from ig_search.domain.travel import TravelModel
from ig_search.domain.visibility import Visibility
import numpy as np
from scipy.sparse import csr_matrix

CACHE_VERSION = 'views-v2'      # zmiana formatu pliku .npz -> nowy klucz cache


class ViewSet:

    def __init__(self, xs, ys, yaws, cells, indptr, indices, r, alpha, n_cells,
                 build_time: float = 0.0, cam_x=None, cam_y=None):
        self.xs, self.ys, self.yaws = xs, ys, yaws
        self.cam_x, self.cam_y = cam_x, cam_y              # położenie kamery w pozie
        self.cells = cells                                  # komórka beliefu pozy
        self.indptr, self.indices = indptr, indices
        self.r, self.alpha = r, alpha                       # per niezerowy element
        self.n_cells = n_cells
        self.ok = np.ones(len(xs), bool)                    # wykluczone po nieudanej nawigacji
        self.build_time = build_time
        self.V = self.RV = None                             # po apply_model()

    def __len__(self):
        return len(self.xs)

    def apply_model(self, model: DetectorModel):
        """Wylicza V = P_d(r, α) i RV = r·P_d (do oczekiwanej odległości detekcji)."""
        pd = model.pd(self.r, self.alpha).astype(np.float32)
        shape = (len(self.xs), self.n_cells)
        self.V = csr_matrix((pd, self.indices, self.indptr), shape=shape)
        self.RV = csr_matrix((pd * self.r, self.indices, self.indptr), shape=shape)

    # ------------------------------------------------------------ budowa
    @staticmethod
    def build(vis: Visibility, travel: TravelModel, params: ViewParams,
              cache_dir: str = '') -> 'ViewSet':
        key = _cache_key(vis, travel, params)
        path = os.path.join(cache_dir, f'views_{key}.npz') if cache_dir else ''
        if path and os.path.exists(path):
            return ViewSet._load(path)
        t0 = time.monotonic()
        g = travel.grid
        s = max(1, int(round(params.vp_step / g.res)))
        sel = np.zeros(g.shape, bool)
        sel[s // 2::s, s // 2::s] = True
        cells = np.flatnonzero(sel.ravel() & travel.traversable
                               & (travel.clearance >= params.edge_clearance))
        yaws = np.arange(params.n_yaw) * 2.0 * math.pi / params.n_yaw
        xs, ys, ws, cs, kx, ky = [], [], [], [], [], []
        indptr, idx, rr, aa = [0], [], [], []
        for c in cells:
            for yaw in yaws:
                pose = vis.camera_pose(travel.cx[c], travel.cy[c], yaw)
                v = vis.visible(pose)
                if len(v) == 0:
                    continue
                kx.append(pose.position[0])
                ky.append(pose.position[1])
                xs.append(travel.cx[c])
                ys.append(travel.cy[c])
                ws.append(yaw)
                cs.append(c)
                order = np.argsort(v.ids)
                idx.append(v.ids[order])
                rr.append(v.r[order])
                aa.append(v.alpha[order])
                indptr.append(indptr[-1] + len(v))
        if not xs:
            raise RuntimeError('brak póz obserwacji - sprawdź mapę, vp_step i przejezdność')
        vs = ViewSet(np.array(xs), np.array(ys), np.array(ws), np.array(cs, np.int64),
                     np.array(indptr, np.int64), np.concatenate(idx).astype(np.int32),
                     np.concatenate(rr).astype(np.float32),
                     np.concatenate(aa).astype(np.float32), g.size, time.monotonic() - t0,
                     np.array(kx), np.array(ky))
        if path:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(path, xs=vs.xs, ys=vs.ys, yaws=vs.yaws, cells=vs.cells,
                                indptr=vs.indptr, indices=vs.indices, r=vs.r, alpha=vs.alpha,
                                n_cells=g.size, build_time=vs.build_time, cam_x=vs.cam_x,
                                cam_y=vs.cam_y)
        return vs

    @staticmethod
    def _load(path: str) -> 'ViewSet':
        d = np.load(path)
        return ViewSet(d['xs'], d['ys'], d['yaws'], d['cells'], d['indptr'], d['indices'],
                       d['r'], d['alpha'], int(d['n_cells']), float(d['build_time']),
                       d['cam_x'], d['cam_y'])


def _cache_key(vis: Visibility, travel: TravelModel, params: ViewParams) -> str:
    h = hashlib.sha1(CACHE_VERSION.encode())
    h.update(np.ascontiguousarray(vis.hm.h).tobytes())
    h.update(np.ascontiguousarray(travel.traversable).tobytes())     # z mapą przejezdności
    h.update(json.dumps([asdict(vis.cam) if isinstance(vis.cam, CameraParams) else str(vis.cam),
                         vis.hfov, vis.vfov, asdict(params), asdict(travel.p),
                         vars(travel.grid), vars(vis.hm.grid)], sort_keys=True,
                        default=str).encode())
    return h.hexdigest()[:16]
