"""
Model detektora używany przez planer i belief (DESIGN §3.2). Bez ROS.

  P_d(r, α) - prawdopodobieństwo wykrycia próbki w widocznej komórce,
  σ(r)      - odchylenie std. błędu pozycji detekcji,
  λ_fa      - intensywność fałszywych alarmów [1/(m²·klatka)].

Dwa źródła: model parametryczny (przed kalibracją) albo tablica z kalibracji w Gazebo
(M6, plik YAML {r_edges, alpha_edges, pd[r][alpha], sigma_r_edges, sigma, lambda_fa}).
Tablica bez dopasowywania funkcji analitycznej - wartości stałe w przedziałach.
"""
from ig_search.domain.params import CameraParams, DetectorParams
import numpy as np
import yaml


class DetectorModel:

    def __init__(self, det: DetectorParams, cam: CameraParams, hfov: float):
        self.r_max = cam.r_max
        self.r_min = cam.r_min
        self.half_fov = hfov / 2.0
        self.lambda_fa = det.lambda_fa
        self._p = det
        self._table = None
        if det.calibration_file:
            self.load_table(det.calibration_file)

    # ------------------------------------------------------------ tablica
    def load_table(self, path: str):
        with open(path) as f:
            t = yaml.safe_load(f)
        self._table = {
            'r_edges': np.asarray(t['r_edges'], float),
            'alpha_edges': np.asarray(t.get('alpha_edges', [0.0, np.pi]), float),
            'pd': np.atleast_2d(np.asarray(t['pd'], float)),
            'sigma_r_edges': np.asarray(t.get('sigma_r_edges', t['r_edges']), float),
            'sigma': np.asarray(t['sigma'], float),
        }
        if 'lambda_fa' in t:
            self.lambda_fa = float(t['lambda_fa'])

    @staticmethod
    def _bin(edges, x):
        return np.clip(np.searchsorted(edges, x, side='right') - 1, 0, len(edges) - 2)

    # ------------------------------------------------------------ model
    def pd(self, r, alpha):
        """P_d dla tablic r, α (ta sama długość); 0 poza [r_min, r_max]."""
        r = np.asarray(r, float)
        alpha = np.asarray(alpha, float)
        if self._table is not None:
            t = self._table
            out = t['pd'][self._bin(t['r_edges'], r), self._bin(t['alpha_edges'], alpha)]
            out = np.where(r > t['r_edges'][-1], 0.0, out)
        else:
            p = self._p
            span = max(self.r_max - p.r_full, 1e-6)
            radial = np.where(r <= p.r_full, 1.0, np.clip((self.r_max - r) / span, 0.0, 1.0))
            # łagodny spadek w zewnętrznym 1/3 pola widzenia
            edge = np.clip((alpha / max(self.half_fov, 1e-6) - 2.0 / 3.0) * 3.0, 0.0, 1.0)
            out = p.pd_max * radial * (1.0 - p.edge_falloff * edge)
        return np.where((r < self.r_min) | (r > self.r_max), 0.0, out)

    def sigma(self, r):
        r = np.asarray(r, float)
        if self._table is not None:
            t = self._table
            return t['sigma'][self._bin(t['sigma_r_edges'], r)]
        return self._p.sigma0 + self._p.sigma1 * r
