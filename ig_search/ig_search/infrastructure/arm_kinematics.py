"""
Kinematyka ramienia Sirius z URDF i numeryczne IK dla skanu i pozycji przed chwytem.

Bez ROS (tylko numpy/scipy) - robot-specyficzne, więc w infrastrukturze. Wynikiem są
cele w przestrzeni stawów; trajektorię bezkolizyjną planuje move_group.

Konwencje z URDF: ramka optyczna kamery - oś z wzdłuż osi optycznej; tcp_connector - oś z
to kierunek podejścia chwytaka, szczęki zamykają się wzdłuż osi y, palce 5 cm przed TCP.
"""
import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares

ARM_JOINTS = [f'joint_{i}' for i in range(1, 7)]
# Pozycja transportowa (domyślna) = `ik_ready` od autora: ramię złożone do przodu i w górę,
# w obrysie łazika, poza polem widzenia kamery przedniej. joint_3 = 1,85 przekracza limit
# z URDF (±1,58), więc tymczasowo przycięte do 1,58 (najwyżej 1,10 m zamiast 0,99 m).
# W SRDF jako stan `transport`.
STOW = [0.0, -0.5, 1.58, -1.53, 0.96, -0.06]
# [m] końcówki palców przed tcp_connector wzdłuż osi z: łącze szczęki 0,05 + środek bryły
# kolizyjnej palca 0,0552 + połowa jej długości 0,0415 (URDF)
FINGERTIP_OFFSET = 0.147
ARM_ROOT = np.array([0.1145, 0.0])  # [m] xy podstawy ramienia (root_link) w base_link
# Przed chwytem: palce dokładnie nad próbką (na pionie przez jej środek), oś podejścia pionowo
# w dół. Z IK osiągalne dla próbki 0,60-0,81 m od podstawy ramienia (ReportParams.workspace_*).
PREGRASP_POS_TOL = 0.005            # [m]
PREGRASP_AXIS_TOL = math.radians(2.0)


def rpy_to_R(r, p, y):
    cr, sr, cp, sp = math.cos(r), math.sin(r), math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_R(axis, q):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * K @ K


def _floats(el, attr, default):
    return [float(v) for v in (el.get(attr, default) if el is not None else default).split()]


class Chain:
    """Łańcuch kinematyczny base -> tip z URDF (stawy stałe i obrotowe)."""

    def __init__(self, urdf: str, base: str, tip: str):
        joints = {j.find('child').get('link'): j for j in ET.fromstring(urdf).findall('joint')}
        self.steps, link = [], tip
        while link != base:
            j = joints[link]
            o, ax, lim = j.find('origin'), j.find('axis'), j.find('limit')
            self.steps.append({
                'name': j.get('name'), 'type': j.get('type'),
                'xyz': np.array(_floats(o, 'xyz', '0 0 0')),
                'R': rpy_to_R(*_floats(o, 'rpy', '0 0 0')),
                'axis': _floats(ax, 'xyz', '0 0 1'),
                'lim': (float(lim.get('lower')), float(lim.get('upper')))
                if lim is not None and lim.get('lower') is not None else None})
            link = j.find('parent').get('link')
        self.steps.reverse()

    def fk(self, q: dict) -> np.ndarray:
        T = np.eye(4)
        for s in self.steps:
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = s['R'], s['xyz']
            T = T @ M
            if s['type'] == 'revolute' and s['name'] in q:
                R = np.eye(4)
                R[:3, :3] = axis_R(s['axis'], q[s['name']])
                T = T @ R
        return T

    def limits(self, names):
        lim = {s['name']: s['lim'] for s in self.steps}
        return np.array([lim[n] if lim.get(n) else (-math.pi, math.pi) for n in names])


class ArmIK:
    """IK numeryczne (najmniejsze kwadraty, wiele startów) w ramce base_link."""

    def __init__(self, urdf: str, base='base_link', camera='wrist_left_camera_optical',
                 tcp='tcp_connector', min_height=0.12, n_starts=12, seed=0):
        self.cam = Chain(urdf, base, camera)
        self.tcp = Chain(urdf, base, tcp)
        self.lim = self.cam.limits(ARM_JOINTS)
        self.min_height = min_height
        self.n_starts = n_starts
        self.rng = np.random.default_rng(seed)

    def _solve(self, residual, q0, accept, n_starts=None):
        lo, hi = self.lim[:, 0] + 1e-3, self.lim[:, 1] - 1e-3
        best = None
        for k in range(n_starts or self.n_starts):
            x0 = np.asarray(q0, float) if k == 0 else self.rng.uniform(lo * 0.8, hi * 0.8)
            r = least_squares(residual, np.clip(x0, lo, hi), bounds=(lo, hi))
            if accept(r.x) and (best is None or r.cost < best.cost):
                best = r
        return None if best is None else best.x

    def look_at(self, target, dist, q0=STOW, min_down=0.7, ground_z=None):
        """
        Stawy, przy których oś optyczna kamery nadgarstka przechodzi przez target.

        target w base_link, kamera w odległości dist, patrzy w dół (składowa z kierunku
        ≤ -min_down), co najmniej min_height nad gruntem (ground_z = wysokość gruntu).
        """
        target = np.asarray(target, float)
        gz = target[2] if ground_z is None else ground_z

        def res(x):
            T = self.cam.fk(dict(zip(ARM_JOINTS, x)))
            c, z = T[:3, 3], T[:3, 2]
            d = target - c
            n = np.linalg.norm(d)
            u = d / max(n, 1e-9)
            reach = np.linalg.norm(c[:2] - ARM_ROOT)   # mniej wyciągnięte ramię
            return np.r_[np.cross(z, u) * 2.0, (n - dist) * 3.0,
                         max(0.0, u[2] + min_down) * 2.0,
                         max(0.0, gz + self.min_height - c[2]) * 10.0,
                         max(0.0, reach - 0.5) * 0.3,
                         0.02 * (x - np.asarray(q0))]

        def ok(x):
            T = self.cam.fk(dict(zip(ARM_JOINTS, x)))
            d = target - T[:3, 3]
            return (T[:3, 2] @ d > 0.95 * np.linalg.norm(d)
                    and abs(np.linalg.norm(d) - dist) < 0.05)
        return self._solve(res, q0, ok)

    def pregrasp(self, position, axis, height, down=(0.0, 0.0, -1.0), q0=STOW):
        """
        Stawy dla chwytaka pionowo nad próbką; szczęki prostopadle do osi próbki.

        position, axis - środek i oś próbki w base_link; down - pion w dół w base_link (na
        zboczu base_link jest pochylony, a chwytak ma być pionowo w świecie); height -
        prześwit końcówek palców nad środkiem próbki. None, gdy nieosiągalne.
        """
        p = np.asarray(position, float)
        a = np.asarray(axis, float)
        a /= max(np.linalg.norm(a), 1e-9)
        d = np.asarray(down, float)
        return self._pregrasp_dir(p, a, d / np.linalg.norm(d), height, q0)

    def vertical_path(self, position, axis, heights, down=(0.0, 0.0, -1.0), q0=STOW):
        """
        Stawy dla kolejnych wysokości palców nad osią próbki - pionowy odcinek chwytu.

        Każdy punkt startuje z poprzedniego (ciągłość, ten sam kierunek szczęk). None, gdy
        któryś punkt jest nieosiągalny.
        """
        p = np.asarray(position, float)
        a = np.asarray(axis, float)
        a /= max(np.linalg.norm(a), 1e-9)
        d = np.asarray(down, float)
        d /= np.linalg.norm(d)
        path, q = [], np.asarray(q0, float)
        for h in heights:
            x = self._pregrasp_dir(p, a, d, h, q, n_starts=1)
            if x is None:
                x = self._pregrasp_dir(p, a, d, h, q)
            if x is None:
                return None
            path.append(x)
            q = x
        return path

    def _pregrasp_dir(self, p, a, d, height, q0, n_starts=None):
        y_base = np.cross(d, a)                     # szczęki ⟂ osi próbki i ⟂ podejściu
        if np.linalg.norm(y_base) < 1e-3:           # próbka wzdłuż kierunku podejścia
            y_base = np.cross(d, [1.0, 0.0, 0.0])
        y_base /= np.linalg.norm(y_base)
        tcp_goal = p - d * (height + FINGERTIP_OFFSET)
        best, best_cost = None, np.inf
        for sign in (1.0, -1.0):
            y_des = sign * y_base

            def res(x, y_des=y_des):
                T = self.tcp.fk(dict(zip(ARM_JOINTS, x)))
                return np.r_[(T[:3, 3] - tcp_goal) * 5.0, np.cross(T[:3, 2], d) * 2.0,
                             np.cross(T[:3, 1], y_des) * 1.0, 0.02 * (x - np.asarray(q0))]

            def ok(x, y_des=y_des):
                T = self.tcp.fk(dict(zip(ARM_JOINTS, x)))
                return (np.linalg.norm(T[:3, 3] - tcp_goal) < PREGRASP_POS_TOL
                        and T[:3, 2] @ d > math.cos(PREGRASP_AXIS_TOL)
                        and abs(T[:3, 1] @ y_des) > 0.95)
            x = self._solve(res, q0, ok, n_starts)
            if x is not None:
                cost = float(np.sum(res(x) ** 2))
                if cost < best_cost:
                    best, best_cost = x, cost
        return best
