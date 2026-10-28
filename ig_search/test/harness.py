"""
Atrapy portów i prosty detektor do testów logiki misji (bez ROS, bez Gazebo).

Detektor testowy jest celowo prosty (model planera + szum) - sprawdza logikę automatu,
nie służy do oceny strategii (eksperymenty prowadzi się w Gazebo, PDR §6).
"""
import math

from ig_search.application.ports import (
    Frame, Manipulator, Navigator, Reporter, SamplePose, Status)
import numpy as np


class FakeNav(Navigator):
    """Jedzie po prostej z v_max, potem obraca się do kursu celu; z `reach` - do R(q)."""

    def __init__(self, q, v=0.6, w=0.8, fail=None):
        self.q = list(q)
        self.goal = None
        self.st = Status.IDLE
        self.v, self.w = v, w
        self.fail = fail or (lambda goal: False)
        self.goals = []
        self.backups = []
        self.reach = None

    def go_to(self, x, y, yaw, precise=False, via=(), reach=None):
        self.goal = (x, y, yaw)
        self.reach = reach
        self.goals.append(self.goal)
        self.st = Status.FAILED if self.fail(self.goal) else Status.ACTIVE
        if reach is not None and self.st == Status.ACTIVE:
            if reach.stop_early and reach.contains(tuple(self.q)):
                self.st = Status.SUCCEEDED
            elif reach.direct:                          # na prostej łazik-ŝ, ŝ w środku R(q)
                rp = reach.params
                d = rp.workspace_root_x + 0.5 * (rp.workspace_r_min + reach.r_max)
                b = math.atan2(reach.y - self.q[1], reach.x - self.q[0])
                self.goal = (reach.x - d * math.cos(b), reach.y - d * math.sin(b), b)

    def cancel(self):
        self.goal = None
        self.st = Status.IDLE

    def back_up(self, distance):
        self.reach = None
        self.backups.append(distance)
        self.q[0] -= distance * math.cos(self.q[2])
        self.q[1] -= distance * math.sin(self.q[2])
        self.goal = None
        self.st = Status.SUCCEEDED

    def status(self):
        return self.st

    def tick(self, dt):
        if self.st != Status.ACTIVE:
            return
        if (self.reach is not None and self.reach.stop_early
                and self.reach.contains(tuple(self.q))):
            self.st = Status.SUCCEEDED
            return
        gx, gy, gyaw = self.goal
        dx, dy = gx - self.q[0], gy - self.q[1]
        d = math.hypot(dx, dy)
        if self.reach is not None and self.reach.direct:
            # korekta z bieżącej pozy (Nav2Navigator): przodem do ŝ, jazda prosto także wstecz
            self.q[2] = gyaw
            s = min(d, self.v * dt)
            if d > 1e-6:
                self.q[0] += dx / d * s
                self.q[1] += dy / d * s
            if d - s <= 1e-6:
                self.st = Status.SUCCEEDED
            return
        if self.reach is not None and d <= 1e-6:        # bez obrotu do kursu
            self.st = Status.SUCCEEDED
            return
        if d > 1e-6:
            self.q[2] = math.atan2(dy, dx)
            s = min(d, self.v * dt)
            self.q[0] += dx / d * s
            self.q[1] += dy / d * s
            return
        e = (gyaw - self.q[2] + math.pi) % (2 * math.pi) - math.pi
        if abs(e) <= self.w * dt:
            self.q[2] = gyaw
            self.st = Status.SUCCEEDED
        else:
            self.q[2] += math.copysign(self.w * dt, e)


class FakeManip(Manipulator):

    def __init__(self, duration=3.0):
        self.st = Status.IDLE
        self.t_left = 0.0
        self.duration = duration
        self.calls = []

    def _start(self, name, *args):
        self.calls.append((name,) + args)
        self.st = Status.ACTIVE
        self.t_left = self.duration

    def scan(self, x, y, dz):
        self._start('scan', x, y, dz)

    def pregrasp(self, sample):
        self._start('pregrasp', sample)

    def grasp(self, sample):
        self._start('grasp', sample)

    def stow(self):
        self._start('stow')

    def status(self):
        return self.st

    def tick(self, dt):
        if self.st == Status.ACTIVE:
            self.t_left -= dt
            if self.t_left <= 0:
                self.st = Status.SUCCEEDED


class ListReporter(Reporter):

    def __init__(self):
        self.reports = []

    def report(self, r):
        self.reports.append(r)


class ToyDetector:
    """Detekcje próbek w V(q) z P_d modelu, szum σ(r) i fałszywe alarmy Poissona."""

    def __init__(self, vis, model, probes, rng, fa_scale=1.0):
        self.vis, self.model = vis, model
        self.probes = np.asarray(probes, float).reshape(-1, 2)
        self.rng = rng
        self.fa_scale = fa_scale
        self.found = np.zeros(len(self.probes), bool)

    def frame(self, stamp, pose):
        v = self.vis.visible(pose)
        dets = []
        if len(v):
            cells = self.vis.grid.flat_index(self.probes[:, 0], self.probes[:, 1])
            for k, c in enumerate(cells):
                if self.found[k]:
                    continue
                hit = np.flatnonzero(v.ids == c)
                if len(hit):
                    i = hit[0]
                    pd = float(self.model.pd(v.r[i:i + 1], v.alpha[i:i + 1])[0])
                    if self.rng.random() < pd:
                        s = float(self.model.sigma(v.r[i:i + 1])[0])
                        dets.append(self.probes[k] + self.rng.normal(0, s, 2))
            area = len(v) * self.vis.grid.res ** 2
            for _ in range(self.rng.poisson(self.model.lambda_fa * area * self.fa_scale)):
                i = self.rng.integers(len(v))
                c = v.ids[i]
                dets.append(np.array([self.vis.cx[c], self.vis.cy[c]]))
        d = np.array([[x, y, 0.0] for x, y in dets]).reshape(-1, 3)
        poses = [SamplePose(p, np.array([1.0, 0.0, 0.0])) for p in d]
        return Frame(stamp, pose, d, poses)


def run_mission(mission, nav, detector, manip=None, wrist_det=None, dt=0.2, t_max=1800.0,
                on_report=None):
    """Pętla symulacji: ruch, klatki, krok misji. Zwraca czas zakończenia."""
    t = 0.0
    n_rep = 0
    while not mission.finished and t < t_max:
        nav.tick(dt)
        if manip is not None:
            manip.tick(dt)
        q = tuple(nav.q)
        mission.on_front_frame(detector.frame(t, mission.w.vis.camera_pose(*q)))
        if wrist_det is not None and manip is not None and manip.calls and \
                manip.calls[-1][0] == 'scan' and manip.st != Status.IDLE:
            x, y, z = manip.calls[-1][1:]
            pose = wrist_pose_above(x, y, z)
            mission.on_wrist_frame(wrist_det.frame(t, pose))
        mission.step(t, q)
        if on_report is not None and len(mission.reports) > n_rep:
            n_rep = len(mission.reports)
            on_report(mission.reports[-1])
        t += dt
    return t


def wrist_pose_above(x, y, z, height=0.6):
    """Kamera nadgarstka nad punktem (wysokość jak WristParams.scan_height), oś w dół."""
    from ig_search.domain.visibility import CameraPose
    R = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])  # x_cam = -z świata
    return CameraPose(np.array([x, y, z + height]), R)
