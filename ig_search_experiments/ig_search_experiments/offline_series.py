r"""
Szybka seria epizodów bez Gazebo i bez ROS (strojenie strategii, porównania wstępne).

  ros2 run ig_search_experiments offline_series --out <katalog> --seeds 1 2 3 \
      --variant frontier="planner.strategy=frontier" \
      --variant chase="planner.strategy=chase planner.frontier_min_gain=10"

Świat jak w węźle ig_search (mapa wysokości, mapa przejezdności, parametry z
config/ig_search.yaml), próbki rozmieszczane tym samym kodem i z tym samym ziarnem co spawner
w sim_episode (probe_supervisor: prostokąt, mapa przejezdności z odstępem 0,5 m, mapa miejsc,
odstęp 1 m, bez koła 2 m wokół startu) - pozycje jak w Gazebo przed upadkiem próbki. Łazik
startuje w (0, 0, 0).

Uproszczenia (wyniki tylko do porównań względnych i strojenia, nie zamiast Gazebo):
  - detektor testowy z modelem planera (P_d(r, α), σ(r), fałszywe alarmy Poissona λ_fa);
    --pd-scale / --fa-scale rozstrajają go względem modelu planera;
  - jazda po punktach pośrednich misji (trasa z grafu przejezdności) ze stałą prędkością
    --v, obrót w miejscu z --w; bez Nav2, poślizgu i utknięć;
  - ramię jako atrapa: każdy ruch (skan, pozycja przed chwytem, chwyt, złożenie) trwa
    --arm-time s; kamera nadgarstka nad celem skanu, detektor testowy z modelem nadgarstka.
Wynik każdego epizodu - <out>/<etykieta>/seed_<n>/summary.json w formacie probe_watch, więc
analiza: `ros2 run ig_search_experiments analyze_series <out>`.
"""
import argparse
import json
import math
import multiprocessing
import os
import random
import time

from ament_index_python.packages import get_package_share_directory
from ig_search.application.mission import Mission
from ig_search.application.ports import (
    Frame, Manipulator, Navigator, Reporter, SamplePose, Status)
from ig_search.application.world import build_world
from ig_search.domain.params import from_dict, with_section
from ig_search.domain.strategies import make_strategy
from ig_search.domain.terrain import HeightMap
from ig_search.domain.traversability import TraversabilityMap
from ig_search.domain.visibility import CameraPose
import numpy as np
from probe_supervisor.domain.placement import (
    KeepOutCircle, PlacementConstraints, sample_positions)
from probe_supervisor.domain.spawn_area import MaskedArea, RectangleArea, TraversabilityMask
import yaml

# jak w sim_episode.launch.py
AREA = (1.0, 20.0, -10.0, 8.0)
START = (0.0, 0.0, 0.0)


def probe_positions(seed: int, trav_yaml: str, spawn_yaml: str, count: int = 3):
    area = MaskedArea(RectangleArea(*AREA), TraversabilityMask.load(trav_yaml), 0.5,
                      TraversabilityMask.load(spawn_yaml))
    constraints = PlacementConstraints(min_separation=1.0,
                                       keep_out=(KeepOutCircle(0.0, 0.0, 2.0),))
    return sample_positions(area, count, [], constraints, random.Random(seed))


class PathNav(Navigator):
    """Jazda po punktach pośrednich ze stałą prędkością, potem obrót do kursu celu."""

    def __init__(self, q, v, w):
        self.q = list(q)
        self.v, self.w = v, w
        self.st = Status.IDLE
        self.path, self.goal, self.reach = [], None, None

    def go_to(self, x, y, yaw, precise=False, via=(), reach=None):
        self.goal, self.reach = (x, y, yaw), reach
        self.path = [tuple(p[:2]) for p in via] + [(x, y)]
        self.st = Status.ACTIVE
        if reach is not None:
            if reach.stop_early and reach.contains(tuple(self.q)):
                self.st = Status.SUCCEEDED
            elif reach.direct:                          # na prostej łazik-ŝ, ŝ w środku R(q)
                rp = reach.params
                d = rp.workspace_root_x + 0.5 * (rp.workspace_r_min + reach.r_max)
                b = math.atan2(reach.y - self.q[1], reach.x - self.q[0])
                self.goal = (reach.x - d * math.cos(b), reach.y - d * math.sin(b), b)
                self.path = [self.goal[:2]]

    def cancel(self):
        self.goal, self.path = None, []
        self.st = Status.IDLE

    def back_up(self, distance):
        self.q[0] -= distance * math.cos(self.q[2])
        self.q[1] -= distance * math.sin(self.q[2])
        self.cancel()
        self.st = Status.SUCCEEDED

    def status(self):
        return self.st

    def tick(self, dt):
        if self.st != Status.ACTIVE:
            return
        if self.reach is not None and self.reach.stop_early and \
                self.reach.contains(tuple(self.q)):
            self.st = Status.SUCCEEDED
            return
        budget = self.v * dt
        while self.path and budget > 0:
            gx, gy = self.path[0]
            dx, dy = gx - self.q[0], gy - self.q[1]
            d = math.hypot(dx, dy)
            if d > 1e-6:
                self.q[2] = math.atan2(dy, dx)
            s = min(d, budget)
            if d > 1e-6:
                self.q[0] += dx / d * s
                self.q[1] += dy / d * s
            budget -= s
            if d - s <= 1e-6:
                self.path.pop(0)
            if self.reach is not None and self.reach.stop_early and \
                    self.reach.contains(tuple(self.q)):
                self.st = Status.SUCCEEDED
                return
        if self.path:
            return
        if self.reach is not None:                      # bez obrotu do kursu
            self.q[2] = self.goal[2] if self.reach.direct else self.q[2]
            self.st = Status.SUCCEEDED
            return
        e = (self.goal[2] - self.q[2] + math.pi) % (2 * math.pi) - math.pi
        if abs(e) <= self.w * dt:
            self.q[2] = self.goal[2]
            self.st = Status.SUCCEEDED
        else:
            self.q[2] += math.copysign(self.w * dt, e)


class FakeManip(Manipulator):
    """Każdy ruch ramienia kończy się po `duration` s."""

    def __init__(self, duration):
        self.st = Status.IDLE
        self.t_left = 0.0
        self.duration = duration
        self.last = None

    def _start(self, name, *args):
        self.last = (name,) + args
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

    def scan_camera(self, height):
        """Kamera nadgarstka podczas skanu (oś optyczna w dół nad celem) albo None."""
        if self.last is None or self.last[0] != 'scan' or self.st == Status.IDLE:
            return None
        x, y, z = self.last[1:]
        R = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        return CameraPose(np.array([x, y, z + height]), R)


class ListReporter(Reporter):

    def __init__(self):
        self.reports = []

    def report(self, r):
        self.reports.append(r)


class ToyDetector:
    """Detekcje próbek w V(q) z P_d modelu (× pd_scale), szum σ(r), fałszywe alarmy Poissona."""

    def __init__(self, vis, model, probes, rng, pd_scale=1.0, fa_scale=1.0):
        self.vis, self.model = vis, model
        self.probes = np.asarray(probes, float).reshape(-1, 2)
        self.cells = vis.grid.flat_index(self.probes[:, 0], self.probes[:, 1])
        self.rng = rng
        self.pd_scale, self.fa_scale = pd_scale, fa_scale
        self.gone = np.zeros(len(self.probes), bool)    # zgłoszone (zebrane) - niewidoczne

    def frame(self, stamp, pose):
        v = self.vis.visible(pose)
        dets = []
        if len(v):
            for k, c in enumerate(self.cells):
                if self.gone[k]:
                    continue
                hit = np.flatnonzero(v.ids == c)
                if len(hit):
                    i = hit[0]
                    pd = self.pd_scale * float(self.model.pd(v.r[i:i + 1], v.alpha[i:i + 1])[0])
                    if self.rng.random() < pd:
                        s = float(self.model.sigma(v.r[i:i + 1])[0])
                        dets.append(self.probes[k] + self.rng.normal(0, s, 2))
            area = len(v) * self.vis.grid.res ** 2
            for _ in range(self.rng.poisson(self.model.lambda_fa * area * self.fa_scale)):
                c = v.ids[self.rng.integers(len(v))]
                dets.append(np.array([self.vis.cx[c], self.vis.cy[c]]))
        d = np.array([[x, y, 0.0] for x, y in dets]).reshape(-1, 3)
        return Frame(stamp, pose, d, [SamplePose(p, np.array([1.0, 0.0, 0.0])) for p in d])


def load_params(config: str, overrides: dict):
    with open(os.path.join(config, 'ig_search.yaml')) as f:
        ros = yaml.safe_load(f)['ig_search']['ros__parameters']
    nested = {k: v for k, v in ros.items() if isinstance(v, dict)}
    for key, value in overrides.items():
        section, name = key.split('.', 1)
        nested.setdefault(section, {})[name] = value
    p = from_dict(nested)
    for sec in ('detector', 'wrist_detector'):
        cf = getattr(p, sec).calibration_file
        if cf and not os.path.isabs(cf):
            p = with_section(p, sec, calibration_file=os.path.join(config, cf))
    return p, ros


def parse_overrides(text: str) -> dict:
    out = {}
    for item in text.split():
        k, v = item.split('=', 1)
        out[k] = yaml.safe_load(v)
    return out


_W = {}                                                 # świat współdzielony przez fork


def run_episode(job):
    label, overrides, seed, a = job
    world, ok0 = _W['world'], _W['ok']
    p, _ = load_params(_W['config'], overrides)
    world.params = p
    world.views.ok[:] = ok0
    probes = _W['probes'][seed]
    nav = PathNav(START, a.v, a.w)
    rep = ListReporter()
    manip = FakeManip(a.arm_time)
    mission = Mission(world, make_strategy(p), nav, rep, manip=manip)
    rng = np.random.default_rng(seed)
    det = ToyDetector(world.vis, world.model, probes, rng, a.pd_scale, a.fa_scale)
    wrist = ToyDetector(world.wrist_vis, world.wrist_model, probes, rng, a.pd_scale,
                        a.fa_scale)
    t, n_rep, t0 = 0.0, 0, time.monotonic()
    names = [f'probe_object_{i + 1}' for i in range(len(probes))]
    reports = []
    while not mission.finished and t < p.mission.t_max + 1.0:
        nav.tick(a.dt)
        manip.tick(a.dt)
        q = tuple(nav.q)
        mission.on_front_frame(det.frame(t, world.vis.camera_pose(*q)))
        cam = manip.scan_camera(p.wrist.scan_height)
        if cam is not None:
            mission.on_wrist_frame(wrist.frame(t, cam))
        mission.step(t, q)
        while len(mission.reports) > n_rep:
            r = mission.reports[n_rep]
            n_rep += 1
            err = [math.hypot(r.x - px, r.y - py) for px, py in probes]
            k = int(np.argmin(err))
            if err[k] < 0.15:
                det.gone[k] = wrist.gone[k] = True
            reports.append({'index': r.index, 't': round(mission.elapsed, 2),
                            'probe': names[k], 'error_now': round(err[k], 4),
                            'grasped': False})
        t += a.dt
    if not mission.finished:
        mission.timeout()
    out = os.path.join(a.out, label, f'seed_{seed}')
    os.makedirs(out, exist_ok=True)
    summary = {'state': mission.state.name, 'strategy': p.planner.strategy, 'seed': seed,
               'variant': label, 'detail': mission.detail, 'elapsed': round(mission.elapsed, 2),
               'wall': round(time.monotonic() - t0, 1), 'reports': reports,
               'correct_0.15': len({r['probe'] for r in reports if r['error_now'] < 0.15}),
               'distance': round(mission._odo, 2), 'offline': True,
               'reference': {n: [x, y, 0.0] for n, (x, y) in zip(names, probes)}}
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    return label, seed, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--variant', action='append', required=True,
                    help='etykieta="sekcja.pole=wartość ..." (nadpisania parametrów)')
    ap.add_argument('--v', type=float, default=0.7, help='[m/s] prędkość jazdy')
    ap.add_argument('--w', type=float, default=0.8, help='[rad/s] obrót w miejscu')
    ap.add_argument('--dt', type=float, default=0.4, help='[s] krok (klatka i krok misji)')
    ap.add_argument('--arm-time', type=float, default=8.0, help='[s] jeden ruch ramienia')
    ap.add_argument('--pd-scale', type=float, default=1.0)
    ap.add_argument('--fa-scale', type=float, default=1.0)
    ap.add_argument('--jobs', type=int, default=max(os.cpu_count() // 2, 1))
    a = ap.parse_args(argv)
    config = os.path.join(get_package_share_directory('ig_search'), 'config')
    p, ros = load_params(config, {})
    hm = HeightMap.load(os.path.join(config, ros['heightmap']))
    trav = os.path.join(config, ros['traversability_map'])
    spawn = os.path.join(config, 'maps', 'erc2025_spawn.yaml')
    world = build_world(p, hm, os.path.expanduser(ros.get('cache_dir', '')),
                        TraversabilityMap.load(trav))
    _W.update(world=world, ok=world.views.ok.copy(), config=config,
              probes={s: probe_positions(s, trav, spawn) for s in a.seeds})
    jobs = []
    for seed in a.seeds:
        for v in a.variant:
            label, text = v.split('=', 1)
            if not os.path.exists(os.path.join(a.out, label, f'seed_{seed}', 'summary.json')):
                jobs.append((label, parse_overrides(text), seed, a))
    print(f'{len(jobs)} epizodów, {a.jobs} procesów', flush=True)
    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(a.jobs) as pool:
        for label, seed, s in pool.imap_unordered(run_episode, jobs):
            print(f'{label} seed {seed}: {s["state"]} {s["elapsed"]:.0f} s, '
                  f'poprawne {s["correct_0.15"]}/{len(s["reference"])}, '
                  f'zgłoszenia {len(s["reports"])}, droga {s["distance"]:.0f} m '
                  f'({s["wall"]:.0f} s ściennych)', flush=True)


if __name__ == '__main__':
    main()
