"""
Obserwator epizodu: prawdziwe pozycje próbek z Gazebo, błędy zgłoszeń i potrącenia próbek.

Pozycje odniesienia - próbki w chwili startu misji (pierwszy /ig_search/status). Przy każdym
zgłoszeniu: odległość ŝ od najbliższej próbki teraz i w pozycji odniesienia. Próbka
„potrącona” - przesunięta o więcej niż `moved_threshold` od odniesienia. Na końcu misji
(DONE / TIMEOUT) zapis <output>/summary.json, probes.csv (t, nazwa, x, y, z), robot.csv
(t, x, y, z, yaw, roll, pitch - prawdziwa poza łazika, do analizy dojazdów i wywrotek)
i status.csv (t, stan, zgłoszenia, masa i entropia beliefu - przebieg w czasie, M9).

  ros2 run ig_search_experiments probe_watch --ros-args -p output:=<katalog>
"""
import csv
import json
import math
import os
import time

from ig_search_interfaces.msg import MissionStatus, SampleReport
import numpy as np
from probe_detector.infrastructure.gz_world_poses import GzWorldPoses
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

DONE, TIMEOUT = MissionStatus.DONE, MissionStatus.TIMEOUT


class ProbeWatch(Node):

    def __init__(self):
        super().__init__('probe_watch')
        self.declare_parameter('output', '/tmp/probe_watch')
        self.declare_parameter('moved_threshold', 0.02)       # [m]
        self.declare_parameter('seed', -1)                    # do summary.json (analiza serii)
        self.declare_parameter('variant', '')                 # np. horizon_depth:=5
        self.out = self.get_parameter('output').value
        self.moved_threshold = self.get_parameter('moved_threshold').value
        os.makedirs(self.out, exist_ok=True)
        self.gt = GzWorldPoses('erc2025', 'sirius', 'probe_object_', rate=10)
        self.ref = None                     # {nazwa: xyz} przy starcie misji
        self.last = {}                      # {nazwa: xyz} ostatnia znana poza (także zebranej)
        self.rows = []
        self.moved = {}                     # nazwa -> (czas misji, przesunięcie) pierwszego ruchu
        self.max_shift = {}
        self.removed = {}                   # nazwa -> czas misji zniknięcia (zebrana)
        self.reports = []
        self.robot_rows = []                # (czas misji, x, y, z, yaw, roll, pitch) łazika
        self.status_rows = []               # (czas misji, stan, zgłoszenia, masa, entropia)
        self.strategy = ''
        self.distance = 0.0                 # [m] droga łazika od startu misji
        self.elapsed = 0.0
        self.wall0 = None
        self.done = False
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(MissionStatus, '/ig_search/status', self.on_status, 10)
        self.create_subscription(SampleReport, '/ig_search/report', self.on_report, reliable)
        self.create_timer(0.5, self.sample)

    def _probes(self):
        with self.gt._lock:
            return {k: v[:3, 3].copy() for k, v in self.gt._probes.items()}

    def on_status(self, msg):
        self.elapsed = msg.elapsed
        self.strategy = msg.strategy
        if not self.status_rows or msg.elapsed - self.status_rows[-1][0] >= 1.0:
            self.status_rows.append([round(msg.elapsed, 2), int(msg.state), int(msg.reports),
                                     round(msg.belief_mass, 4), round(msg.belief_entropy, 2)])
        if self.ref is None:
            probes = self._probes()
            if probes:
                self.ref = probes
                self.wall0 = time.monotonic()
                self.get_logger().info(f'odniesienie: {len(probes)} próbek')
        if msg.state in (DONE, TIMEOUT) and not self.done:
            self.done = True
            self.finish(msg)

    def sample(self):
        if self.ref is None:
            return
        T = self.gt.robot_latest()
        if T is not None:
            x, y, z = T[:3, 3]
            if self.robot_rows:
                self.distance += math.hypot(x - self.robot_rows[-1][1], y - self.robot_rows[-1][2])
            self.robot_rows.append([round(self.elapsed, 2), round(float(x), 4), round(float(y), 4),
                                    round(float(z), 4),
                                    round(math.atan2(T[1, 0], T[0, 0]), 4),
                                    round(math.atan2(T[2, 1], T[2, 2]), 4),
                                    round(-math.asin(max(-1.0, min(1.0, T[2, 0]))), 4)])
        now = self._probes()
        for k in self.ref:
            if k not in now and k not in self.removed:
                self.removed[k] = round(self.elapsed, 1)
                self.get_logger().info(f'{k} usunięta (zebrana), t = {self.elapsed:.1f} s')
        for k in [k for k in self.removed if k in now]:  # upuszczona z chwytaka - wróciła
            del self.removed[k]
            self.get_logger().info(f'{k} z powrotem w świecie, t = {self.elapsed:.1f} s')
        self.last.update(now)
        for k, p in now.items():
            self.rows.append([round(self.elapsed, 2), k, *np.round(p, 4)])
            if k in self.ref:
                d = float(np.linalg.norm(p[:2] - self.ref[k][:2]))
                self.max_shift[k] = max(self.max_shift.get(k, 0.0), d)
                if d > self.moved_threshold and k not in self.moved:
                    self.moved[k] = (round(self.elapsed, 1), round(d, 3))
                    self.get_logger().warn(f'{k} przesunięta o {d:.3f} m '
                                           f'(t = {self.elapsed:.1f} s)')

    def on_report(self, msg):
        e = np.array([msg.estimate.x, msg.estimate.y])
        # z ostatnią pozą próbek już zebranych: próbka znika z Gazebo przy chwycie (podmiana
        # na przyspawaną kopię), zanim przyjdzie zgłoszenie
        now = {**self.last, **self._probes()}

        def nearest(probes):
            if not probes:
                return None, math.inf
            k = min(probes, key=lambda n: np.linalg.norm(probes[n][:2] - e))
            return k, float(np.linalg.norm(probes[k][:2] - e))
        k_now, d_now = nearest(now)
        _, d_ref = nearest(self.ref or {})
        r = {'index': msg.index, 't': round(self.elapsed, 1), 'probe': k_now,
             'error_now': round(d_now, 3), 'error_ref': round(d_ref, 3),
             'pregrasp_reached': msg.pregrasp_reached, 'grasped': msg.grasped,
             'has_sample_pose': msg.has_sample_pose}
        self.reports.append(r)
        self.get_logger().info(f'zgłoszenie {r}')

    def finish(self, status):
        wall = time.monotonic() - self.wall0 if self.wall0 else None
        summary = {
            'state': 'DONE' if status.state == DONE else 'TIMEOUT',
            'strategy': self.strategy,
            'seed': self.get_parameter('seed').value,
            'variant': self.get_parameter('variant').value,
            'detail': status.detail,
            'elapsed': round(status.elapsed, 1),
            'wall': None if wall is None else round(wall, 1),
            'rtf': None if not wall else round(status.elapsed / wall, 3),
            'reports': self.reports,
            'correct_0.15': len({r['probe'] for r in self.reports if r['error_now'] < 0.15}),
            'distance': round(self.distance, 1),
            'moved': self.moved,
            'removed': self.removed,
            'max_shift': {k: round(v, 3) for k, v in self.max_shift.items()},
            'reference': {k: np.round(v, 3).tolist() for k, v in (self.ref or {}).items()},
        }
        with open(os.path.join(self.out, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=1)
        with open(os.path.join(self.out, 'probes.csv'), 'w', newline='') as f:
            csv.writer(f).writerows([['t', 'probe', 'x', 'y', 'z']] + self.rows)
        with open(os.path.join(self.out, 'status.csv'), 'w', newline='') as f:
            csv.writer(f).writerows([['t', 'state', 'reports', 'belief_mass',
                                      'belief_entropy']] + self.status_rows)
        with open(os.path.join(self.out, 'robot.csv'), 'w', newline='') as f:
            csv.writer(f).writerows([['t', 'x', 'y', 'z', 'yaw', 'roll', 'pitch']]
                                    + self.robot_rows)
        self.get_logger().info(f'koniec epizodu: {json.dumps(summary)}')


def main(args=None):
    rclpy.init(args=args)
    node = ProbeWatch()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
