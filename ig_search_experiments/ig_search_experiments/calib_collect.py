"""
M6: zbieranie danych do kalibracji modelu detektora ig_search w Gazebo (DESIGN §3.2).

Dla każdej klatki detektora (kamera przednia albo na nadgarstku) liczone są te same wielkości
co w planerze: V(q) z przesłanianiem przez teren (ig_search.domain.visibility) dla prawdziwej
pozy kamery (poza łazika z Gazebo + base_link -> kamera z TF), a dla każdej próbki w V(q):
r, α, wykrycie, błąd pozycji 2D. Detekcje nieprzypisane do żadnej próbki to fałszywe alarmy
(liczone osobno w V(q) i poza nim). Wyniki: <output>/<kamera>_probes.csv, <kamera>_frames.csv.

Faza `front`: łazik teleportowany w losowe pozy wokół próbek (odległość, kąt do próbki),
ramię w pozycji transportowej. Faza `wrist`: łazik obok próbki, ramię kieruje kamerę na próbkę
z kilku wysokości i przesunięć (MoveItManipulator.scan z ig_search); klatki oznaczone, czy
ramię stało.

  ros2 launch ig_search_experiments calib.launch.py phase:=front output:=<katalog>
"""
import csv
import math
import os
import threading
import time

from ament_index_python.packages import get_package_share_directory
from ig_search.application.ports import Status
from ig_search.domain.params import from_dict
from ig_search.domain.terrain import HeightMap
from ig_search.domain.visibility import CameraPose, Visibility
from ig_search.infrastructure.arm_kinematics import ARM_ROOT
from ig_search.infrastructure.moveit_manipulator import MoveItManipulator
from ig_search_experiments.gz_service import set_pose
import numpy as np
from probe_detector.infrastructure.gz_world_poses import GzWorldPoses
from probe_supervisor_interfaces.srv import SpawnRandomProbes
import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3DArray
import yaml

GATE = 0.75                 # [m] detekcja przypisana do próbki bliżej niż tyle (2D)
OPT_TO_VIS = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
# ramka optyczna (z w przód, x w prawo, y w dół) -> ramka kamery planera (x w przód, y w lewo)
PROBE_KEYS = ['pose', 't', 'probe', 'moving', 'r', 'alpha', 'detected', 'err_x', 'err_y',
              'err_z', 'err_range', 'err_cross', 'score']
FRAME_KEYS = ['pose', 't', 'moving', 'n_visible_cells', 'n_det', 'n_fa_visible', 'n_fa_other',
              'n_fa_near', 'fa_dist']
NEAR = 0.15                 # [m] nieprzypisana detekcja tak blisko próbki = podwójna, nie FA


def quat_to_R(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class CalibCollect(Node):

    def __init__(self):
        super().__init__('calib_collect')
        P = self.declare_parameter
        P('output', '/tmp/calib')
        P('phase', 'front')                 # front | wrist
        P('world', 'erc2025')
        P('n_poses', 60)
        P('seed', 1)
        P('settle', 2.0)                    # [s] po teleporcie
        P('dwell', 3.0)                     # [s] zbierania klatek w pozie
        P('min_range', 0.8)                 # [m] faza front: odległość łazik - próbka
        P('max_range', 6.5)
        P('max_slope_deg', 12.0)
        P('wrist_heights', [0.25, 0.35, 0.5, 0.65])
        P('wrist_offsets', [0.0, 0.1, -0.1])
        P('wrist_probes', 3)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.out, self.phase, self.world = g('output'), g('phase'), g('world')
        os.makedirs(self.out, exist_ok=True)
        self.rng = np.random.default_rng(g('seed'))
        share = get_package_share_directory('ig_search')
        with open(os.path.join(share, 'config', 'ig_search.yaml')) as f:
            cfg = yaml.safe_load(f)['ig_search']['ros__parameters']
        hm_path = os.path.join(share, 'config', cfg.pop('heightmap'))
        cfg.pop('cache_dir', None)
        self.params = from_dict(cfg)
        self.hm = HeightMap.load(hm_path)
        grid = self.hm.grid.coarsen(int(round(self.params.belief.cell_size / self.hm.grid.res)))
        self.grid = grid
        p = self.params
        cam, hfov, vfov = ((p.camera, p.hfov, p.vfov) if self.phase == 'front'
                           else (p.wrist_camera, p.wrist_hfov, p.wrist_vfov))
        self.vis = Visibility(self.hm, grid, cam, hfov, vfov)
        self.gt = GzWorldPoses(self.world, 'sirius', 'probe_object_')
        self.tf = Buffer(cache_time=Duration(seconds=30))
        self.tf_listener = TransformListener(self.tf, self, spin_thread=True)
        self.pose_id = -1
        self.stats = {'msgs': 0, 'tf_fail': 0, 'gt_fail': 0, 'frames': 0}
        self.collecting = False
        self.arm_moving = False
        self._lock = threading.Lock()
        self.files = {}
        for kind, keys in (('probes', PROBE_KEYS), ('frames', FRAME_KEYS)):
            f = open(os.path.join(self.out, f'{self.phase}_{kind}.csv'), 'w', newline='')
            w = csv.DictWriter(f, keys)
            w.writeheader()
            self.files[kind] = (f, w)
        self.create_subscription(Detection3DArray, f'/{self.phase}/probe_detector/detections',
                                 self.on_dets, 10)
        self.create_subscription(JointState, '/joint_states', self.on_joints, 10)
        self.spawn = self.create_client(SpawnRandomProbes, '/spawn_random_probes')
        self.manip = None
        if self.phase == 'wrist':
            self.manip = MoveItManipulator(self, self.gt.robot_latest)
            self.create_timer(0.05, self.manip.tick)

    # ------------------------------------------------------------ pomocnicze
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def sleep(self, sec):
        t_end = self.now() + sec
        while rclpy.ok() and self.now() < t_end:
            time.sleep(0.02)

    def on_joints(self, msg):
        if len(msg.velocity) >= len(msg.name):
            v = {n: abs(x) for n, x in zip(msg.name, msg.velocity)}
            self.arm_moving = any(v.get(f'joint_{i}', 0.0) > 0.01 for i in range(1, 7))

    def probes(self):
        with self.gt._lock:
            return {k: v[:3, 3].copy() for k, v in self.gt._probes.items()}

    def teleport(self, x, y, yaw):
        return set_pose(self.world, 'sirius', x, y, float(self.hm.height_at(x, y)) + 0.25, yaw)

    def slope_ok(self, x, y):
        gx, gy = self.vis.gradient_at(x, y)
        return math.degrees(math.atan(math.hypot(gx, gy))) <= self.get_parameter(
            'max_slope_deg').value

    # ------------------------------------------------------------ klatki
    def on_dets(self, msg):
        if not self.collecting:
            return
        self.stats['msgs'] += 1
        t = Time.from_msg(msg.header.stamp)
        try:
            tf = self.tf.lookup_transform('base_link', msg.header.frame_id, t,
                                          timeout=Duration(seconds=0.2))
        except TransformException:
            self.stats['tf_fail'] += 1
            return
        T_base = self.gt.robot_at(t.nanoseconds)
        if T_base is None:
            self.stats['gt_fail'] += 1
            return
        self.stats['frames'] += 1
        tr, ro = tf.transform.translation, tf.transform.rotation
        T_bc = np.eye(4)
        T_bc[:3, :3] = quat_to_R(ro.x, ro.y, ro.z, ro.w)
        T_bc[:3, 3] = [tr.x, tr.y, tr.z]
        T_cam = T_base @ T_bc
        pose = CameraPose(T_cam[:3, 3].copy(), T_cam[:3, :3] @ OPT_TO_VIS)
        vs = self.vis.visible(pose)
        dets = []
        for d in msg.detections:
            if not d.results:
                continue
            p = d.results[0].pose.pose.position
            pw = (T_cam @ np.array([p.x, p.y, p.z, 1.0]))[:3]
            dets.append((pw, d.results[0].hypothesis.score))
        probes = self.probes()
        names = sorted(probes)
        P = np.array([probes[n] for n in names]) if names else np.zeros((0, 3))
        # przypisanie zachłanne po odległości 2D
        pairs = sorted(((float(np.linalg.norm(dp[0][:2] - P[j, :2])), i, j)
                        for i, dp in enumerate(dets) for j in range(len(names))))
        det_of, used = {}, set()
        for dist, i, j in pairs:
            if dist > GATE:
                break
            if i in used or j in det_of:
                continue
            det_of[j] = i
            used.add(i)
        vis_cells = {int(c): k for k, c in enumerate(vs.ids)}
        moving = int(self.arm_moving)
        with self._lock:
            pw = self.files['probes'][1]
            for j, n in enumerate(names):
                c = int(self.grid.flat_index(P[j, 0], P[j, 1]))
                if c not in vis_cells:
                    continue
                k = vis_cells[c]
                row = {'pose': self.pose_id, 't': round(t.nanoseconds * 1e-9, 3), 'probe': n,
                       'moving': moving, 'r': round(float(vs.r[k]), 3),
                       'alpha': round(float(vs.alpha[k]), 4), 'detected': int(j in det_of)}
                if j in det_of:
                    e = dets[det_of[j]][0] - P[j]
                    ray = P[j, :2] - pose.position[:2]
                    ray /= max(np.linalg.norm(ray), 1e-9)
                    row.update(err_x=round(float(e[0]), 4), err_y=round(float(e[1]), 4),
                               err_z=round(float(e[2]), 4),
                               err_range=round(float(e[:2] @ ray), 4),
                               err_cross=round(float(e[0] * -ray[1] + e[1] * ray[0]), 4),
                               score=round(float(dets[det_of[j]][1]), 3))
                pw.writerow(row)
            fa_vis = fa_other = fa_near = 0
            dists = []
            for i, (p, _) in enumerate(dets):
                if i in used:
                    continue
                d = float(np.min(np.linalg.norm(P[:, :2] - p[:2], axis=1))) if len(P) else 99.0
                dists.append(f'{d:.3f}')
                if d < NEAR:                    # druga detekcja tej samej próbki
                    fa_near += 1
                    continue
                c = int(self.grid.flat_index(p[0], p[1]))
                if c in vis_cells:
                    fa_vis += 1
                else:
                    fa_other += 1
            self.files['frames'][1].writerow({
                'pose': self.pose_id, 't': round(t.nanoseconds * 1e-9, 3), 'moving': moving,
                'n_visible_cells': len(vs), 'n_det': len(dets), 'n_fa_visible': fa_vis,
                'n_fa_other': fa_other, 'n_fa_near': fa_near, 'fa_dist': ' '.join(dists)})

    # ------------------------------------------------------------ fazy
    def wait_ready(self):
        while rclpy.ok() and (self.gt.robot_latest() is None or not self.probes()):
            if self.gt.robot_latest() is not None and self.spawn.service_is_ready():
                fut = self.spawn.call_async(SpawnRandomProbes.Request(count=self.n_spawn()))
                while rclpy.ok() and not fut.done():
                    time.sleep(0.1)
                self.sleep(8.0)                 # próbki spadają i układają się
                continue
            self.get_logger().info('czekam na Gazebo i spawner', throttle_duration_sec=10.0)
            time.sleep(0.5)

    def n_spawn(self):
        return 10 if self.phase == 'front' else self.get_parameter('wrist_probes').value

    def collect(self, sec, moving_ok=True):
        self.collecting = True
        self.sleep(sec)
        self.collecting = False
        for f, _ in self.files.values():
            f.flush()

    def run_front(self):
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        n_done, tries = 0, 0
        while rclpy.ok() and n_done < g('n_poses') and tries < 20 * g('n_poses'):
            tries += 1
            probes = list(self.probes().values())
            target = probes[self.rng.integers(len(probes))]
            d = self.rng.uniform(g('min_range'), g('max_range'))
            phi = self.rng.uniform(-math.pi, math.pi)
            x, y = target[0] - d * math.cos(phi), target[1] - d * math.sin(phi)
            if not np.isfinite(self.hm.height_at(x, y)) or not self.slope_ok(x, y):
                continue
            if min(np.hypot(p[0] - x, p[1] - y) for p in probes) < 0.7:
                continue
            yaw = phi + self.rng.uniform(-0.8, 0.8)     # próbka w różnych miejscach kadru
            if not self.teleport(x, y, yaw):
                self.get_logger().warn(f'teleport do ({x:.1f}, {y:.1f}) nieudany',
                                       throttle_duration_sec=10.0)
                continue
            self.pose_id = n_done
            self.sleep(g('settle'))
            self.collect(g('dwell'))
            n_done += 1
            self.get_logger().info(f'poza {n_done}/{g("n_poses")}: d = {d:.1f} m, {self.stats}',
                                   throttle_duration_sec=20.0)

    def run_wrist(self):
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        while self.manip._ik is None or not self.manip.client.server_is_ready():
            time.sleep(0.5)
        k = 0
        for name, p in sorted(self.probes().items()):
            phi = self.rng.uniform(-math.pi, math.pi)
            reach = ARM_ROOT[0] + 0.68
            x, y = p[0] - reach * math.cos(phi), p[1] - reach * math.sin(phi)
            if not self.teleport(x, y, phi):
                continue
            self.sleep(g('settle'))
            self.manip.stow()
            self._wait(30.0)
            for h in g('wrist_heights'):
                for off in g('wrist_offsets'):
                    q = self.probes().get(name)
                    if q is None:
                        break
                    side = np.array([-math.sin(phi), math.cos(phi)])
                    tx, ty = q[0] + off * side[0], q[1] + off * side[1]
                    self.pose_id = k
                    self.manip.scan_height = h
                    self.collecting = True              # także w ruchu ramienia (moving = 1)
                    self.manip.scan(tx, ty, float(self.hm.height_at(tx, ty)))
                    st = self._wait(40.0)
                    self.collect(g('dwell'))
                    k += 1
                    self.get_logger().info(f'{name}: h = {h}, przesunięcie {off} - {st.name}')
            self.manip.stow()
            self._wait(30.0)

    def _wait(self, timeout):
        t_end = self.now() + timeout
        while rclpy.ok() and self.manip.status() == Status.ACTIVE and self.now() < t_end:
            time.sleep(0.05)
        return self.manip.status()

    def run(self):
        self.wait_ready()
        self.get_logger().info(f'start fazy {self.phase}: {len(self.probes())} próbek')
        if self.phase == 'front':
            self.run_front()
        else:
            self.run_wrist()
        for f, _ in self.files.values():
            f.close()
        self.get_logger().info(f'koniec zbierania: {self.stats}')


def main(args=None):
    rclpy.init(args=args)
    node = CalibCollect()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
