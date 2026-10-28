"""
Węzeł ROS 2 (lifecycle): przeszukiwanie terenu i zgłaszanie próbek.

Wejścia:
  /front/probe_detector/detections  vision_msgs/Detection3DArray (ramka optyczna, każda klatka)
  /wrist/probe_detector/detections  vision_msgs/Detection3DArray (ramka optyczna)
  TF map -> base_link, map -> ramka klatki (na jej czas; klatki czekają w kolejce na TF)
  akcje: navigate_through_poses (Nav2), move_action (move_group); /robot_description, /joint_states
Wyjścia:
  ~/report   ig_search_interfaces/SampleReport   zgłoszenia próbek
  ~/status   ig_search_interfaces/MissionStatus  co status_period
  ~/belief   nav_msgs/OccupancyGrid              p (skala względna) do RViz
  ~/markers  visualization_msgs/MarkerArray      kandydaci, zgłoszenia, cel bieżącej akcji
                                                  i zaplanowane pozy obserwacji (ig_horizon)
  /cmd_vel   geometry_msgs/TwistStamped           tylko przy dokładnym dojeździe
  ~/snapshots, ~/snapshot_streams/*               drzewo zachowań misji na bieżąco (py_trees_ros):
             py-trees-tree-viewer (przeglądarka) albo py-trees-tree-watcher (terminal)

configure: mapa wysokości, modele, pozy obserwacji (cache), adaptery; activate: start misji.
"""
from collections import deque
import os
import time

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
from ig_search.application.mission import Mission
from ig_search.application.ports import Frame, Report, Reporter, SamplePose
from ig_search.application.world import build_world
from ig_search.domain.candidates import find_candidates
from ig_search.domain.params import flatten, from_flat, SearchParams, with_section
from ig_search.domain.strategies import make_strategy
from ig_search.domain.strategies.base import OBSERVE
from ig_search.domain.terrain import HeightMap
from ig_search.domain.traversability import TraversabilityMap
from ig_search.infrastructure.moveit_manipulator import MoveItManipulator
from ig_search.infrastructure.nav2_navigator import Nav2Navigator
from ig_search.infrastructure.ros_conversions import (
    detection_to_sample, optical_camera_pose, T_to_q, transform_to_T, yaw_to_quat)
from ig_search_interfaces.msg import MissionStatus, SampleReport
from nav_msgs.msg import OccupancyGrid
import numpy as np
import py_trees_ros
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.lifecycle import Node as LifecycleNode
from rclpy.lifecycle import State as LcState
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

NODE_PARAMS = {
    'heightmap': 'maps/erc2025_heightmap.yaml',   # względem share/ig_search/config
    # mapa przejezdności (format nav2_map_server); '' - tylko kryteria z mapy wysokości
    'traversability_map': 'maps/erc2025_traversability.yaml',
    'cache_dir': '~/.cache/ig_search',
    # skalowanie prędkości i przyspieszeń ramienia w move_group (1.0 - limity z joint_limits)
    'arm_velocity_scaling': 0.5,
    'arm_carry_scaling': 0.2,    # z próbką w chwytaku
    'map_frame': 'map',
    'base_frame': 'base_link',
    'odom_frame': 'odom',        # ramka lokalna bez korekt GNSS (pomiar do chwytu)
    'front_topic': '/front/probe_detector/detections',
    'wrist_topic': '/wrist/probe_detector/detections',
    'front_rate': 5.0,           # [Hz czasu symulacji] tempo detektora na łaziku (CPU)
    'wrist_rate': 5.0,
    'step_rate': 5.0,            # [Hz] krok automatu misji
    'status_period': 1.0,        # [s]
    'belief_period': 2.0,        # [s]
    'tf_timeout': 0.2,           # [s]
    'tf_wait': 2.0,              # [s] klatka czeka tyle na TF na swój czas, potem odrzucona
    'probe_object': True,        # próbka jako przeszkoda w scenie MoveIt przed chwytem
    'bt_snapshots': True,        # migawki drzewa zachowań (py_trees_ros) do podglądu na bieżąco
}


class RosReporter(Reporter):

    def __init__(self, node, publisher, frame):
        self.node, self.pub, self.frame = node, publisher, frame

    def report(self, r: Report):
        m = SampleReport()
        m.header.frame_id = self.frame
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.index = r.index
        m.estimate.x, m.estimate.y, m.estimate.z = float(r.x), float(r.y), float(r.z)
        m.probability, m.threshold = float(r.probability), float(r.threshold)
        x, y, psi = r.robot
        m.robot_pose.position.x, m.robot_pose.position.y = float(x), float(y)
        o = m.robot_pose.orientation
        o.x, o.y, o.z, o.w = yaw_to_quat(psi)
        if r.sample_pose is not None:
            m.has_sample_pose = True
            p = r.sample_pose.position
            m.sample_pose.position.x, m.sample_pose.position.y, m.sample_pose.position.z = (
                float(v) for v in p)
            a = r.sample_pose.axis / np.linalg.norm(r.sample_pose.axis)
            # kwaternion obracający z na oś próbki
            z = np.array([0.0, 0.0, 1.0])
            v, c = np.cross(z, a), float(z @ a)
            q = np.r_[v, 1.0 + c] if c > -0.999999 else np.array([1.0, 0.0, 0.0, 0.0])
            q /= np.linalg.norm(q)
            o = m.sample_pose.orientation
            o.x, o.y, o.z, o.w = (float(v) for v in q)
        m.pregrasp_reached = bool(r.pregrasp_reached)
        m.grasped = bool(r.grasped)
        self.pub.publish(m)


class SearchNode(LifecycleNode):

    def __init__(self, **kwargs):
        super().__init__('ig_search', **kwargs)
        self._search_keys = list(flatten(SearchParams()))
        for k, v in flatten(SearchParams()).items():
            self.declare_parameter(k, v)
        for k, v in NODE_PARAMS.items():
            self.declare_parameter(k, v)
        self.mission = None
        self._bt = None                 # py_trees_ros.trees.BehaviourTree wokół drzewa misji
        self._bt_used = False           # parametry i usługi py_trees_ros są już na węźle
        self._subs, self._timers, self._pubs = [], [], {}
        self._last = {'front': -1e9, 'wrist': -1e9}
        self._started = False
        self._queue = deque()
        self._stats = {c: {'frames': 0, 'with_dets': 0, 'dets': 0, 'dropped': 0}
                       for c in ('front', 'wrist')}

    def P(self, name):
        return self.get_parameter(name).value

    # ------------------------------------------------------------ lifecycle
    def on_configure(self, state: LcState) -> TransitionCallbackReturn:
        try:
            params = from_flat({k: self.P(k) for k in self._search_keys})
            config = os.path.join(get_package_share_directory('ig_search'), 'config')
            for sec in ('detector', 'wrist_detector'):   # tablice z kalibracji (M6)
                cf = getattr(params, sec).calibration_file
                if cf and not os.path.isabs(cf):
                    params = with_section(params, sec, calibration_file=os.path.join(config, cf))
            path, trav = (p if not p or os.path.isabs(p) else os.path.join(config, p)
                          for p in (self.P('heightmap'), self.P('traversability_map')))
            t0 = time.monotonic()
            world = build_world(params, HeightMap.load(path),
                                os.path.expanduser(self.P('cache_dir')),
                                TraversabilityMap.load(trav) if trav else None)
        except Exception as e:  # noqa: BLE001 - każdy błąd konfiguracji kończy przejście
            self.get_logger().error(f'konfiguracja nieudana: {e}')
            return TransitionCallbackReturn.FAILURE
        self.get_logger().info(
            f'świat: {world.grid.width}x{world.grid.height} komórek po {world.grid.res} m, '
            f'{len(world.views)} póz obserwacji (prekomputacja {world.views.build_time:.1f} s), '
            f'przejezdne {world.travel.traversable.mean():.0%}, {time.monotonic() - t0:.1f} s')

        self.tf = Buffer(cache_time=Duration(seconds=30))
        self.tf_listener = TransformListener(self.tf, self, spin_thread=True)
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pubs = {
            'report': self.create_lifecycle_publisher(SampleReport, '~/report', reliable),
            'status': self.create_lifecycle_publisher(MissionStatus, '~/status', 10),
            'belief': self.create_lifecycle_publisher(OccupancyGrid, '~/belief', latched),
            'markers': self.create_lifecycle_publisher(MarkerArray, '~/markers', 10)}
        nav = Nav2Navigator(self, self.robot_q, self.P('map_frame'),
                            xy_tol=0.08, yaw_tol=0.06)
        manip = None
        if params.mission.use_wrist:
            manip = MoveItManipulator(self, self.base_T, params.wrist.scan_height,
                                      params.wrist.pregrasp_height,
                                      params.camera.target_height,
                                      probe_object=self.P('probe_object'),
                                      grasp_depth=params.wrist.grasp_depth,
                                      map_frame=self.P('map_frame'),
                                      velocity_scaling=self.P('arm_velocity_scaling'),
                                      carry_scaling=self.P('arm_carry_scaling'),
                                      local_base_fn=self.base_T_local)
        reporter = RosReporter(self, self._pubs['report'], self.P('map_frame'))
        self.mission = Mission(world, make_strategy(params), nav, reporter, manip,
                               log=self.get_logger().info)
        self._setup_snapshots()
        return TransitionCallbackReturn.SUCCESS

    def _setup_snapshots(self):
        """
        Drzewo misji tykane przez py_trees_ros: po każdym tyku migawka do strumieni podglądu.

        py_trees_ros deklaruje na węźle parametry i usługi ~/snapshot_streams/*, więc tylko przy
        pierwszej konfiguracji; po ponownej drzewo jest tykane bez migawek.
        """
        if not self.P('bt_snapshots'):
            return
        if self._bt_used:
            self.get_logger().warn('migawki drzewa tylko przy pierwszej konfiguracji węzła')
            return
        self._bt_used = True
        self._bt = py_trees_ros.trees.BehaviourTree(self.mission.tree)
        self._bt.setup(node=self, timeout=5.0)
        self.mission.tick = self._bt.tick
        self.get_logger().info('drzewo zachowań: py-trees-tree-viewer albo '
                               'py-trees-tree-watcher (węzeł /ig_search)')

    def on_activate(self, state: LcState) -> TransitionCallbackReturn:
        self._subs = [
            self.create_subscription(Detection3DArray, self.P('front_topic'),
                                     lambda m: self._on_dets('front', m), 10),
            self.create_subscription(Detection3DArray, self.P('wrist_topic'),
                                     lambda m: self._on_dets('wrist', m), 10)]
        self._timers = [
            self.create_timer(1.0 / self.P('step_rate'), self._step),
            self.create_timer(self.P('status_period'), self._publish_status),
            self.create_timer(self.P('belief_period'), self._publish_belief),
            self.create_timer(0.05, self._process_queue),
            self.create_timer(0.05, self._tick_adapters),
            self.create_timer(30.0, self._log_stats)]
        return super().on_activate(state)

    def on_deactivate(self, state: LcState) -> TransitionCallbackReturn:
        for s in self._subs:
            self.destroy_subscription(s)
        for t in self._timers:
            self.destroy_timer(t)
        self._subs, self._timers = [], []
        if self.mission is not None:
            self.mission.nav.cancel()
        return super().on_deactivate(state)

    def on_cleanup(self, state: LcState) -> TransitionCallbackReturn:
        if self._bt is not None:
            self._bt.shutdown(destroy_node=False)
            self._bt = None
        for pub in self._pubs.values():
            self.destroy_lifecycle_publisher(pub)
        self._pubs, self.mission = {}, None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LcState) -> TransitionCallbackReturn:
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------ TF
    def _lookup(self, frame, stamp=None):
        try:
            tf = self.tf.lookup_transform(
                self.P('map_frame'), frame, stamp or Time(),
                Duration(seconds=self.P('tf_timeout')) if stamp else Duration())
        except TransformException as e:
            self.get_logger().warn(f'TF {frame}: {e}', throttle_duration_sec=5.0)
            return None
        return transform_to_T(tf)

    def base_T(self):
        return self._lookup(self.P('base_frame'))

    def base_T_local(self):
        """base_link w ramce odometrii (bez korekt GNSS)."""
        try:
            return transform_to_T(self.tf.lookup_transform(
                self.P('odom_frame'), self.P('base_frame'), Time()))
        except TransformException:
            return None

    def robot_q(self):
        T = self.base_T()
        return None if T is None else T_to_q(T)

    # ------------------------------------------------------------ wejścia
    def _on_dets(self, cam, msg: Detection3DArray):
        """
        Klatka detektora do kolejki; przetwarzana, gdy TF na jej czas będzie dostępny.

        Przy obciążonym CPU EKF / korekcja GNSS publikują TF z opóźnieniem - lookup z krótkim
        limitem czasu gubiłby większość klatek.
        """
        if self.mission is None or self.mission.finished or not self._started:
            return
        t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        if t - self._last[cam] < 1.0 / self.P(f'{cam}_rate'):
            return                                   # tempo detektora na łaziku
        self._last[cam] = t
        self._queue.append((cam, msg))

    def _process_queue(self):
        map_frame = self.P('map_frame')
        now = self.get_clock().now().nanoseconds * 1e-9
        keep = deque()
        while self._queue:
            cam, msg = self._queue.popleft()
            stamp = Time.from_msg(msg.header.stamp)
            if self.tf.can_transform(map_frame, msg.header.frame_id, stamp):
                self._apply(cam, msg, stamp)
            elif now - stamp.nanoseconds * 1e-9 > self.P('tf_wait'):
                self._stats[cam]['dropped'] += 1
            else:
                keep.append((cam, msg))
        self._queue = keep

    def _apply(self, cam, msg, stamp):
        try:
            T = transform_to_T(self.tf.lookup_transform(self.P('map_frame'),
                                                        msg.header.frame_id, stamp))
        except TransformException:
            self._stats[cam]['dropped'] += 1
            return
        T_loc = None
        if cam == 'wrist':                                   # do chwytu: ramka odometrii
            try:
                T_loc = transform_to_T(self.tf.lookup_transform(
                    self.P('odom_frame'), msg.header.frame_id, stamp))
            except TransformException:
                pass
        poses = []
        for d in msg.detections:
            sp = detection_to_sample(d.bbox.center)          # w ramce optycznej kamery
            loc = (None, None) if T_loc is None else (
                T_loc[:3, :3] @ sp.position + T_loc[:3, 3], T_loc[:3, :3] @ sp.axis)
            poses.append(SamplePose(T[:3, :3] @ sp.position + T[:3, 3], T[:3, :3] @ sp.axis,
                                    *loc))
        dets = np.array([sp.position for sp in poses]).reshape(-1, 3)
        st = self._stats[cam]
        st['frames'] += 1
        st['with_dets'] += bool(len(poses))
        st['dets'] += len(poses)
        frame = Frame(stamp.nanoseconds * 1e-9, optical_camera_pose(T), dets, poses)
        if cam == 'front':
            self.mission.on_front_frame(frame)
        else:
            self.mission.on_wrist_frame(frame)

    def _log_stats(self):
        if self._started:
            self.get_logger().info('klatki ' + ', '.join(
                f'{cam}: {s["frames"]} (z detekcjami {s["with_dets"]}, detekcji {s["dets"]}, '
                f'odrzucone {s["dropped"]})' for cam, s in self._stats.items())
                + f', w kolejce {len(self._queue)}, timery: nav {self.mission.nav.ticks}'
                + (f', ramię {self.mission.manip.ticks}' if self.mission.manip else ''))

    def _tick_adapters(self):
        self.mission.nav.tick()
        if self.mission.manip is not None:
            self.mission.manip.tick()

    def _ready(self) -> bool:
        """Misja startuje dopiero, gdy są TF, Nav2 i (z nadgarstkiem) move_group."""
        if self._started:
            return True
        waiting = []
        if not self.mission.nav.client.server_is_ready():
            waiting.append('navigate_through_poses')
        if self.mission.manip is not None and not self.mission.manip.client.server_is_ready():
            waiting.append('move_group')
        if waiting:
            self.get_logger().info(f'czekam na: {", ".join(waiting)}', throttle_duration_sec=10.0)
            return False
        self._started = True
        return True

    def _step(self):
        q = self.robot_q()
        if q is None or self.mission is None or not self._ready():
            return
        was_finished = self.mission.finished
        self.mission.step(self.get_clock().now().nanoseconds * 1e-9, q)
        if self.mission.finished and not was_finished:
            self._publish_status()
            self._publish_belief()

    # ------------------------------------------------------------ wyjścia
    def _publish_status(self):
        if self.mission is None:
            return
        s = self.mission.status()
        m = MissionStatus()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.P('map_frame')
        m.state, m.strategy, m.reports = s['state'], s['strategy'], s['reports']
        m.elapsed, m.belief_mass = float(s['elapsed']), float(s['belief_mass'])
        m.belief_entropy, m.detail = float(s['belief_entropy']), s['detail']
        self._pubs['status'].publish(m)

    def _publish_belief(self):
        if self.mission is None:
            return
        b, g = self.mission.belief, self.mission.w.grid
        msg = OccupancyGrid()
        msg.header.frame_id = self.P('map_frame')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = float(g.res)
        msg.info.width, msg.info.height = g.width, g.height
        msg.info.origin.position.x, msg.info.origin.position.y = g.origin_x, g.origin_y
        msg.info.origin.orientation.w = 1.0
        p = b.p
        scale = max(float(p[b.active].max()) if b.active.any() else 0.0, 1e-6)
        data = np.where(b.active, np.round(100.0 * p / scale), -1).astype(np.int8)
        msg.data = data.tolist()
        self._pubs['belief'].publish(msg)
        self._publish_markers()

    def _publish_markers(self):
        frame = self.P('map_frame')
        arr = MarkerArray()
        arr.markers.append(Marker(action=Marker.DELETEALL))
        cands = find_candidates(self.mission.belief, self.mission.p.candidates)
        for i, c in enumerate(cands[:20]):
            for kind, mtype in ((0, Marker.CYLINDER), (1, Marker.TEXT_VIEW_FACING)):
                m = Marker(type=mtype, action=Marker.ADD, ns='candidates', id=2 * i + kind)
                m.header.frame_id = frame
                m.pose.position.x, m.pose.position.y = c.x, c.y
                m.pose.orientation.w = 1.0
                if kind == 0:
                    m.scale.x = m.scale.y = 0.3
                    m.scale.z = 0.05
                    m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 0.3 + 0.7 * c.mass
                else:
                    m.pose.position.z = 0.4
                    m.scale.z = 0.2
                    m.text = f'{c.mass:.2f}'
                    m.color.r = m.color.g = m.color.b = m.color.a = 1.0
                arr.markers.append(m)
        for r in self.mission.reports:
            m = Marker(type=Marker.CUBE, action=Marker.ADD, ns='reports', id=r.index)
            m.header.frame_id = frame
            m.pose.position.x, m.pose.position.y, m.pose.position.z = r.x, r.y, r.z
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.15
            m.color.g, m.color.a = 1.0, 0.9
            arr.markers.append(m)
        arr.markers += self._plan_markers(frame)
        self._pubs['markers'].publish(arr)

    def _plan_markers(self, frame):
        """Cel bieżącej akcji i dalsze zaplanowane pozy obserwacji: strzałki, numery, linia."""
        a = self.mission.action
        if a is None:
            return []
        poses = list(a.plan) or [(a.x, a.y, a.yaw)]
        hm = self.mission.w.heightmap
        out, line = [], Marker(type=Marker.LINE_STRIP, action=Marker.ADD, ns='plan', id=0)
        line.header.frame_id = frame
        line.pose.orientation.w = 1.0
        line.scale.x = 0.04
        line.color.r, line.color.g, line.color.b, line.color.a = 0.2, 0.6, 1.0, 0.8
        for k, (x, y, yaw) in enumerate(poses):
            z = float(hm.height_at(x, y)) + 0.3
            line.points.append(Point(x=float(x), y=float(y), z=z))
            fade = 1.0 / (1 + k)                    # dalsze pozy bledsze
            for kind, mtype in ((0, Marker.ARROW), (1, Marker.TEXT_VIEW_FACING)):
                m = Marker(type=mtype, action=Marker.ADD, ns='plan', id=1 + 2 * k + kind)
                m.header.frame_id = frame
                m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), z
                if kind == 0:
                    o = m.pose.orientation
                    o.x, o.y, o.z, o.w = yaw_to_quat(yaw)
                    m.scale.x, m.scale.y, m.scale.z = 0.6, 0.12, 0.12
                    m.color.r, m.color.g, m.color.b = 0.2, 0.6, 1.0
                    m.color.a = 0.4 + 0.6 * fade
                else:
                    m.pose.orientation.w = 1.0
                    m.pose.position.z = z + 0.3
                    m.scale.z = 0.25
                    m.text = str(k + 1) if a.kind == OBSERVE else 'dojazd'
                    m.color.r = m.color.g = m.color.b = m.color.a = 1.0
                out.append(m)
        if len(poses) > 1:
            out.append(line)
        return out


def main(args=None):
    rclpy.init(args=args)
    node = SearchNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
