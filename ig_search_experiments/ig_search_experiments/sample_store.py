"""
Symulacja trzymania i zebrania próbki przez chwytak (komunikaty ig_search, ramka map).

- ~/sample_grasped (punkt między szczękami po ich zamknięciu na próbce): jeśli oś którejś
  próbki przechodzi bliżej niż `grasp_distance` od tego punktu (szczęki naprawdę objęły
  próbkę), próbka jest zastępowana kopią `grasped_probe` w tej samej pozie, którą
  DetachableJoint w modelu łazika (6DOF.urdf.xacro) przyspawa do tcp_connector - w DART
  próbka trzymana samym tarciem wypadała w drodze do pozycji transportowej. Wynik:
  /ig_search/sample_attached (Bool).
- ~/sample_stored (ramię z próbką w pozycji transportowej): próbka zebrana - odłączenie
  i usunięcie `grasped_probe` z Gazebo. Bez przyspawanej próbki - jak wcześniej: usuwana
  próbka najbliższa chwytakowi, o ile bliżej niż `max_distance`.
- ~/sample_released (chwytak otwarty z próbką): odłączenie, próbka wraca do świata pod
  pierwotną nazwą w bieżącej pozie i spada.

Kopia pod stałą nazwą, bo DetachableJoint ma nazwę modelu dziecka w konfiguracji i dołącza
go, gdy tylko model istnieje - prawdziwych próbek nie można więc wskazać mu z góry.

  ros2 run ig_search_experiments sample_store
"""
import math
import os
import threading
import time

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PointStamped
from ig_search_experiments.gz_service import create_model, remove_model
import numpy as np
from probe_detector.infrastructure.gz_world_poses import GzWorldPoses
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool

GRASPED = 'grasped_probe'           # child_model DetachableJoint w 6DOF.urdf.xacro
JOINT_TOPIC = '/sirius/grasped_probe'
PROBE_HALF_LENGTH = 0.1             # [m] walec próbki (probe.sdf) wzdłuż osi z modelu
WAIT = 5.0                          # [s ściennych] na model w świecie i stan złącza


def _quat(R):
    """Kwaternion (x, y, z, w) z macierzy obrotu."""
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2,
                      R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2,
                      R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2,
                      R[1, 0] - R[0, 1])
    return x, y, z, w


def axis_distance(T, p) -> float:
    """Odległość punktu p od odcinka osi próbki o pozie T (środek, oś z modelu)."""
    c, a = T[:3, 3], T[:3, 2]
    t = float(np.clip(np.dot(np.asarray(p) - c, a), -PROBE_HALF_LENGTH, PROBE_HALF_LENGTH))
    return float(np.linalg.norm(np.asarray(p) - (c + t * a)))


class SampleStore(Node):

    def __init__(self):
        super().__init__('sample_store')
        self.declare_parameter('world', 'erc2025')
        self.declare_parameter('max_distance', 0.3)       # [m] próbka w chwytaku (bez spawu)
        self.declare_parameter('grasp_distance', 0.04)    # [m] oś próbki od punktu chwytu
        self.world = self.get_parameter('world').value
        self.max_distance = self.get_parameter('max_distance').value
        self.grasp_distance = self.get_parameter('grasp_distance').value
        self.sdf = os.path.join(get_package_share_directory('probe_supervisor'),
                                'objects', 'probe', 'probe.sdf')
        # 10 Hz wystarcza (próbka między szczękami stoi); pełne tempo /pose/info zjadało CPU
        self.gt = GzWorldPoses(self.world, 'sirius', 'probe_object_', rate=10)
        self.gt_held = GzWorldPoses(self.world, 'sirius', GRASPED, rate=10)
        self.held = None                # pierwotna nazwa przyspawanej próbki
        from gz.msgs10.empty_pb2 import Empty
        from gz.msgs10.stringmsg_pb2 import StringMsg
        from gz.transport13 import Node as GzNode
        self._empty = Empty()
        self._gz = GzNode()
        self._attach = self._gz.advertise(f'{JOINT_TOPIC}/attach', Empty)
        self._detach = self._gz.advertise(f'{JOINT_TOPIC}/detach', Empty)
        self._joint_state = None
        self._joint_changed = threading.Event()
        self._gz.subscribe(StringMsg, f'{JOINT_TOPIC}/state', self._on_joint_state)
        self._attached_pub = self.create_publisher(Bool, '/ig_search/sample_attached', 10)
        self.create_subscription(PointStamped, '/ig_search/sample_grasped', self.on_grasped, 10)
        self.create_subscription(PointStamped, '/ig_search/sample_stored', self.on_stored, 10)
        self.create_subscription(PointStamped, '/ig_search/sample_released', self.on_released,
                                 10)

    def _on_joint_state(self, msg):
        self._joint_state = msg.data
        self._joint_changed.set()

    def _wait(self, cond) -> bool:
        t0 = time.monotonic()
        while not cond():
            if time.monotonic() - t0 > WAIT:
                return False
            time.sleep(0.05)
        return True

    def _probe_poses(self):
        with self.gt._lock:
            return {k: v.copy() for k, v in self.gt._probes.items()}

    def _held_pose(self):
        with self.gt_held._lock:
            T = self.gt_held._probes.get(GRASPED)
            return None if T is None else T.copy()

    def _spawn(self, name, T) -> bool:
        return create_model(self.world, self.sdf, name, tuple(float(v) for v in T[:3, 3]),
                            _quat(T[:3, :3]))

    def on_grasped(self, msg):
        p = (msg.point.x, msg.point.y, msg.point.z)
        probes = self._probe_poses()
        ok = False
        if self.held is not None:
            self.get_logger().warn(f'{self.held} już w chwytaku - pomijam')
        elif not probes:
            self.get_logger().warn('brak próbek w świecie')
        else:
            name = min(probes, key=lambda k: axis_distance(probes[k], p))
            d = axis_distance(probes[name], p)
            if d > self.grasp_distance:
                self.get_logger().warn(f'oś najbliższej próbki {name} {100 * d:.1f} cm od '
                                       'punktu chwytu - szczęki jej nie objęły, bez spawu')
            elif not remove_model(self.world, name):
                self.get_logger().error(f'nie udało się usunąć {name}')
            elif not self._spawn(GRASPED, probes[name]):
                self.get_logger().error(f'nie udało się utworzyć {GRASPED} - {name} przywracana')
                self._spawn(name, probes[name])
            else:
                self.held = name
                # pierwsze pojawienie się modelu dziecka dołącza samo, po odłączeniu - na
                # żądanie; żądanie dopiero, gdy model jest w świecie
                self._wait(lambda: self._held_pose() is not None)
                self._attach.publish(self._empty)
                ok = self._wait(lambda: self._joint_state == 'attached')
                if ok:
                    self.get_logger().info(f'{name} w chwytaku (oś {100 * d:.1f} cm od punktu '
                                           'chwytu) - przyspawana do tcp_connector')
                else:
                    self.get_logger().warn(f'{name} w chwytaku jako {GRASPED}, ale złącze '
                                           f'nie potwierdziło spawu (stan {self._joint_state})')
        self._attached_pub.publish(Bool(data=ok))

    def on_stored(self, msg):
        if self.held is not None:
            self._detach.publish(self._empty)
            if remove_model(self.world, GRASPED):
                self.get_logger().info(f'{self.held} zebrana - usunięta z Gazebo')
            else:
                self.get_logger().error(f'nie udało się usunąć {GRASPED}')
            self.held = None
            return
        probes = {k: v[:3, 3] for k, v in self._probe_poses().items()}
        p = (msg.point.x, msg.point.y, msg.point.z)
        if not probes:
            self.get_logger().warn('brak próbek w świecie')
            return
        name = min(probes, key=lambda k: math.dist(probes[k], p))
        d = math.dist(probes[name], p)
        if d > self.max_distance:
            self.get_logger().warn(f'najbliższa próbka {name} {d:.2f} m od chwytaka - '
                                   'nie usuwam (chwytak pusty?)')
            return
        if remove_model(self.world, name):
            self.get_logger().info(f'{name} zebrana ({d:.2f} m od chwytaka) - usunięta z Gazebo')
        else:
            self.get_logger().error(f'nie udało się usunąć {name}')

    def on_released(self, msg):
        if self.held is None:
            return
        name, self.held = self.held, None
        self._detach.publish(self._empty)
        T = self._held_pose()
        if T is None or not remove_model(self.world, GRASPED):
            self.get_logger().error(f'{GRASPED}: brak pozy albo nie dało się usunąć')
            return
        if self._spawn(name, T):
            self.get_logger().info(f'{name} upuszczona z chwytaka w ({T[0, 3]:.2f}, '
                                   f'{T[1, 3]:.2f}, {T[2, 3]:.2f})')
        else:
            self.get_logger().error(f'nie udało się przywrócić {name}')


def main(args=None):
    rclpy.init(args=args)
    node = SampleStore()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
