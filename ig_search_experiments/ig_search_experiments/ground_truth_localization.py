"""
Lokalizacja z prawdziwej pozy Gazebo (tylko symulacja, decyzja D3 w ig_search/docs/PDR.md).

Publikuje TF map -> odom tak, by map -> base_link było równe pozie modelu robota w świecie
Gazebo (/world/<świat>/pose/info przez gz-transport; ramka świata = map). odom -> base_link
pozostaje z EKF, więc Nav2 działa bez zmian. Uruchamiać z sirius_sim localization:=ground_truth.
"""
from geometry_msgs.msg import TransformStamped
from ig_search.infrastructure.ros_conversions import transform_to_T
import numpy as np
from probe_detector.infrastructure.gz_world_poses import GzWorldPoses
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener


def R_to_quat(R):
    """Kwaternion (x, y, z, w) z macierzy obrotu."""
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
    y = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
    z = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


class GroundTruthLocalization(Node):

    def __init__(self):
        super().__init__('ground_truth_localization')
        self.declare_parameter('world', 'erc2025')
        self.declare_parameter('robot_model', 'sirius')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('rate', 20.0)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.gt = GzWorldPoses(p('world'), p('robot_model'), probe_prefix='probe_object_',
                               rate=2 * p('rate'))
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self, spin_thread=True)
        self.broadcaster = TransformBroadcaster(self)
        self.create_timer(1.0 / p('rate'), self._publish)
        self.get_logger().info('map -> odom z prawdziwej pozy Gazebo')

    def _publish(self):
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        now = self.get_clock().now()
        # najnowsze z obu źródeł: przy 0,6 m/s rozjazd czasów o 50 ms to ~3 cm
        T_map_base = self.gt.robot_latest()
        if T_map_base is None:
            return
        try:
            tf = self.tf.lookup_transform(p('odom_frame'), p('base_frame'), Time())
        except TransformException as e:
            self.get_logger().warn(f'odom -> base_link: {e}', throttle_duration_sec=5.0)
            return
        T = T_map_base @ np.linalg.inv(transform_to_T(tf))
        m = TransformStamped()
        m.header.stamp = now.to_msg()
        m.header.frame_id = p('map_frame')
        m.child_frame_id = p('odom_frame')
        m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = (
            float(v) for v in T[:3, 3])
        q = R_to_quat(T[:3, :3])
        r = m.transform.rotation
        r.x, r.y, r.z, r.w = (float(v) for v in q)
        self.broadcaster.sendTransform(m)


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthLocalization()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
