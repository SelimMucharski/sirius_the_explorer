#!/usr/bin/env python3
"""ROS 2 adapter: depth image -> /front/obstacles (PointCloud2 in base_link) for Nav2 costmaps.

Height filtering happens in the robot frame so that slopes of the Mars-yard terrain
are not marked as obstacles (the costmap's own height filter works in the global frame).
"""

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from tf2_ros import Buffer, TransformException, TransformListener

from sirius_navigation.obstacle_cloud import (
    ObstacleFilter,
    PinholeIntrinsics,
    depth_to_points,
    filter_obstacles,
)


def quaternion_to_matrix(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class ObstacleCloudNode(Node):

    def __init__(self):
        super().__init__('obstacle_cloud')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('min_height', 0.15)
        self.declare_parameter('max_height', 1.3)
        self.declare_parameter('min_range', 0.3)
        self.declare_parameter('max_range', 8.0)
        self.declare_parameter('stride', 4)
        self.declare_parameter('max_slope', 0.35)  # tan of the steepest drivable incline
        self.declare_parameter('depth_scale', 0.001)  # 16UC1 depth is in millimeters

        p = self.get_parameter
        self._base_frame = p('base_frame').value
        self._depth_scale = p('depth_scale').value
        self._filter = ObstacleFilter(p('min_height').value, p('max_height').value,
                                      p('min_range').value, p('max_range').value, p('stride').value,
                                      p('max_slope').value)
        self._intrinsics = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._pub = self.create_publisher(PointCloud2, 'obstacles', 5)
        self.create_subscription(CameraInfo, 'depth/camera_info', self._on_info, qos_profile_sensor_data)
        self.create_subscription(Image, 'depth/image_raw', self._on_depth, qos_profile_sensor_data)

    def _on_info(self, msg: CameraInfo) -> None:
        self._intrinsics = PinholeIntrinsics(msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def _on_depth(self, msg: Image) -> None:
        if self._intrinsics is None:
            return
        if msg.encoding == '16UC1':
            depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width) * self._depth_scale
        elif msg.encoding == '32FC1':
            depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        else:
            self.get_logger().error(f'Unsupported depth encoding {msg.encoding}', once=True)
            return
        try:
            tf = self._tf_buffer.lookup_transform(self._base_frame, msg.header.frame_id, Time())
        except TransformException as e:
            self.get_logger().warn(f'No transform {msg.header.frame_id} -> {self._base_frame}: {e}',
                                   throttle_duration_sec=5.0)
            return
        t = tf.transform.translation
        points = filter_obstacles(
            depth_to_points(depth, self._intrinsics, self._filter.stride),
            quaternion_to_matrix(tf.transform.rotation), np.array([t.x, t.y, t.z]), self._filter)

        cloud = PointCloud2()
        cloud.header.stamp = msg.header.stamp
        cloud.header.frame_id = self._base_frame
        cloud.height = 1
        cloud.width = points.shape[0]
        cloud.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                        for i, n in enumerate('xyz')]
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = 12 * points.shape[0]
        cloud.is_dense = True
        cloud.data = points.tobytes()
        self._pub.publish(cloud)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleCloudNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
