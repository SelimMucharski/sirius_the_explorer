#!/usr/bin/env python3
"""ROS 2 adapter: depth image -> obstacle and ground clouds (PointCloud2 in base_link) for Nav2.

  <ns>/obstacles  points sticking out of the local ground  -> costmap marking
  <ns>/ground     points on the ground                      -> costmap clearing

Points on the robot's own arm are removed first: the collision boxes of `self_filter_links`
are read from /robot_description and placed with TF on every frame.

Classification is relative to the local ground (see sirius_navigation.obstacle_cloud), so
slopes of the Mars-yard terrain are not marked as obstacles, while low obstacles are, at
any distance (the costmap's own height filter works in the global frame).

With `terrain_heightmap` (.yaml {image: .npy, resolution, origin}, the prior height map known
before the mission) points are also classified against that terrain, looked up through the
`map_frame` -> base transform: piles of rocks, whose gaps the camera cannot see, are obstacles.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
import yaml

from sirius_navigation.obstacle_cloud import (
    ObstacleFilter,
    OrientedBox,
    PinholeIntrinsics,
    TerrainPrior,
    classify_points,
    depth_to_points,
)

ARM_LINKS = ['link_1', 'link_2', 'link_3', 'link_4', 'link_5', 'tcp_connector',
             'gripper_left', 'gripper_right', 'wrist_camera']


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch),
                              np.cos(yaw), np.sin(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def collision_boxes(urdf: str, links) -> dict:
    """{link: [(R, t, half_extents)]} for the box collision shapes of `links` (link frame)."""
    boxes = {}
    for link in ET.fromstring(urdf).findall('link'):
        if link.get('name') not in links:
            continue
        for collision in link.findall('collision'):
            box = collision.find('geometry/box')
            if box is None:
                continue
            origin = collision.find('origin')
            xyz = [float(v) for v in (origin.get('xyz', '0 0 0') if origin is not None else '0 0 0').split()]
            rpy = [float(v) for v in (origin.get('rpy', '0 0 0') if origin is not None else '0 0 0').split()]
            size = np.array([float(v) for v in box.get('size').split()])
            boxes.setdefault(link.get('name'), []).append((rpy_to_matrix(*rpy), np.array(xyz), size / 2))
    return boxes


def quaternion_to_matrix(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_terrain_prior(yaml_path: str, margin: float) -> TerrainPrior:
    """Height map .yaml {image: .npy (rows along +y), resolution, origin: [x, y]}."""
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    npy = meta['image']
    if not os.path.isabs(npy):
        npy = os.path.join(os.path.dirname(yaml_path), npy)
    return TerrainPrior(np.load(npy).astype(np.float64), float(meta['origin'][0]),
                        float(meta['origin'][1]), float(meta['resolution']), margin)


class ObstacleCloudNode(Node):

    def __init__(self):
        super().__init__('obstacle_cloud')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('min_height', 0.15)
        self.declare_parameter('max_height', 1.3)
        self.declare_parameter('min_range', 0.3)
        self.declare_parameter('max_range', 8.0)
        self.declare_parameter('stride', 4)
        self.declare_parameter('ground_cell_size', 0.15)  # [m] local ground window = 3 cells
        self.declare_parameter('depth_scale', 0.001)  # 16UC1 depth is in millimeters
        self.declare_parameter('self_filter_links', ARM_LINKS)
        self.declare_parameter('self_filter_padding', 0.05)  # [m]
        self.declare_parameter('terrain_heightmap', '')  # '' - local ground only
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('prior_margin', 0.05)  # [m]

        p = self.get_parameter
        self._base_frame = p('base_frame').value
        self._depth_scale = p('depth_scale').value
        self._filter = ObstacleFilter(p('min_height').value, p('max_height').value,
                                      p('min_range').value, p('max_range').value, p('stride').value,
                                      p('ground_cell_size').value)
        self._intrinsics = None
        self._self_links = list(p('self_filter_links').value)
        self._self_padding = p('self_filter_padding').value
        self._self_boxes = {}  # link -> boxes in the link frame, filled from robot_description
        self._map_frame = p('map_frame').value
        self._prior = None
        if p('terrain_heightmap').value:
            self._prior = load_terrain_prior(p('terrain_heightmap').value, p('prior_margin').value)
            self.get_logger().info(f'Terrain prior {self._prior.heights.shape} cells of '
                                   f"{self._prior.resolution} m from {p('terrain_heightmap').value}")

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._obstacles_pub = self.create_publisher(PointCloud2, 'obstacles', 5)
        self._ground_pub = self.create_publisher(PointCloud2, 'ground', 5)
        self.create_subscription(String, '/robot_description', self._on_description,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(CameraInfo, 'depth/camera_info', self._on_info, qos_profile_sensor_data)
        self.create_subscription(Image, 'depth/image_raw', self._on_depth, qos_profile_sensor_data)

    def _on_description(self, msg: String) -> None:
        self._self_boxes = collision_boxes(msg.data, self._self_links)
        self.get_logger().info(f'Self filter: {sum(map(len, self._self_boxes.values()))} boxes on '
                               f'{sorted(self._self_boxes)}')

    def _self_boxes_in_base(self):
        boxes = []
        for link, shapes in self._self_boxes.items():
            try:
                tf = self._tf_buffer.lookup_transform(self._base_frame, link, Time())
            except TransformException:
                continue
            rot = quaternion_to_matrix(tf.transform.rotation)
            t = tf.transform.translation
            pos = np.array([t.x, t.y, t.z])
            boxes += [OrientedBox(rot @ r, rot @ c + pos, half) for r, c, half in shapes]
        return boxes

    def _base_in_map(self):
        """(rotation, translation) of the base frame in the map frame, None if unknown."""
        if self._prior is None:
            return None
        try:
            tf = self._tf_buffer.lookup_transform(self._map_frame, self._base_frame, Time())
        except TransformException as e:
            self.get_logger().warn(f'No transform {self._base_frame} -> {self._map_frame}, '
                                   f'local ground only: {e}', throttle_duration_sec=5.0)
            return None
        t = tf.transform.translation
        return quaternion_to_matrix(tf.transform.rotation), np.array([t.x, t.y, t.z])

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
        obstacles, ground = classify_points(
            depth_to_points(depth, self._intrinsics, self._filter.stride),
            quaternion_to_matrix(tf.transform.rotation), np.array([t.x, t.y, t.z]), self._filter,
            self._self_boxes_in_base(), self._self_padding, self._prior, self._base_in_map())
        self._obstacles_pub.publish(self._to_cloud(obstacles, msg.header.stamp))
        self._ground_pub.publish(self._to_cloud(ground, msg.header.stamp))

    def _to_cloud(self, points: np.ndarray, stamp) -> PointCloud2:
        cloud = PointCloud2()
        cloud.header.stamp = stamp
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
        return cloud


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleCloudNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise  # a real error; otherwise spin was interrupted by shutdown (SIGINT race)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
