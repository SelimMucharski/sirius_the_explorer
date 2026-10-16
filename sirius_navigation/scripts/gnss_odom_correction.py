#!/usr/bin/env python3
"""ROS 2 adapter: GNSS correction of the odometry drift, publishes map -> odom.

  gps/fix (sensor_msgs/NavSatFix) + odometry/filtered (odom -> base_link, speed, turn rate)
  -> TF map -> odom: a translation (sirius_navigation.gnss_correction) and the fixed rotation
     odom_yaw (the heading of the odom frame in the map frame, i.e. the rover's heading at
     start; 0 when it starts facing the map x axis)

The rover pose at the fix time is the odometry message with the nearest stamp; TF gives only
the fixed base_link -> antenna offset. Until the first fix map -> odom is the identity (same
as the static transform it replaces). A fix without a covariance (the Gazebo navsat sensor
leaves it at zero) uses horizontal_stddev. The datum is the map origin; sirius_sim.launch.py
reads it from the world's spherical_coordinates, so the map frame is the Gazebo world frame.
"""

from bisect import bisect_left
from collections import deque
import math

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import NavSatFix, NavSatStatus
from sirius_navigation.gnss_correction import Datum, OffsetFilter, OffsetNoise
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class GnssOdomCorrection(Node):

    def __init__(self):
        super().__init__('gnss_odom_correction')
        self.declare_parameter('datum_latitude', 0.0)
        self.declare_parameter('datum_longitude', 0.0)
        self.declare_parameter('datum_altitude', 0.0)
        self.declare_parameter('datum_heading', 0.0)       # ENU heading of the map x axis [rad]
        self.declare_parameter('odom_yaw', 0.0)            # heading of odom in map [rad]
        self.declare_parameter('horizontal_stddev', 0.85)  # [m], fixes without a covariance
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('max_odom_age', 0.1)        # [s] fix <-> odometry stamp
        self.declare_parameter('publish_rate', 30.0)       # [Hz] map -> odom
        self.declare_parameter('noise_still', 1e-4)        # [m^2/s]
        self.declare_parameter('noise_per_turn_rate', 0.02)
        self.declare_parameter('noise_per_speed', 2e-3)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self._datum = Datum(p('datum_latitude'), p('datum_longitude'), p('datum_altitude'),
                            p('datum_heading'))
        self._yaw = p('odom_yaw')
        self._filter = OffsetFilter(OffsetNoise(p('noise_still'), p('noise_per_turn_rate'),
                                                p('noise_per_speed')))
        self._speed = self._turn_rate = 0.0
        self._odom = deque(maxlen=500)     # (stamp [s], x, y, yaw) in odom, ~10 s at 50 Hz
        self._antenna = {}                 # antenna frame -> (x, y) in base_link
        self._rejected = 0
        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, self, spin_thread=True)
        self._broadcaster = TransformBroadcaster(self)
        self.create_subscription(NavSatFix, 'gps/fix', self._on_fix, qos_profile_sensor_data)
        self.create_subscription(Odometry, 'odometry/filtered', self._on_odom, 10)
        self.create_timer(1.0 / p('publish_rate'), self._publish)
        self.get_logger().info(
            f'datum {self._datum.latitude:.7f}, {self._datum.longitude:.7f}, '
            f'heading {math.degrees(self._datum.heading):.2f} deg')

    def _on_odom(self, msg: Odometry) -> None:
        twist = msg.twist.twist
        self._speed = math.hypot(twist.linear.x, twist.linear.y)
        self._turn_rate = twist.angular.z
        pose = msg.pose.pose
        self._odom.append((Time.from_msg(msg.header.stamp).nanoseconds * 1e-9,
                           pose.position.x, pose.position.y, _yaw(pose.orientation)))

    def _odom_at(self, t):
        """Odometry sample nearest to t, or None if none is within max_odom_age."""
        if not self._odom:
            return None
        stamps = [o[0] for o in self._odom]
        i = min(bisect_left(stamps, t), len(stamps) - 1)
        k = min((i, max(i - 1, 0)), key=lambda j: abs(stamps[j] - t))
        if abs(stamps[k] - t) > self.get_parameter('max_odom_age').value:
            return None
        return self._odom[k]

    def _antenna_offset(self, frame):
        """Antenna (x, y) in base_link, looked up once (fixed transform)."""
        if frame not in self._antenna:
            base = self.get_parameter('base_frame').value
            try:
                tf = self._tf.lookup_transform(base, frame, Time())
            except TransformException as e:
                self.get_logger().warn(f'{base} -> {frame}: {e}', throttle_duration_sec=5.0)
                return None
            self._antenna[frame] = (tf.transform.translation.x, tf.transform.translation.y)
        return self._antenna[frame]

    def _on_fix(self, msg: NavSatFix) -> None:
        if msg.status.status < NavSatStatus.STATUS_FIX or not math.isfinite(msg.latitude):
            return
        now = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        odom, antenna = self._odom_at(now), self._antenna_offset(msg.header.frame_id)
        if odom is None or antenna is None:
            self.get_logger().warn('no odometry at the fix time', throttle_duration_sec=5.0)
            return
        # antenna position in odom, then rotated into the map axes
        _, ox, oy, oyaw = odom
        c, s = math.cos(oyaw), math.sin(oyaw)
        ax, ay = ox + c * antenna[0] - s * antenna[1], oy + s * antenna[0] + c * antenna[1]
        c, s = math.cos(self._yaw), math.sin(self._yaw)
        antenna_odom_in_map = np.array([c * ax - s * ay, s * ax + c * ay])
        antenna_map = self._datum.to_map(msg.latitude, msg.longitude, msg.altitude)
        known = (msg.position_covariance_type != NavSatFix.COVARIANCE_TYPE_UNKNOWN
                 and msg.position_covariance[0] > 0.0)
        if known:
            variance = 0.5 * (msg.position_covariance[0] + msg.position_covariance[4])
        else:
            variance = self.get_parameter('horizontal_stddev').value ** 2
        self._filter.predict(now, self._speed, self._turn_rate)
        if not self._filter.update(now, antenna_map - antenna_odom_in_map, variance):
            self._rejected += 1
            self.get_logger().warn(f'GNSS fix rejected as an outlier ({self._rejected} so far)',
                                   throttle_duration_sec=5.0)

    def _publish(self) -> None:
        offset, _ = self._filter.state()
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.get_parameter('map_frame').value
        msg.child_frame_id = self.get_parameter('odom_frame').value
        msg.transform.translation.x = float(offset[0])
        msg.transform.translation.y = float(offset[1])
        msg.transform.rotation.z = math.sin(self._yaw / 2)
        msg.transform.rotation.w = math.cos(self._yaw / 2)
        self._broadcaster.sendTransform(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GnssOdomCorrection()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
