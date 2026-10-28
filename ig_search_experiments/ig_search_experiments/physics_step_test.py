"""
Test kroku fizyki Gazebo: ta sama sekwencja jazdy, zapis prawdziwej pozy i RTF.

Bez Nav2 i misji - /cmd_vel prosto do kontrolera bazy, czasy w czasie symulacji. Przed jazdą
zrzut próbek ze stałym seedem (spawner) i zapis ich pozycji spoczynkowych. Wynik:
<output>/trajectory.csv (t, x, y, z, roll, pitch, yaw), probes.csv, summary.json.

  ros2 launch ig_search_experiments physics_test.launch.py world:=<sdf> output:=<katalog>
"""
import csv
import json
import math
import os
import time

from geometry_msgs.msg import TwistStamped
import numpy as np
from probe_detector.infrastructure.gz_world_poses import GzWorldPoses
from probe_supervisor_interfaces.srv import SpawnRandomProbes
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

# (czas trwania [s symulacji], v [m/s], ω [rad/s]) - start w (0, 0) przodem do +x
SCHEDULE = [
    (15.0, 0.5, 0.0),      # prosto po płaskim, do x ≈ 7,5
    (3.14, 0.0, -0.5),     # obrót w miejscu ~90° w prawo
    (9.0, 0.5, 0.0),       # w stronę wzgórza przy (5, -5): wjazd na zbocze
    (6.28, 0.0, 0.5),      # obrót ~180° na zboczu
    (9.0, 0.5, 0.0),       # zjazd
    (3.0, 0.0, 0.0),       # zatrzymanie
]


def rpy(T):
    R = T[:3, :3]
    return (math.atan2(R[2, 1], R[2, 2]), math.asin(max(-1.0, min(1.0, -R[2, 0]))),
            math.atan2(R[1, 0], R[0, 0]))


class PhysicsStepTest(Node):

    def __init__(self):
        super().__init__('physics_step_test')
        self.declare_parameter('output', '/tmp/physics_step_test')
        self.declare_parameter('probes', 5)
        self.out = self.get_parameter('output').value
        os.makedirs(self.out, exist_ok=True)
        self.gt = GzWorldPoses('erc2025', 'sirius', 'probe_object_')
        self.cmd = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.spawn = self.create_client(SpawnRandomProbes, '/spawn_random_probes')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def wait(self, sec, record=None):
        t_end = self.now() + sec
        while rclpy.ok() and self.now() < t_end:
            rclpy.spin_once(self, timeout_sec=0.02)
            if record is not None:
                record()

    def send(self, v, w):
        m = TwistStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'base_link'
        m.twist.linear.x, m.twist.angular.z = float(v), float(w)
        self.cmd.publish(m)

    def run(self):
        while rclpy.ok() and (self.gt.robot_latest() is None
                              or self.cmd.get_subscription_count() == 0 or self.now() < 1.0):
            rclpy.spin_once(self, timeout_sec=0.1)
        self.wait(5.0)                                  # ustalenie się robota
        while not self.spawn.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('czekam na spawner')
        fut = self.spawn.call_async(SpawnRandomProbes.Request(
            count=self.get_parameter('probes').value))
        while rclpy.ok() and not fut.done():
            rclpy.spin_once(self, timeout_sec=0.1)
        self.wait(8.0)                                  # próbki spadają i układają się
        probes = [T[:3, 3].tolist() for T in self.gt.probes()]

        rows = []
        last = [-1.0]

        def record():
            t = self.now()
            if t - last[0] >= 0.1:
                T = self.gt.robot_latest()
                if T is not None:
                    rows.append([t, *T[:3, 3].tolist(), *rpy(T)])
                    last[0] = t

        sim0, wall0 = self.now(), time.monotonic()
        for dur, v, w in SCHEDULE:
            t_end = self.now() + dur
            while rclpy.ok() and self.now() < t_end:
                self.send(v, w)
                self.wait(0.05, record)
        self.send(0.0, 0.0)
        rtf = (self.now() - sim0) / (time.monotonic() - wall0)

        with open(os.path.join(self.out, 'trajectory.csv'), 'w', newline='') as f:
            csv.writer(f).writerows([['t', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']] + rows)
        with open(os.path.join(self.out, 'probes.csv'), 'w', newline='') as f:
            csv.writer(f).writerows([['x', 'y', 'z']] + probes)
        a = np.array(rows)
        summary = {'rtf_driving': rtf, 'final': rows[-1][1:4] if rows else None,
                   'max_abs_roll_deg': float(np.degrees(np.abs(a[:, 4]).max())),
                   'max_abs_pitch_deg': float(np.degrees(np.abs(a[:, 5]).max())),
                   'n_samples': len(rows)}
        with open(os.path.join(self.out, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=1)
        self.get_logger().info(f'koniec testu: {summary}')


def main(args=None):
    rclpy.init(args=args)
    node = PhysicsStepTest()
    try:
        node.run()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
