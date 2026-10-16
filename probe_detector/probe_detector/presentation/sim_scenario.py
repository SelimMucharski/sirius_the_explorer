"""
Simulation test scenario for the probe detectors (launch/sim_probe_test.launch.py).

World erc2025, manipulator 6DOF, the rover at the start point S facing +x:
 1. spawns probes with probe_supervisor: `far` ahead of the rover (front camera) and `near`
    just in front of it (wrist camera, within reach of the arm),
 2. wrist phases, once per jaw opening in gripper_openings: for every near probe and each
    distance in wrist_distances, moves the arm so that the wrist camera looks at the probe from
    that distance (IK on the URDF chain from /robot_description), dwells,
 3. front phase: arm straight up (out of the front camera's view), the rover turns in place
    (two turns) so the probes cross the front camera's view at every bearing, then drives
    1.6 m forward and back,
 4. ends (the launch file shuts down, the sim evaluator writes its report on exit).
The current phase is published on /probe_sim_scenario/phase for the evaluator.
"""

import math
import time
import xml.etree.ElementTree as ET

from geometry_msgs.msg import TwistStamped
import numpy as np
from probe_detector.infrastructure.gz_world_poses import GzWorldPoses
from probe_supervisor_interfaces.srv import SpawnRandomProbes
from rcl_interfaces.msg import Parameter as ParamMsg
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy.optimize import least_squares
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_JOINTS = [f'joint_{i}' for i in range(1, 7)]
HOME = [0.0] * 6                     # arm straight up: out of the front camera's view


def rpy_to_R(r, p, y):
    cr, sr, cp, sp = math.cos(r), math.sin(r), math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_R(axis, q):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * K @ K


def _floats(el, attr, default):
    return [float(v) for v in (el.get(attr, default) if el is not None else default).split()]


class Chain:
    """Kinematic chain base -> tip from a URDF string (fixed + revolute joints)."""

    def __init__(self, urdf, base, tip):
        joints = {j.find('child').get('link'): j for j in ET.fromstring(urdf).findall('joint')}
        self.steps, link = [], tip
        while link != base:
            j = joints[link]
            o = j.find('origin')
            ax = j.find('axis')
            lim = j.find('limit')
            self.steps.append({
                'name': j.get('name'), 'type': j.get('type'),
                'xyz': np.array(_floats(o, 'xyz', '0 0 0')),
                'R': rpy_to_R(*_floats(o, 'rpy', '0 0 0')),
                'axis': _floats(ax, 'xyz', '0 0 1'),
                'lim': (float(lim.get('lower')), float(lim.get('upper'))) if lim is not None
                else None})
            link = j.find('parent').get('link')
        self.steps.reverse()

    def fk(self, q):
        T = np.eye(4)
        for s in self.steps:
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = s['R'], s['xyz']
            T = T @ M
            if s['type'] == 'revolute' and s['name'] in q:
                R = np.eye(4)
                R[:3, :3] = axis_R(s['axis'], q[s['name']])
                T = T @ R
        return T

    def limits(self, names):
        lim = {s['name']: s['lim'] for s in self.steps}
        return [lim[n] for n in names]


class Scenario(Node):

    def __init__(self):
        super().__init__('probe_sim_scenario')
        self.declare_parameter('far_count', 8)
        self.declare_parameter('near_count', 3)
        self.declare_parameter('wrist_distances', [0.25, 0.4, 0.6])
        # jaw opening per side for the wrist passes [m]: 0.0326 = fully open, 0.0 = closed
        self.declare_parameter('gripper_openings', [0.0326, 0.0])
        self.declare_parameter('dwell', 3.0)
        self.declare_parameter('turns', 2.0)
        self.declare_parameter('turn_rate', 0.25)
        self.urdf = None
        self.gt = GzWorldPoses()
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/robot_description',
                                 lambda m: setattr(self, 'urdf', m.data), latched)
        self.arm_pub = self.create_publisher(JointTrajectory,
                                             '/manip_controller/joint_trajectory', 10)
        self.grip_pub = self.create_publisher(JointTrajectory,
                                              '/gripper_controller/joint_trajectory', 10)
        self.phase_pub = self.create_publisher(String, '/probe_sim_scenario/phase', latched)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.spawn_cli = self.create_client(SpawnRandomProbes, '/spawn_random_probes')
        self.param_cli = self.create_client(SetParameters,
                                            '/probe_spawner_node/set_parameters')

    # ------------------------------------------------------------ helpers
    def wait(self, sec):
        end = self.get_clock().now() + Duration(seconds=sec)
        while rclpy.ok() and self.get_clock().now() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_for(self, cond, what, timeout=120.0):
        t0 = time.time()
        while rclpy.ok() and not cond():
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - t0 > timeout:
                raise RuntimeError(f'timeout waiting for {what}')

    def call(self, cli, req):
        self.wait_for(cli.service_is_ready, cli.srv_name)
        fut = cli.call_async(req)
        self.wait_for(fut.done, cli.srv_name)
        return fut.result()

    def spawn(self, count, **area):
        req = SetParameters.Request()
        for k, v in area.items():
            value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(v))
            req.parameters.append(ParamMsg(name=k, value=value))
        self.call(self.param_cli, req)
        res = self.call(self.spawn_cli, SpawnRandomProbes.Request(count=count))
        self.get_logger().info(res.message)

    def _trajectory(self, pub, joints, positions, sec):
        self.wait_for(lambda: pub.get_subscription_count() > 0, pub.topic_name)
        msg = JointTrajectory(joint_names=joints)
        msg.points.append(JointTrajectoryPoint(positions=[float(v) for v in positions],
                                               time_from_start=Duration(seconds=sec).to_msg()))
        pub.publish(msg)

    def move_arm(self, q, sec=8.0):
        self._trajectory(self.arm_pub, ARM_JOINTS, q, sec)
        self.wait(sec + 1.0)

    def set_gripper(self, opening, sec=2.0):
        self._trajectory(self.grip_pub, ['joint_gripper_left', 'joint_gripper_right'],
                         [opening] * 2, sec)
        self.wait(sec + 0.5)

    def phase(self, name):
        """Tag the following results for the evaluator."""
        self.phase_pub.publish(String(data=name))
        self.get_logger().info(f'phase: {name}')

    def drive(self, v, w, sec):
        end = self.get_clock().now() + Duration(seconds=sec)
        while rclpy.ok() and self.get_clock().now() < end:
            m = TwistStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = 'base_link'
            m.twist.linear.x, m.twist.angular.z = v, w
            self.cmd_pub.publish(m)
            self.wait(0.05)
        stop = TwistStamped()
        stop.header.stamp = self.get_clock().now().to_msg()
        self.cmd_pub.publish(stop)

    def ik_look_at(self, chain, target, dist, q0):
        """
        Find joint angles pointing the wrist camera at a target.

        The optical axis goes through target (base_link) at distance dist, the camera stays
        above the ground.
        """
        lim = np.array(chain.limits(ARM_JOINTS))
        best = None
        rng = np.random.default_rng(0)

        def res(x):
            T = chain.fk(dict(zip(ARM_JOINTS, x)))
            c, z = T[:3, 3], T[:3, 2]
            d = target - c
            n = np.linalg.norm(d)
            return np.r_[np.cross(z, d / n) * 2.0, (n - dist) * 3.0,
                         max(0.0, 0.12 - c[2]) * 10.0, 0.02 * (x - q0)]

        for k in range(12):
            x0 = q0 if k == 0 else rng.uniform(lim[:, 0] * 0.8, lim[:, 1] * 0.8)
            r = least_squares(res, np.clip(x0, lim[:, 0] + 1e-3, lim[:, 1] - 1e-3),
                              bounds=(lim[:, 0], lim[:, 1]))
            T = chain.fk(dict(zip(ARM_JOINTS, r.x)))
            ok = (T[:3, 2] @ (target - T[:3, 3]) > 0
                  and abs(np.linalg.norm(target - T[:3, 3]) - dist) < 0.03)
            if ok and (best is None or r.cost < best.cost):
                best = r
        return None if best is None else best.x

    # ------------------------------------------------------------ run
    def run(self):
        self.wait_for(lambda: self.urdf is not None and self.gt.robot_latest() is not None,
                      'robot_description + Gazebo poses')
        chain = Chain(self.urdf, 'base_link', 'wrist_left_camera_optical')
        self.move_arm(HOME)
        p = self.get_parameter
        # robot at S facing +x: near probes within the arm's reach, far ones in front
        self.spawn(p('near_count').value, area_min_x=0.75, area_max_x=1.05, area_min_y=-0.45,
                   area_max_y=0.45, spawn_z=0.8, min_separation=0.3, keep_out_radius=0.6)
        self.spawn(p('far_count').value, area_min_x=1.5, area_max_x=6.5, area_min_y=-3.0,
                   area_max_y=3.0, spawn_z=0.8, min_separation=0.6, keep_out_radius=1.2)
        self.wait(4.0)                                           # probes fall and settle

        Tb_inv = np.linalg.inv(self.gt.robot_latest())
        near = [c for c in ((Tb_inv @ Tp[:, 3])[:3] for Tp in self.gt.probes())
                if c[0] < 1.3 and abs(c[1]) < 0.6]
        self.get_logger().info(f'wrist phase: {len(near)} probes within reach')
        q = np.array(HOME)
        for opening in p('gripper_openings').value:
            self.set_gripper(opening)
            self.phase(f'wrist_{"open" if opening > 0.015 else "closed"}')
            for c in near:
                for dist in p('wrist_distances').value:
                    sol = self.ik_look_at(chain, c, dist, q)
                    if sol is None:
                        self.get_logger().warn(f'no IK for probe at {c.round(2)} from {dist} m')
                        continue
                    q = sol
                    self.get_logger().info(f'looking at {c.round(2)} from {dist} m')
                    self.move_arm(q, 8.0)
                    self.wait(p('dwell').value)

        self.phase('front')
        self.move_arm(HOME)
        w = p('turn_rate').value
        sec = 2 * math.pi * p('turns').value / w
        self.get_logger().info(f'turning {p("turns").value} turns at {w} rad/s ({sec:.0f} s)')
        self.drive(0.0, w, sec)
        self.get_logger().info('driving forward and back')
        self.drive(0.2, 0.0, 8.0)                               # probes come closer
        self.drive(-0.2, 0.0, 8.0)
        self.wait(2.0)
        self.get_logger().info('scenario done')


def main(args=None):
    rclpy.init(args=args)
    node = Scenario()
    try:
        node.run()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
