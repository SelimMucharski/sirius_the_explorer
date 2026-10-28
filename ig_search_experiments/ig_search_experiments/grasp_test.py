"""
Test samego chwytu: próbka przed łazikiem, pozycja z Gazebo, adapter MoveItManipulator z ig_search.

Dla każdej próby: zrzut próbki w zasięgu ramienia (odległość od podstawy ramienia, przesunięcie
w bok i kurs osi próbki z list parametrów), pozycja przed chwytem, chwyt, pozycja transportowa.
Mierzone z Gazebo: podniesienie środka próbki po chwycie i czy po dojściu do pozycji transportowej
próbka jest przy chwytaku. Wynik: <output>/summary.json.

  ros2 launch ig_search_experiments grasp_test.launch.py output:=<katalog> [grasp_depth:=0.012]
"""
import json
import math
import os
import subprocess
import threading
import time

from ament_index_python.packages import get_package_share_directory
from ig_search.application.ports import SamplePose, Status
from ig_search.infrastructure.arm_kinematics import ARM_JOINTS, ARM_ROOT, FINGERTIP_OFFSET
from ig_search.infrastructure.moveit_manipulator import MoveItManipulator
from ig_search_experiments.gz_service import remove_model
import numpy as np
from probe_detector.infrastructure.gz_world_poses import GzWorldPoses
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


class GraspTest(Node):

    def __init__(self):
        super().__init__('grasp_test')
        P = self.declare_parameter
        P('output', '/tmp/grasp_test')
        P('world', 'empty')
        P('reach', [0.70])                  # [m] od podstawy ramienia
        P('lateral_deg', [0.0, 30.0, -30.0])
        P('yaw_deg', [0.0, 45.0, 90.0])     # kurs osi próbki względem łazika
        P('grasp_depth', 0.012)
        P('carry_scaling', 0.2)
        P('step_time', 1.2)
        P('offset', 0.0)                    # [m] błąd w poprzek próbki, znak naprzemiennie
        P('upright', False)                 # próbka wbita pionowo (statyczna, jak spawner)
        P('bury', 0.35)                     # zakopana część długości (upright)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.out, self.world = g('output'), g('world')
        os.makedirs(self.out, exist_ok=True)
        self.trials = [(r, la, yw) for r in g('reach') for la in g('lateral_deg')
                       for yw in g('yaw_deg')]
        self.gt = GzWorldPoses(self.world, 'sirius', 'gt_probe')
        self.manip = MoveItManipulator(self, self.base_T, grasp_depth=g('grasp_depth'),
                                       carry_scaling=g('carry_scaling'),
                                       grasp_step_time=g('step_time'))
        self.create_timer(0.05, self.manip.tick)
        self.sdf = os.path.join(get_package_share_directory('probe_supervisor'),
                                'objects', 'probe', 'probe.sdf')

    def base_T(self):
        return self.gt.robot_latest()

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def sleep(self, sec):
        t_end = self.now() + sec
        while rclpy.ok() and self.now() < t_end:
            time.sleep(0.02)

    def wait(self, timeout):
        t_end = self.now() + timeout
        while rclpy.ok() and self.manip.status() == Status.ACTIVE and self.now() < t_end:
            time.sleep(0.05)
        return self.manip.status()

    def probe(self, name):
        with self.gt._lock:
            T = self.gt._probes.get(name)
        return None if T is None else T.copy()

    def tip_world(self):
        T = self.base_T()
        tcp = self.manip._ik.tcp.fk(dict(zip(ARM_JOINTS, self.manip._q)))
        return (T @ np.r_[tcp[:3, 3] + tcp[:3, 2] * FINGERTIP_OFFSET, 1.0])[:3]

    def spawn(self, name, reach, lateral, yaw):
        T = self.base_T()
        psi = math.atan2(T[1, 0], T[0, 0])
        b = np.array([ARM_ROOT[0] + reach * math.cos(lateral), reach * math.sin(lateral), 0.0])
        p = T[:3, :3] @ b + T[:3, 3]
        if self.get_parameter('upright').value:
            # wbita pionowo czubkiem w dół, statyczna (grunt płaski: z = 0 świata)
            with open(self.sdf) as f:
                sdf = f.read().replace('<model name="probe">',
                                       '<model name="probe">\n    <static>true</static>', 1)
            z = 0.1 - self.get_parameter('bury').value * 0.2
            subprocess.run(['ros2', 'run', 'ros_gz_sim', 'create', '-string', sdf, '-name', name,
                            '-x', f'{p[0]}', '-y', f'{p[1]}', '-z', f'{z}', '-Y', f'{psi + yaw}'],
                           capture_output=True, timeout=10)
            return
        # walec z osią z położony (roll 90°): oś wzdłuż y modelu, obrócona o kurs
        subprocess.run(['ros2', 'run', 'ros_gz_sim', 'create', '-file', self.sdf, '-name', name,
                        '-x', f'{p[0]}', '-y', f'{p[1]}', '-z', f'{p[2] + 0.08}',
                        '-R', f'{math.pi / 2}', '-P', '0', '-Y', f'{psi + yaw}'],
                       capture_output=True, timeout=10)

    def remove(self, name):
        remove_model(self.world, name)

    def run(self):
        while rclpy.ok() and (self.base_T() is None or self.manip._ik is None
                              or not self.manip.client.server_is_ready()):
            self.get_logger().info(
                f'czekam: poza łazika {self.base_T() is not None}, URDF '
                f'{self.manip._ik is not None}, move_group {self.manip.client.server_is_ready()}',
                throttle_duration_sec=10.0)
            time.sleep(0.2)
        self.get_logger().info(f'start: {len(self.trials)} prób')
        self.sleep(3.0)
        self.manip.stow()
        self.wait(30.0)
        results = []
        for k, (reach, lat, yaw) in enumerate(self.trials):
            name = f'gt_probe_{k}'
            self.spawn(name, reach, math.radians(lat), math.radians(yaw))
            self.sleep(3.0)
            T0 = self.probe(name)
            r = {'trial': k, 'reach': reach, 'lateral_deg': lat, 'yaw_deg': yaw}
            if T0 is None:
                r['error'] = 'brak próbki'
                results.append(r)
                continue
            a = T0[:3, 2].copy()
            side = np.cross([0.0, 0.0, 1.0], a)     # poziomo, w poprzek osi próbki
            side /= max(np.linalg.norm(side), 1e-9)
            off = self.get_parameter('offset').value * (1 if k % 2 == 0 else -1)
            sp = SamplePose(T0[:3, 3] + off * side, a)
            r['offset'] = off
            self.manip.pregrasp(sp)
            r['pregrasp'] = self.wait(40.0).name
            if r['pregrasp'] == 'SUCCEEDED':
                self.manip.grasp(sp)
                r['grasp'] = self.wait(60.0).name
                r['jaws_after_lift'] = [round(float(v), 4) for v in self.manip._jaws]
                T1 = self.probe(name)
                if T1 is None:
                    r['probe_lost'] = 'po chwycie'
                else:
                    r['lift_dz'] = round(float(T1[2, 3] - T0[2, 3]), 3)
                    cos_tilt = min(1.0, abs(float(T1[:3, 2] @ T0[:3, 2])))
                    r['tilt_deg'] = round(math.degrees(math.acos(cos_tilt)), 1)
            self.manip.stow()
            r['stow'] = self.wait(60.0).name
            self.sleep(1.0)
            T2 = self.probe(name)
            if T2 is not None:
                r['probe_after_stow'] = np.round(T2[:3, 3], 3).tolist()
                r['tip_after_stow'] = np.round(self.tip_world(), 3).tolist()
            r['held_after_stow'] = bool(T2 is not None and
                                        np.linalg.norm(T2[:3, 3] - self.tip_world()) < 0.15)
            self.get_logger().info(f'próba {r}')
            results.append(r)
            self.remove(name)
            self.manip._holding = False
            self.manip._attach_probe(False)
            self.sleep(1.0)
        summary = {'params': {n: self.get_parameter(n).value for n in
                              ('grasp_depth', 'carry_scaling', 'step_time', 'offset',
                               'upright')},
                   'held': sum(r.get('held_after_stow', False) for r in results),
                   'grasped': sum(r.get('grasp') == 'SUCCEEDED' for r in results),
                   'trials': results}
        with open(os.path.join(self.out, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=1)
        self.get_logger().info(f'koniec: utrzymane {summary["held"]}/{len(results)}, '
                               f'chwycone {summary["grasped"]}/{len(results)}')


def main(args=None):
    rclpy.init(args=args)
    node = GraspTest()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    th = threading.Thread(target=ex.spin, daemon=True)
    th.start()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
