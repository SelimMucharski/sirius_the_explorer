import math
import os
import random
import subprocess
import threading
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from std_srvs.srv import Trigger
from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from ament_index_python.packages import get_package_share_directory

from probe_supervisor_interfaces.srv import SpawnRandomProbes
from probe_supervisor.domain.spawn_area import MaskedArea, RectangleArea, TraversabilityMask
from probe_supervisor.domain.orientation import Orientation, sample_orientation
from probe_supervisor.domain.upright import (
    Footprints, HeightGrid, sample_upright, UprightParams)
from probe_supervisor.domain.placement import (
    KeepOutCircle, PlacementConstraints, sample_positions)


class ProbeSpawnerNode(Node):

    def __init__(self):
        super().__init__('probe_spawner_node')

        # Random-spawn area and constraints (world frame); read on every call,
        # so they can be changed at runtime with `ros2 param set`.
        self.declare_parameter('area_min_x', -5.0)
        self.declare_parameter('area_max_x', 5.0)
        self.declare_parameter('area_min_y', -5.0)
        self.declare_parameter('area_max_y', 5.0)
        self.declare_parameter('spawn_z', 0.5)
        self.declare_parameter('min_separation', 0.5)
        # Keep probes away from the robot's spawn point
        self.declare_parameter('keep_out_x', 0.0)
        self.declare_parameter('keep_out_y', 0.0)
        self.declare_parameter('keep_out_radius', 1.0)
        self.declare_parameter('max_attempts_per_probe', 200)
        # Traversability map (nav2_map_server YAML, white = traversable): probes only on it,
        # at least map_clearance from its edge. '' -> the rectangle alone.
        self.declare_parameter('traversability_map', '')
        self.declare_parameter('map_clearance', 0.5)
        # Spots where a probe stays visible (same format; e.g. without places where the
        # collision mesh lies below the visual one and the probe sinks out of sight).
        # Checked at the point only. '' -> not used.
        self.declare_parameter('spawn_map', '')
        # Per-probe pose bridges (separate processes, ~50 Hz each) only feed probes_poses /
        # probes_markers; headless experiment runs switch them off to save CPU for Gazebo.
        self.declare_parameter('track_poses', True)
        # Probes stuck upright into the ground (as at ERC): this share of the probes is spawned
        # static, tip down, partly buried, at the visual-terrain height from `heightmap`
        # (ig_search format); not into stones/slabs from `upright_keep_out` ({discs: [[x, y, r]]}).
        # Without a height map all probes are dropped as before.
        self.declare_parameter('upright_fraction', 0.0)
        self.declare_parameter('upright_bury_min', 0.25)
        self.declare_parameter('upright_bury_max', 0.45)
        self.declare_parameter('upright_max_tilt_deg', 12.0)
        self.declare_parameter('heightmap', '')
        self.declare_parameter('upright_keep_out', '')
        self._maps = {}
        # < 0 -> nondeterministic
        self.declare_parameter('random_seed', -1)

        seed = self.get_parameter('random_seed').value
        self.rng = random.Random(None if seed < 0 else seed)

        # Spawning blocks on subprocesses; keep it off the timer's callback group
        # so poses keep being published while a batch is spawning.
        srv_group = MutuallyExclusiveCallbackGroup()
        self.srv = self.create_service(
            Trigger, 'spawn_probe_trigger', self.spawn_callback,
            callback_group=srv_group
        )
        self.random_srv = self.create_service(
            SpawnRandomProbes, 'spawn_random_probes', self.spawn_random_callback,
            callback_group=srv_group
        )

        self.pose_pub = self.create_publisher(PoseArray, 'probes_poses', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'probes_markers', 10)

        self.lock = threading.Lock()
        self.spawned_probes = {}
        self.subscribers = {}
        self.bridge_processes = {}
        self.counter = 0

        self.timer = self.create_timer(0.1, self.update_callback)

        self.get_logger().info('ProbeSpawner ready.')

    def spawn_callback(self, request, response):
        x, y = 1.0, float(self.counter) * 0.5
        name, error = self.spawn_probe(x, y, self.get_parameter('spawn_z').value)
        response.success = error is None
        response.message = f'Spawned {name} with bridge' if name else error
        return response

    def spawn_random_callback(self, request, response):
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        try:
            area = RectangleArea(p('area_min_x'), p('area_max_x'),
                                 p('area_min_y'), p('area_max_y'))
        except ValueError as e:
            response.success = False
            response.message = str(e)
            return response
        masks = {}
        for name in ('traversability_map', 'spawn_map'):
            path = p(name)
            if not path:
                continue
            if path not in self._maps:
                try:
                    self._maps[path] = TraversabilityMask.load(path)
                except (OSError, KeyError, ValueError) as e:
                    response.success = False
                    response.message = f'Cannot load {name} {path}: {e}'
                    self.get_logger().error(response.message)
                    return response
            masks[name] = self._maps[path]
        if 'traversability_map' in masks:
            area = MaskedArea(area, masks['traversability_map'], p('map_clearance'),
                              masks.get('spawn_map'))

        constraints = PlacementConstraints(
            min_separation=p('min_separation'),
            keep_out=(KeepOutCircle(p('keep_out_x'), p('keep_out_y'),
                                    p('keep_out_radius')),),
            max_attempts_per_probe=p('max_attempts_per_probe'),
        )
        with self.lock:
            existing = [(pose.position.x, pose.position.y)
                        for pose in self.spawned_probes.values()]

        positions = sample_positions(area, request.count, existing, constraints, self.rng)

        upright = UprightParams(p('upright_fraction'), p('upright_bury_min'),
                                p('upright_bury_max'), p('upright_max_tilt_deg'))
        ground = foot = None
        if upright.fraction > 0 and p('heightmap'):
            ground = self._load(p('heightmap'), HeightGrid.load)
            if p('upright_keep_out'):
                foot = self._load(p('upright_keep_out'), Footprints.load)
        errors = []
        for x, y in positions:
            pose = sample_upright(self.rng, x, y, ground, upright, foot) if ground else None
            name, error = self.spawn_probe(x, y, p('spawn_z'), pose)
            if name:
                response.names.append(name)
            else:
                errors.append(error)

        response.success = len(response.names) == request.count
        response.message = f'Spawned {len(response.names)}/{request.count} probes'
        if len(positions) < request.count:
            response.message += (f'; only {len(positions)} positions fit the area '
                                 f'with min_separation={constraints.min_separation}')
        if errors:
            response.message += f'; {len(errors)} spawn errors: {errors[0]}'
        # rclpy fixes the severity per call site, so one call site per severity
        if response.success:
            self.get_logger().info(response.message)
        else:
            self.get_logger().warn(response.message)
        return response

    def _load(self, path, loader):
        if path not in self._maps:
            self._maps[path] = loader(path)
        return self._maps[path]

    def spawn_probe(self, x, y, z, upright=None):
        """
        Spawn one probe in Gazebo and bridge its pose. Returns (name, None) or (None, error).

        upright = (z, roll, pitch, yaw): a static probe stuck into the ground in that pose;
        None - dropped from `z` with a random orientation (it falls and lies).
        """
        if upright is None:
            q = sample_orientation(self.rng)
            roll, pitch, yaw = q.to_rpy()
        else:
            z, roll, pitch, yaw = upright
            cr, sr = math.cos(roll / 2), math.sin(roll / 2)
            cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
            cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
            q = Orientation(sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)

        try:
            pkg_dir = get_package_share_directory('probe_supervisor')
            sdf_path = os.path.join(pkg_dir, 'objects', 'probe', 'probe.sdf')
        except Exception as e:
            self.get_logger().error(f'Pkg error: {e}')
            return None, f'Pkg error: {e}'

        if not os.path.exists(sdf_path):
            self.get_logger().error('SDF not found.')
            return None, 'SDF not found.'

        self.counter += 1
        name = f'probe_object_{self.counter}'

        source = ['-file', sdf_path]
        if upright is not None:     # static: a dynamic body inside the terrain is thrown out
            with open(sdf_path) as f:
                sdf = f.read().replace('<model name="probe">',
                                       '<model name="probe">\n    <static>true</static>', 1)
            source = ['-string', sdf]
        cmd_spawn = [
            'ros2', 'run', 'ros_gz_sim', 'create',
            *source,
            '-name', name,
            '-x', f'{x}',
            '-y', f'{y}',
            '-z', f'{z}',
            '-R', f'{roll}',
            '-P', f'{pitch}',
            '-Y', f'{yaw}'
        ]

        cmd_bridge = [
            'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
            f'/model/{name}/pose@geometry_msgs/msg/Pose[gz.msgs.Pose'
        ]

        try:
            res = subprocess.run(cmd_spawn, capture_output=True, text=True, timeout=5.0)
        except Exception as e:
            self.get_logger().error(f'Exception: {e}')
            return None, f'Exception: {e}'

        if res.returncode != 0:
            self.get_logger().error(f'Spawn error: {res.stderr}')
            return None, f'Spawn error: {res.stderr}'

        track = self.get_parameter('track_poses').value
        if track:
            self.bridge_processes[name] = subprocess.Popen(cmd_bridge)

        initial_pose = Pose(
            position=Point(x=float(x), y=float(y), z=float(z)),
            orientation=Quaternion(x=q.x, y=q.y, z=q.z, w=q.w)
        )
        with self.lock:
            self.spawned_probes[name] = initial_pose

        if track:
            self.subscribers[name] = self.create_subscription(
                Pose,
                f'/model/{name}/pose',
                lambda msg, n=name: self.pose_callback(msg, n),
                10
            )

        self.get_logger().info(
            f'Spawned {name} at ({x:.2f}, {y:.2f}) rpy ({roll:.2f}, {pitch:.2f}, {yaw:.2f})'
            + (f' upright (static, z={z:.3f})' if upright is not None else '')
            + (' with bridge' if track else ''))
        return name, None

    def stop_bridges(self):
        """Stop the per-probe pose bridges (separate processes that outlive the node)."""
        for proc in self.bridge_processes.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in self.bridge_processes.values():
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()

    def pose_callback(self, msg, name):
        with self.lock:
            if name in self.spawned_probes:
                self.spawned_probes[name] = msg

    def update_callback(self):
        with self.lock:
            probes = list(self.spawned_probes.items())
        if not probes:
            return

        pose_array = PoseArray()
        pose_array.header.frame_id = 'map'
        pose_array.header.stamp = self.get_clock().now().to_msg()

        marker_array = MarkerArray()

        for idx, (name, pose) in enumerate(probes):
            pose_array.poses.append(pose)

            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'probes'
            marker.id = idx
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose = pose
            marker.scale.x = 0.03
            marker.scale.y = 0.03
            marker.scale.z = 0.2
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.8
            marker_array.markers.append(marker)

        self.pose_pub.publish(pose_array)
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = ProbeSpawnerNode()
    # Two threads are enough (spawn service + marker timer); the default (one per CPU) kept
    # ~50% of a core busy while idle, which the simulation needs (real-time factor).
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop_bridges()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
