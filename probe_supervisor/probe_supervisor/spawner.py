import os
import random
import subprocess
import threading
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger
from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from ament_index_python.packages import get_package_share_directory

from probe_supervisor_interfaces.srv import SpawnRandomProbes
from probe_supervisor.domain.spawn_area import RectangleArea
from probe_supervisor.domain.orientation import sample_orientation
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

        errors = []
        for x, y in positions:
            name, error = self.spawn_probe(x, y, p('spawn_z'))
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
        log = self.get_logger().info if response.success else self.get_logger().warn
        log(response.message)
        return response

    def spawn_probe(self, x, y, z):
        """Spawn one probe in Gazebo and bridge its pose. Returns (name, None) or (None, error)."""
        # Random 3D orientation: the probe falls and lies in a random direction
        q = sample_orientation(self.rng)
        roll, pitch, yaw = q.to_rpy()

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

        cmd_spawn = [
            'ros2', 'run', 'ros_gz_sim', 'create',
            '-file', sdf_path,
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

        self.bridge_processes[name] = subprocess.Popen(cmd_bridge)

        initial_pose = Pose(
            position=Point(x=float(x), y=float(y), z=float(z)),
            orientation=Quaternion(x=q.x, y=q.y, z=q.z, w=q.w)
        )
        with self.lock:
            self.spawned_probes[name] = initial_pose

        self.subscribers[name] = self.create_subscription(
            Pose,
            f'/model/{name}/pose',
            lambda msg, n=name: self.pose_callback(msg, n),
            10
        )

        self.get_logger().info(
            f'Spawned {name} at ({x:.2f}, {y:.2f}) rpy ({roll:.2f}, {pitch:.2f}, {yaw:.2f}) '
            'with bridge')
        return name, None

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
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
