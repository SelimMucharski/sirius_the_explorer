"""Localization for the Unity simulation: odom EKF (wheels + IMU); map = odom or GPS-fused."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params = get_package_share_directory('sirius_navigation') + '/config/ekf_unity.yaml'
    use_gps = LaunchConfiguration('use_gps')
    return LaunchDescription([
        DeclareLaunchArgument('use_gps', default_value='false',
                              description='Fuse GPS into the map frame (navsat_transform + map EKF)'),
        Node(package='robot_localization', executable='ekf_node', name='ekf_odom',
             parameters=[params], remappings=[('odometry/filtered', '/odometry/filtered')]),
        # Same as ROS 1 localization.launch: map == odom when no global source is used.
        Node(package='tf2_ros', executable='static_transform_publisher', name='map_to_odom',
             arguments=['--frame-id', 'map', '--child-frame-id', 'odom'],
             condition=UnlessCondition(use_gps)),
        Node(package='robot_localization', executable='ekf_node', name='ekf_map',
             parameters=[params], remappings=[('odometry/filtered', '/odometry/global')],
             condition=IfCondition(use_gps)),
        Node(package='robot_localization', executable='navsat_transform_node', name='navsat_transform',
             parameters=[params],
             remappings=[('imu', '/imu/data'), ('gps/fix', '/gps/fix'),
                         ('odometry/filtered', '/odometry/global'), ('odometry/gps', '/odometry/gps')],
             condition=IfCondition(use_gps)),
    ])
