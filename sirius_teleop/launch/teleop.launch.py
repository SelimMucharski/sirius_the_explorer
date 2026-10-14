"""Gamepad teleoperation: joy_node + joy_teleop.

Usage:
  ros2 launch sirius_teleop teleop.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(get_package_share_directory('sirius_teleop'), 'config', 'joy_teleop.yaml')
    use_sim_time = {'use_sim_time': LaunchConfiguration('use_sim_time')}
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('device_id', default_value='0', description='Gamepad index'),
        Node(package='joy', executable='joy_node', name='joy_node',
             parameters=[config, use_sim_time, {'device_id': LaunchConfiguration('device_id')}],
             output='screen'),
        Node(package='sirius_teleop', executable='joy_teleop.py', name='joy_teleop',
             parameters=[config, use_sim_time], output='screen'),
    ])
