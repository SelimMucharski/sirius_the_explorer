"""
End-to-end test of the probe detectors in the full simulation on erc2025.

Gazebo (erc2025, 6DOF arm, no Nav2 / MoveIt) + probe_supervisor spawner + both detectors +
sim evaluator (ground truth comparison) + the test scenario (spawns probes, points the wrist
camera at the near ones, turns the rover). Shuts down when the scenario ends; the report is in
output_dir/summary.txt.

  ros2 launch probe_detector sim_probe_test.launch.py output_dir:=/tmp/probe_eval
  ros2 launch probe_detector sim_probe_test.launch.py debug_image:=true snapshot_period:=4.0
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription,
                            RegisterEventHandler, TimerAction)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _include(package, launch_file, arguments):
    path = os.path.join(get_package_share_directory(package), 'launch', launch_file)
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(path),
                                    launch_arguments=arguments.items())


def generate_launch_description():
    gazebo = get_package_share_directory('sirius_gazebo')
    sim = _include('sirius_gazebo', 'sirius_sim.launch.py', {
        'world': os.path.join(gazebo, 'worlds', 'erc2025.sdf'), 'manipulator': '6DOF',
        'nav2': 'false', 'moveit': 'false', 'rviz': 'false',
        'headless': LaunchConfiguration('headless')})
    detectors = _include('probe_detector', 'probe_detectors.launch.py', {
        'use_sim_time': 'true', 'debug_image': LaunchConfiguration('debug_image')})
    spawner = Node(package='probe_supervisor', executable='spawner',
                   parameters=[{'use_sim_time': True}], output='screen')
    evaluator = Node(package='probe_detector', executable='sim_evaluator', output='screen',
                     parameters=[{'use_sim_time': True,
                                  'output_dir': LaunchConfiguration('output_dir'),
                                  'snapshot_period': LaunchConfiguration('snapshot_period')}])
    scenario = Node(package='probe_detector', executable='sim_scenario', output='screen',
                    parameters=[{'use_sim_time': True}])
    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('debug_image', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument(
            'snapshot_period', default_value='0.0',
            description='Save debug image + ground truth every N sim seconds (needs debug_image)'),
        DeclareLaunchArgument('output_dir', default_value=os.path.expanduser('~/probe_sim_eval')),
        sim, spawner, evaluator,
        TimerAction(period=15.0, actions=[detectors]),       # after the controllers are active
        TimerAction(period=35.0, actions=[scenario]),        # models loaded
        RegisterEventHandler(OnProcessExit(target_action=scenario,
                                           on_exit=[EmitEvent(event=Shutdown())])),
    ])
