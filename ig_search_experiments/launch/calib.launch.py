"""
M6: zbieranie danych do kalibracji detektora (calib_collect.py) - Gazebo headless, erc2025.

Faza front: łazik teleportowany wokół 10 próbek; faza wrist: skany ramieniem nad 3 próbkami
(z MoveIt). Kończy się po zebraniu danych; wynik w `output`.

  ros2 launch ig_search_experiments calib.launch.py phase:=front output:=<katalog>
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription,
                            OpaqueFunction, RegisterEventHandler, TimerAction)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _include(package, launch_file, arguments):
    path = os.path.join(get_package_share_directory(package), 'launch', launch_file)
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(path),
                                    launch_arguments=arguments.items())


def _setup(context):
    phase = LaunchConfiguration('phase').perform(context)
    wrist = phase == 'wrist'
    sim = _include('sirius_gazebo', 'sirius_sim.launch.py', {
        'manipulator': '6DOF', 'nav2': 'false', 'moveit': 'true' if wrist else 'false',
        'rviz': 'false', 'servo': 'false', 'headless': 'true', 'localization': 'none'})
    area = ({'area_min_x': 2.0, 'area_max_x': 6.0, 'area_min_y': -3.0, 'area_max_y': 3.0}
            if wrist else
            {'area_min_x': 1.0, 'area_max_x': 12.0, 'area_min_y': -7.0, 'area_max_y': 6.0})
    spawner = Node(package='probe_supervisor', executable='spawner', output='screen',
                   parameters=[{'use_sim_time': True, 'random_seed': 11, 'spawn_z': 1.5,
                                'min_separation': 1.5, 'keep_out_radius': 2.0, **area}])
    detectors = _include('probe_detector', 'probe_detectors.launch.py', {
        'config': os.path.join(get_package_share_directory('ig_search'), 'config',
                               'probe_detectors.yaml'),
        'use_sim_time': 'true', 'cameras': phase})
    collect = Node(package='ig_search_experiments', executable='calib_collect', output='screen',
                   parameters=[{'use_sim_time': True, 'phase': phase,
                                'output': LaunchConfiguration('output'),
                                'n_poses': int(LaunchConfiguration('n_poses').perform(context))}])
    return [sim, spawner, detectors, TimerAction(period=25.0, actions=[collect]),
            RegisterEventHandler(OnProcessExit(target_action=collect,
                                               on_exit=[EmitEvent(event=Shutdown())]))]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('phase', default_value='front', choices=['front', 'wrist']),
        DeclareLaunchArgument('output'),
        DeclareLaunchArgument('n_poses', default_value='60'),
        OpaqueFunction(function=_setup),
    ])
