"""
Test kroku fizyki: Gazebo (podany świat, ramię 6DOF, headless) + spawner + sekwencja jazdy.

Kończy się po teście; wynik w `output` (zob. physics_step_test.py).

  ros2 launch ig_search_experiments physics_test.launch.py world:=<sdf> output:=<katalog>
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription,
                            RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    L = LaunchConfiguration
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('sirius_gazebo'), 'launch', 'sirius_gazebo.launch.py')),
        launch_arguments={'world': L('world'), 'manipulator': '6DOF',
                          'headless': 'true'}.items())
    spawner = Node(package='probe_supervisor', executable='spawner', output='screen',
                   parameters=[{'use_sim_time': True, 'random_seed': 7,
                                'area_min_x': 2.0, 'area_max_x': 9.0, 'area_min_y': -7.0,
                                'area_max_y': 3.0, 'spawn_z': 1.5, 'min_separation': 1.0,
                                'keep_out_radius': 1.5}])
    test = Node(package='ig_search_experiments', executable='physics_step_test',
                output='screen', parameters=[{'use_sim_time': True, 'output': L('output')}])
    return LaunchDescription([
        DeclareLaunchArgument('world'),
        DeclareLaunchArgument('output'),
        gazebo, spawner, test,
        RegisterEventHandler(OnProcessExit(target_action=test,
                                           on_exit=[EmitEvent(event=Shutdown())])),
    ])
