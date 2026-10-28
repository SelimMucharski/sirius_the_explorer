"""
Test samego chwytu: Gazebo (headless, ramię 6DOF) + MoveIt + grasp_test (zob. grasp_test.py).

Kończy się po teście; wynik w `output`.

  ros2 launch ig_search_experiments grasp_test.launch.py output:=<katalog> [grasp_depth:=0.012]
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
    L = LaunchConfiguration
    gazebo = _include('sirius_gazebo', 'sirius_gazebo.launch.py', {
        'manipulator': '6DOF', 'headless': L('headless')})
    moveit = _include('sirius_moveit_config', 'moveit.launch.py', {
        'use_sim_time': 'true', 'use_rviz': 'false', 'use_servo': 'false'})
    test = Node(package='ig_search_experiments', executable='grasp_test', output='screen',
                parameters=[{'use_sim_time': True, 'output': L('output'),
                             'world': L('world_name'),
                             'grasp_depth': L('grasp_depth'),
                             'carry_scaling': L('carry_scaling'),
                             'step_time': L('step_time'), 'offset': L('offset'),
                             'upright': L('upright')}])
    return LaunchDescription([
        DeclareLaunchArgument('output'),
        DeclareLaunchArgument('headless', default_value='true'),
        # sirius_gazebo domyślnie: worlds/empty.sdf (świat `empty`, płaski grunt)
        DeclareLaunchArgument('world_name', default_value='empty'),
        DeclareLaunchArgument('grasp_depth', default_value='0.012'),
        DeclareLaunchArgument('carry_scaling', default_value='0.2'),
        DeclareLaunchArgument('step_time', default_value='1.2'),
        DeclareLaunchArgument('offset', default_value='0.0'),
        DeclareLaunchArgument('upright', default_value='false',
                              description='próbka wbita pionowo (statyczna)'),
        gazebo, moveit,
        TimerAction(period=20.0, actions=[test]),
        RegisterEventHandler(OnProcessExit(target_action=test,
                                           on_exit=[EmitEvent(event=Shutdown())])),
    ])
