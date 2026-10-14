"""
Full Sirius simulation: Gazebo + ros2_control + localization + Nav2 + MoveIt.

  sirius_gazebo.launch.py      gz sim, robot, bridge, controllers
  ekf_odom (robot_localization) /odometry/wheels + /imu/data -> /odometry/filtered, odom->base_link
  static map->odom             no global localization source in the empty world
  sirius_navigation/nav2       obstacle_cloud (/front/depth) + Nav2 (-> /cmd_vel)
  sirius_moveit_config/moveit  move_group + MoveIt Servo + RViz (only with the arm)
  sirius_teleop                gamepad: drive, arm jogging, gripper (teleop:=true)

Usage:
  ros2 launch sirius_gazebo sirius_sim.launch.py                    # base + Nav2
  ros2 launch sirius_gazebo sirius_sim.launch.py manipulator:=6DOF  # + MoveIt
  ros2 launch sirius_gazebo sirius_sim.launch.py manipulator:=6DOF nav2:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def _include(package, launch_file, arguments, condition=None):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory(package), 'launch', launch_file)),
        launch_arguments=arguments.items(),
        condition=condition,
    )


def generate_launch_description():
    gazebo_pkg = get_package_share_directory('sirius_gazebo')

    manipulator = LaunchConfiguration('manipulator')
    nav2 = LaunchConfiguration('nav2')
    rviz = LaunchConfiguration('rviz')

    with_moveit = IfCondition(PythonExpression([
        "'", LaunchConfiguration('moveit'), "' == 'true' and '", manipulator, "' != 'none'"]))

    gazebo = _include('sirius_gazebo', 'sirius_gazebo.launch.py', {
        'manipulator': manipulator,
        'world': LaunchConfiguration('world'),
        'headless': LaunchConfiguration('headless'),
    })

    ekf_odom = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_odom',
        parameters=[os.path.join(gazebo_pkg, 'config', 'ekf.yaml'), {'use_sim_time': True}],
        remappings=[('odometry/filtered', '/odometry/filtered')],
        output='screen',
    )

    map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom',
        arguments=['--frame-id', 'map', '--child-frame-id', 'odom'],
        parameters=[{'use_sim_time': True}],
    )

    navigation = _include('sirius_navigation', 'nav2.launch.py', {
        'use_sim_time': 'true',
    }, condition=IfCondition(nav2))

    nav2_rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2_nav2',
        arguments=['-d', os.path.join(get_package_share_directory('nav2_bringup'),
                                      'rviz', 'nav2_default_view.rviz')],
        parameters=[{'use_sim_time': True}],
        output='screen',
        condition=IfCondition(PythonExpression(["'", nav2, "' == 'true' and '", rviz,
                                                "' == 'true'"])),
    )

    moveit = _include('sirius_moveit_config', 'moveit.launch.py', {
        'use_sim_time': 'true',
        'use_rviz': rviz,
    }, condition=with_moveit)

    teleop = _include('sirius_teleop', 'teleop.launch.py', {
        'use_sim_time': 'true',
    }, condition=IfCondition(LaunchConfiguration('teleop')))

    return LaunchDescription([
        DeclareLaunchArgument('manipulator', default_value='none',
                              choices=['none', '6DOF'],
                              description='Which manipulator variant is simulated'),
        DeclareLaunchArgument('world',
                              default_value=os.path.join(gazebo_pkg, 'worlds', 'empty.sdf'),
                              description='Path to the SDF world'),
        DeclareLaunchArgument('headless', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('nav2', default_value='true', choices=['true', 'false'],
                              description='Start Nav2'),
        DeclareLaunchArgument('moveit', default_value='true', choices=['true', 'false'],
                              description='Start MoveIt (ignored when manipulator:=none)'),
        DeclareLaunchArgument('rviz', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('teleop', default_value='false', choices=['true', 'false'],
                              description='Start gamepad teleoperation (sirius_teleop)'),
        gazebo,
        ekf_odom,
        map_to_odom,
        navigation,
        nav2_rviz,
        moveit,
        teleop,
    ])
