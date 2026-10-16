"""
Full Sirius simulation: Gazebo + ros2_control + localization + Nav2 + MoveIt.

  sirius_gazebo.launch.py      gz sim, robot, bridge, controllers
  ekf_odom (robot_localization) /odometry/wheels + /imu/data -> /odometry/filtered, odom->base_link
  localization:=gnss (default) sirius_navigation/gnss_odom_correction: map->odom from /gps/fix
                               (corrects the odometry position drift, bounded error)
  localization:=none           static map->odom (the odometry drift is not corrected)
  sirius_navigation/nav2       obstacle_cloud (/front/depth) + Nav2 (-> /cmd_vel)
  sirius_moveit_config/moveit  move_group + MoveIt Servo + RViz (only with the arm)
  sirius_teleop                gamepad: drive, arm jogging, gripper (teleop:=true)

Usage:
  ros2 launch sirius_gazebo sirius_sim.launch.py                    # base + Nav2
  ros2 launch sirius_gazebo sirius_sim.launch.py manipulator:=6DOF  # + MoveIt
  ros2 launch sirius_gazebo sirius_sim.launch.py manipulator:=6DOF nav2:=false
"""

import math
import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
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


def _world_datum(world_file):
    """[latitude, longitude, heading (rad, ENU)] of a world's spherical_coordinates."""
    sc = ET.parse(world_file).getroot().find('world').find('spherical_coordinates')
    if sc is None:
        raise RuntimeError(
            f'{world_file}: no <spherical_coordinates> (needed by localization:=gnss)')
    value = lambda tag: float(sc.findtext(tag, '0'))  # noqa: E731
    return [value('latitude_deg'), value('longitude_deg'), math.radians(value('heading_deg'))]


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
        'yaw': LaunchConfiguration('yaw'),
    })

    ekf_odom = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_odom',
        parameters=[os.path.join(gazebo_pkg, 'config', 'ekf.yaml'), {'use_sim_time': True}],
        remappings=[('odometry/filtered', '/odometry/filtered')],
        output='screen',
    )

    no_gnss = IfCondition(PythonExpression(
        ["'", LaunchConfiguration('localization'), "' == 'none'"]))

    map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom',
        arguments=['--frame-id', 'map', '--child-frame-id', 'odom'],
        parameters=[{'use_sim_time': True}],
        condition=no_gnss,
    )

    # The rover slides on the erc2025 slopes when it turns in place (skid steering), which no
    # odometry sees; GNSS corrects the position drift in the map frame. The map frame is the
    # Gazebo world frame: the datum is the world's spherical_coordinates.
    def gnss_odom_correction(context):
        if LaunchConfiguration('localization').perform(context) != 'gnss':
            return []
        lat, lon, heading = _world_datum(LaunchConfiguration('world').perform(context))
        return [Node(
            package='sirius_navigation',
            executable='gnss_odom_correction.py',
            name='gnss_odom_correction',
            parameters=[{'use_sim_time': True, 'datum_latitude': lat, 'datum_longitude': lon,
                         'datum_heading': heading,
                         'odom_yaw': float(LaunchConfiguration('yaw').perform(context))}],
            output='screen',
        )]

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
        DeclareLaunchArgument('localization', default_value='gnss', choices=['gnss', 'none'],
                              description='map->odom corrected by GNSS (gnss) or static (none)'),
        DeclareLaunchArgument('yaw', default_value='0.0',
                              description='Rover heading at spawn in the world frame [rad]'),
        gazebo,
        ekf_odom,
        map_to_odom,
        OpaqueFunction(function=gnss_odom_correction),
        navigation,
        nav2_rviz,
        moveit,
        teleop,
    ])
