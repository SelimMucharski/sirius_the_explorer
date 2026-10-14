"""
Gazebo Harmonic simulation of the Sirius rover with ros2_control.

Starts gz sim with a world, robot_state_publisher (sim time), spawns the robot,
bridges the sensors/clock to ROS and, once the robot exists, spawns the controllers
(the controller_manager itself runs inside Gazebo via gz_ros2_control).

Topics: /cmd_vel (TwistStamped) -> sirius_base_controller -> /odometry/wheels,
        /manip_controller/follow_joint_trajectory (when a manipulator is mounted).

Usage:
  ros2 launch sirius_gazebo sirius_gazebo.launch.py
  ros2 launch sirius_gazebo sirius_gazebo.launch.py manipulator:=6DOF headless:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription, RegisterEventHandler)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _spawner(controller, *extra_args, condition=None):
    # Activation goes through the controller_manager update loop; with Nav2 and
    # MoveIt starting at the same time it can exceed the default 5 s switch timeout.
    return Node(
        package='controller_manager',
        executable='spawner',
        arguments=[controller,
                   '--controller-manager', '/controller_manager',
                   '--controller-manager-timeout', '60',
                   '--switch-timeout', '30',
                   *extra_args],
        output='screen',
        condition=condition,
    )


def generate_launch_description():
    gazebo_pkg = get_package_share_directory('sirius_gazebo')
    description_pkg = get_package_share_directory('sirius_description')

    manipulator = LaunchConfiguration('manipulator')
    world = LaunchConfiguration('world')
    headless = LaunchConfiguration('headless')

    controllers_yaml = os.path.join(gazebo_pkg, 'config', 'controllers.yaml')
    robot_description = ParameterValue(
        Command([
            'xacro ', os.path.join(description_pkg, 'robots', 'sirius.urdf.xacro'),
            ' manipulator:=', manipulator,
            ' gazebo_controllers:=', controllers_yaml,
        ]),
        value_type=str,
    )

    # package://sirius_description/... meshes are resolved against GZ_SIM_RESOURCE_PATH
    resource_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH', os.path.dirname(description_pkg))

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': [
                '-r -v 2 ',
                PythonExpression(["'-s --headless-rendering ' if '", headless,
                                  "' == 'true' else ''"]),
                world,
            ],
            'on_exit_shutdown': 'true',
        }.items(),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
        output='screen',
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'sirius',
            '-x', LaunchConfiguration('x'),
            '-y', LaunchConfiguration('y'),
            '-z', LaunchConfiguration('z'),
            '-Y', LaunchConfiguration('yaw'),
        ],
        output='screen',
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{
            'config_file': os.path.join(gazebo_pkg, 'config', 'gz_bridge.yaml'),
            'use_sim_time': True,
        }],
        output='screen',
    )

    spawn_joint_state_broadcaster = _spawner('joint_state_broadcaster')

    # Nav2 publishes /cmd_vel; the EKF consumes /odometry/wheels
    spawn_base_controller = _spawner(
        'sirius_base_controller',
        '--controller-ros-args', '-r ~/cmd_vel:=/cmd_vel -r ~/odom:=/odometry/wheels')

    with_arm = IfCondition(PythonExpression(["'", manipulator, "' != 'none'"]))
    spawn_manip_controller = _spawner('manip_controller', condition=with_arm)
    spawn_gripper_controller = _spawner('gripper_controller', condition=with_arm)

    # gz_ros2_control creates the controller_manager when the robot is spawned
    spawn_controllers_after_robot = RegisterEventHandler(OnProcessExit(
        target_action=spawn_robot,
        on_exit=[spawn_joint_state_broadcaster, spawn_base_controller, spawn_manip_controller,
                 spawn_gripper_controller],
    ))

    return LaunchDescription([
        DeclareLaunchArgument('manipulator', default_value='none',
                              choices=['none', '6DOF'],
                              description='Which manipulator variant is simulated'),
        DeclareLaunchArgument('world',
                              default_value=os.path.join(gazebo_pkg, 'worlds', 'empty.sdf'),
                              description='Path to the SDF world'),
        DeclareLaunchArgument('headless', default_value='false', choices=['true', 'false'],
                              description='Run the Gazebo server without the GUI'),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('z', default_value='0.05'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        resource_path,
        gz_sim,
        robot_state_publisher,
        spawn_robot,
        bridge,
        spawn_controllers_after_robot,
    ])
