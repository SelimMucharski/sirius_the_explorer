"""
Launch ros2_control for the Unity simulation (unity_sim).

Like sirius_control.launch.py (real hardware) but with unity_sim:=true: the hardware
plugin is topic_based_ros2_control, which exchanges sensor_msgs/JointState with Unity
over rosbridge (/sirius/wheel_*, /sirius/manip_*).

Topics: /cmd_vel (TwistStamped) -> sirius_base_controller -> /odometry/wheels.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def spawner(name, condition=None):
    return Node(
        package='controller_manager',
        executable='spawner',
        arguments=[name, '--controller-manager', '/controller_manager'],
        condition=condition,
    )


def generate_launch_description():
    manipulator = LaunchConfiguration('manipulator')
    controllers_yaml = LaunchConfiguration('controllers_file')

    description_pkg = get_package_share_directory('sirius_description')
    xacro_path = os.path.join(description_pkg, 'robots', 'sirius.urdf.xacro')
    default_controllers = os.path.join(
        get_package_share_directory('sirius_control'), 'config', 'controllers_unity.yaml')

    robot_description = ParameterValue(
        Command(['xacro ', xacro_path, ' manipulator:=', manipulator, ' unity_sim:=true']),
        value_type=str,
    )

    return LaunchDescription([
        DeclareLaunchArgument('manipulator', default_value='none',
                              choices=['none', '5DOF', '6DOF'],
                              description='Which manipulator variant is simulated'),
        DeclareLaunchArgument('controllers_file', default_value=default_controllers),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
            output='screen',
        ),
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            parameters=[{'robot_description': robot_description}, controllers_yaml],
            remappings=[
                ('/sirius_base_controller/cmd_vel', '/cmd_vel'),
                ('/sirius_base_controller/odom', '/odometry/wheels'),
            ],
            output='screen',
        ),
        spawner('joint_state_broadcaster'),
        spawner('sirius_base_controller'),
        spawner('manip_controller',
                IfCondition(PythonExpression(["'", manipulator, "' != 'none'"]))),
    ])
