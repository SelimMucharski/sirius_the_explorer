"""
MoveIt2 for the Sirius rover 6-DOF manipulator (simulation).

The robot itself (robot_state_publisher, ros2_control) is started by
sirius_gazebo; this file only adds the MoveIt side.

Usage:
  ros2 launch sirius_moveit_config moveit.launch.py use_sim_time:=true

Starts:
  - move_group (+ add_ground: the ground as a planning-scene obstacle; with octomap:=true
    also an octomap from the front depth camera, config/sensors_3d.yaml)
  - servo_node (MoveIt Servo: realtime jogging of the arm, e.g. from a gamepad)
  - RViz with the MotionPlanning plugin
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')
    use_servo = LaunchConfiguration('use_servo')

    description_pkg = get_package_share_directory('sirius_description')
    moveit_pkg = get_package_share_directory('sirius_moveit_config')

    robot_description = ParameterValue(
        Command(['xacro ', os.path.join(description_pkg, 'robots', 'sirius.urdf.xacro'),
                 ' manipulator:=6DOF']),
        value_type=str,
    )
    with open(os.path.join(moveit_pkg, 'config', 'sirius_6dof.srdf')) as f:
        robot_description_semantic = f.read()

    # MoveIt YAML configs have no ros__parameters wrapper, so they are passed as dicts.
    kinematics_cfg = _load_yaml(os.path.join(moveit_pkg, 'config', 'kinematics.yaml'))
    joint_limits_cfg = _load_yaml(os.path.join(moveit_pkg, 'config', 'joint_limits.yaml'))
    model_params = [
        {'robot_description': robot_description},
        {'robot_description_semantic': robot_description_semantic},
        {'robot_description_kinematics': kinematics_cfg},
        {'robot_description_planning': joint_limits_cfg},
        {'use_sim_time': use_sim_time},
    ]

    def move_group(context):
        if LaunchConfiguration('octomap').perform(context) == 'true':
            octomap = _load_yaml(os.path.join(moveit_pkg, 'config', 'sensors_3d.yaml'))
        else:
            # No 3D sensor for the octomap; an empty string avoids a runtime_error.
            octomap = {'sensors': [''], 'octomap_resolution': 0.0}
        return [Node(
            package='moveit_ros_move_group',
            executable='move_group',
            parameters=model_params + [
                {
                    'planning_pipelines': ['ompl'],
                    'default_planning_pipeline': 'ompl',
                    'ompl': _load_yaml(os.path.join(moveit_pkg, 'config', 'ompl_planning.yaml')),
                },
                _load_yaml(os.path.join(moveit_pkg, 'config', 'moveit_controllers.yaml')),
                octomap,
            ],
            output='screen',
        )]

    add_ground = Node(
        package='sirius_moveit_config',
        executable='add_ground.py',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    servo = Node(
        package='moveit_servo',
        executable='servo_node',  # node name: servo_node (renaming it also renames its internal nodes)
        parameters=model_params + [
            {'moveit_servo': _load_yaml(os.path.join(moveit_pkg, 'config', 'servo.yaml'))},
        ],
        output='screen',
        condition=IfCondition(use_servo),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(moveit_pkg, 'config', 'moveit.rviz')],
        parameters=[{'robot_description': robot_description,
                     'robot_description_semantic': robot_description_semantic,
                     'robot_description_kinematics': kinematics_cfg,
                     'use_sim_time': use_sim_time}],
        output='screen',
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Use the simulation clock'),
        DeclareLaunchArgument('use_rviz', default_value='true',
                              description='Start RViz with the MotionPlanning plugin'),
        DeclareLaunchArgument('use_servo', default_value='true',
                              description='Start MoveIt Servo (realtime arm jogging)'),
        DeclareLaunchArgument('octomap', default_value='false',
                              description='Octomap from the front depth camera '
                                          '(needs obstacle_cloud, e.g. from Nav2 bringup)'),
        OpaqueFunction(function=move_group),
        add_ground,
        servo,
        rviz,
    ])
