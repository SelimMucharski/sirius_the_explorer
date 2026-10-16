"""
Probe detectors for both cameras as lifecycle nodes, configured and activated on start.

  /front/probe_detector  OAK-D W  /front/color/image_raw + /front/color/camera_info
  /wrist/probe_detector  OAK-D SR /wrist/left/image_raw  + /wrist/left/camera_info
  -> /<camera>/probe_detector/{poses, detections, markers[, debug_image]}

  ros2 launch probe_detector probe_detectors.launch.py use_sim_time:=true debug_image:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition

CAMERAS = {
    'front': ('/front/color/image_raw', '/front/color/camera_info'),
    'wrist': ('/wrist/left/image_raw', '/wrist/left/camera_info'),
}


def _transition(node, transition_id):
    return EmitEvent(event=ChangeState(lifecycle_node_matcher=lambda a: a is node,
                                       transition_id=transition_id))


def _autostart(node):
    activate = RegisterEventHandler(OnStateTransition(
        target_lifecycle_node=node, goal_state='inactive',
        entities=[_transition(node, Transition.TRANSITION_ACTIVATE)]))
    return [activate, _transition(node, Transition.TRANSITION_CONFIGURE)]


def generate_launch_description():
    share = get_package_share_directory('probe_detector')
    sim_time = PythonExpression(["'", LaunchConfiguration('use_sim_time'), "' == 'true'"])
    debug = PythonExpression(["'", LaunchConfiguration('debug_image'), "' == 'true'"])
    actions = [
        DeclareLaunchArgument(
            'config', default_value=os.path.join(share, 'config', 'probe_detectors.yaml')),
        DeclareLaunchArgument('use_sim_time', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('debug_image', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('cameras', default_value='front,wrist',
                              description='Comma-separated subset of: front, wrist'),
    ]
    for cam, (image, info) in CAMERAS.items():
        enabled = PythonExpression(
            ["'", cam, "' in '", LaunchConfiguration('cameras'), "'.split(',')"])
        node = LifecycleNode(
            package='probe_detector', executable='probe_detector', name='probe_detector',
            namespace=cam,
            parameters=[LaunchConfiguration('config'),
                        {'use_sim_time': sim_time, 'publish_debug_image': debug}],
            remappings=[('image', image), ('camera_info', info)],
            condition=IfCondition(enabled),
            output='screen')
        actions += [node] + _autostart(node)
    return LaunchDescription(actions)
