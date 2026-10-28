"""
ig_search: detektory próbek (ramka map) + węzeł przeszukiwania (lifecycle, autostart).

Zakłada działający stos robota (sirius_gazebo/sirius_sim: Nav2, MoveIt przy use_wrist).

  ros2 launch ig_search ig_search.launch.py
  ros2 launch ig_search ig_search.launch.py strategy:=meander use_wrist:=false
  ros2 launch ig_search ig_search.launch.py strategy:=ig_horizon horizon_depth:=5
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription,
                            RegisterEventHandler)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import LifecycleNode
from launch_ros.descriptions import ParameterValue
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition


def _transition(node, transition_id):
    return EmitEvent(event=ChangeState(lifecycle_node_matcher=lambda a: a is node,
                                       transition_id=transition_id))


def generate_launch_description():
    share = get_package_share_directory('ig_search')
    sim_time = PythonExpression(["'", LaunchConfiguration('use_sim_time'), "' == 'true'"])
    use_wrist = PythonExpression(["'", LaunchConfiguration('use_wrist'), "' == 'true'"])
    cameras = PythonExpression(["'front,wrist' if '", LaunchConfiguration('use_wrist'),
                                "' == 'true' else 'front'"])
    detectors = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('probe_detector'), 'launch', 'probe_detectors.launch.py')),
        launch_arguments={'config': LaunchConfiguration('detector_config'),
                          'use_sim_time': LaunchConfiguration('use_sim_time'),
                          'cameras': cameras}.items(),
        condition=IfCondition(LaunchConfiguration('detectors')))
    node = LifecycleNode(
        package='ig_search', executable='search_node', name='ig_search', namespace='',
        parameters=[LaunchConfiguration('params_file'),
                    {'use_sim_time': sim_time,
                     'planner.strategy': LaunchConfiguration('strategy'),
                     'planner.horizon_depth': ParameterValue(
                         LaunchConfiguration('horizon_depth'), value_type=int),
                     'mission.use_wrist': use_wrist,
                     'traversability_map': LaunchConfiguration('traversability_map'),
                     'probe_object': PythonExpression(
                         ["'", LaunchConfiguration('probe_object'), "' == 'true'"])}],
        output='screen')
    configure = _transition(node, Transition.TRANSITION_CONFIGURE)
    activate = RegisterEventHandler(OnStateTransition(
        target_lifecycle_node=node, goal_state='inactive',
        entities=[_transition(node, Transition.TRANSITION_ACTIVATE)]))
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('params_file',
                              default_value=os.path.join(share, 'config', 'ig_search.yaml')),
        DeclareLaunchArgument('strategy', default_value='ig_greedy',
                              choices=['ig_greedy', 'ig_horizon', 'meander', 'frontier', 'chase']),
        DeclareLaunchArgument('horizon_depth', default_value='3',
                              description='ig_horizon: liczba obserwacji w planowanej '
                                          'sekwencji (D)'),
        DeclareLaunchArgument('use_wrist', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('traversability_map',
                              default_value='maps/erc2025_traversability.yaml',
                              description='mapa przejezdności (względem share/ig_search/config)'),
        DeclareLaunchArgument('probe_object', default_value='true', choices=['true', 'false'],
                              description='Próbka jako przeszkoda w scenie MoveIt przed chwytem'),
        DeclareLaunchArgument('detectors', default_value='true', choices=['true', 'false'],
                              description='Uruchom probe_detector (front, + wrist)'),
        DeclareLaunchArgument('detector_config',
                              default_value=os.path.join(share, 'config',
                                                         'probe_detectors.yaml'),
                              description='Parametry probe_detector (np. inne wagi modeli)'),
        detectors, node, activate, configure,
    ])
