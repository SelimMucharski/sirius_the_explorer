"""
Jeden epizod przeszukiwania w symulacji.

Gazebo (erc2025, ramię 6DOF) + Nav2 + MoveIt + spawner próbek + losowe rozmieszczenie
M próbek + ig_search.

  ros2 launch ig_search sim_episode.launch.py seed:=1 strategy:=ig_greedy
  ros2 launch ig_search sim_episode.launch.py headless:=false rviz:=true   # demo z GUI
  ros2 launch ig_search sim_episode.launch.py bt_viewer:=true   # drzewo zachowań na bieżąco
  ros2 launch ig_search sim_episode.launch.py strategy:=ig_horizon horizon_depth:=5

Drzewo można też podejrzeć w osobnym terminalu: py-trees-tree-watcher (tekst)
albo py-trees-tree-viewer (przeglądarka).

Próbki losowane z zadanym seedem w prostokącie (area_*), tylko na mapie przejezdności
(`traversability_map`, co najmniej `map_clearance` od jej brzegu) i w miejscach, w których
nie znikają w podłożu (`spawn_map`); spadają i układają się na terenie. Mapa przejezdności
jest też warstwą statyczną globalnej mapy kosztów Nav2.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            SetLaunchConfiguration, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def _variant(prefix, suffix):
    """Ścieżka <prefix>[_real]<suffix> zależnie od argumentu world_variant."""
    variant = LaunchConfiguration('world_variant')
    return PythonExpression(["'", prefix, "' + ('_real' if '", variant,
                             "' == 'real' else '') + '", suffix, "'"])


def _include(package, launch_file, arguments):
    path = os.path.join(get_package_share_directory(package), 'launch', launch_file)
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(path),
                                    launch_arguments=arguments.items())


def generate_launch_description():
    gazebo = get_package_share_directory('sirius_gazebo')
    L = LaunchConfiguration
    sim = _include('sirius_gazebo', 'sirius_sim.launch.py', {
        'world': L('world'), 'manipulator': '6DOF',
        'nav2': 'false', 'moveit': 'true', 'rviz': 'false', 'servo': 'false',
        'octomap': L('octomap'),
        'headless': L('headless'),
        'localization': L('localization'),
        'clock_rate': L('clock_rate')})
    spawner = Node(package='probe_supervisor', executable='spawner', output='screen',
                   parameters=[{'use_sim_time': True, 'random_seed': L('seed'),
                                'area_min_x': L('area_min_x'), 'area_max_x': L('area_max_x'),
                                'area_min_y': L('area_min_y'), 'area_max_y': L('area_max_y'),
                                'spawn_z': 1.5, 'min_separation': 1.0,
                                'keep_out_radius': 2.0,
                                'traversability_map': L('traversability_map'),
                                'map_clearance': 0.5,
                                'spawn_map': L('spawn_map'),
                                # próbki wbite pionowo (statyczne, czubkiem w dół)
                                'upright_fraction': L('upright_fraction'),
                                'heightmap': os.path.join(
                                    get_package_share_directory('ig_search'), 'config',
                                    'maps', 'erc2025_heightmap.yaml'),
                                'upright_keep_out': L('upright_keep_out'),
                                # mosty pozycji próbek tylko do znaczników w RViz
                                'track_poses': PythonExpression(
                                    ["'", L('headless'), "' == 'false'"])}])
    spawn = ExecuteProcess(cmd=['ros2', 'service', 'call', '/spawn_random_probes',
                                'probe_supervisor_interfaces/srv/SpawnRandomProbes',
                                ['{count: ', L('count'), '}']], output='screen')
    # Nav2 bez serwerów, których misja nie używa (route, docking, waypoint follower)
    navigation = _include('ig_search', 'navigation.launch.py', {
        'use_sim_time': 'true', 'map': L('traversability_map'),
        'terrain_heightmap': L('terrain_heightmap'),
        'collision_horizon': L('collision_horizon')})
    gt_localization = Node(
        package='ig_search_experiments', executable='ground_truth_localization',
        parameters=[{'use_sim_time': True}], output='screen',
        condition=IfCondition(PythonExpression(
            ["'", L('localization'), "' == 'ground_truth'"])))
    rviz = Node(package='rviz2', executable='rviz2', name='rviz2_ig_search', output='log',
                arguments=['-d', os.path.join(get_package_share_directory('ig_search'),
                                              'config', 'ig_search.rviz')],
                parameters=[{'use_sim_time': True}],
                condition=IfCondition(L('ig_search_rviz')))
    watch = Node(package='ig_search_experiments', executable='probe_watch', output='screen',
                 parameters=[{'use_sim_time': True, 'output': L('watch_output'),
                              'seed': L('seed'), 'variant': L('variant')}],
                 condition=IfCondition(PythonExpression(["'", L('watch_output'), "' != ''"])))
    # zebranie próbki: ramię z chwyconą próbką w pozycji transportowej -> usunięcie z Gazebo
    store = Node(package='ig_search_experiments', executable='sample_store', output='screen',
                 parameters=[{'use_sim_time': True}])
    bt_viewer = ExecuteProcess(cmd=['py-trees-tree-viewer'], output='log',
                               condition=IfCondition(L('bt_viewer')))
    search = _include('ig_search', 'ig_search.launch.py', {
        'strategy': L('strategy'), 'use_wrist': L('use_wrist'),
        'horizon_depth': L('horizon_depth'),
        'traversability_map': L('traversability_map'),
        'detector_config': L('detector_config'),
        'probe_object': L('octomap')})   # wariant sceny MoveIt: octomap + próbka albo nic
    return LaunchDescription([
        DeclareLaunchArgument('seed', default_value='1'),
        DeclareLaunchArgument('world_variant', default_value='basic', choices=['basic', 'real'],
                              description='real - świat z kamieniami, płytami i żwirem '
                                          '(erc2025_real.sdf) z jego mapami przejezdności'),
        DeclareLaunchArgument('world', default_value=_variant(
            os.path.join(gazebo, 'worlds', 'erc2025'), '.sdf'),
                              description='Świat SDF (np. wariant z innym krokiem fizyki)'),
        DeclareLaunchArgument('count', default_value='3'),
        DeclareLaunchArgument('detector_config', default_value=os.path.join(
            get_package_share_directory('ig_search'), 'config', 'probe_detectors.yaml'),
                              description='Parametry probe_detector (np. inne wagi modeli)'),
        DeclareLaunchArgument('strategy', default_value='ig_greedy',
                              choices=['ig_greedy', 'ig_horizon', 'meander', 'frontier', 'chase']),
        DeclareLaunchArgument('clock_rate', default_value='100',
                              description='/clock ograniczony do tylu Hz (sirius_gazebo; 0 - '
                                          'każdy krok fizyki, ~400 Hz: każdy węzeł rclpy '
                                          'z czasem symulacji zużywał 25-60% rdzenia)'),
        DeclareLaunchArgument('variant', default_value='',
                              description='etykieta wariantu do summary.json (analiza serii)'),
        DeclareLaunchArgument('collision_horizon', default_value='1.5',
                              description='Nav2 RPP: czas sprawdzania kolizji naprzód [s]'),
        DeclareLaunchArgument('horizon_depth', default_value='3',
                              description='ig_horizon: liczba obserwacji w sekwencji (D)'),
        DeclareLaunchArgument('use_wrist', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('octomap', default_value='false', choices=['true', 'false'],
                              description='octomap w scenie MoveIt z kamery przedniej'),
        DeclareLaunchArgument('watch_output', default_value='',
                              description='katalog wyników obserwatora próbek ('' - wyłączony)'),
        DeclareLaunchArgument('headless', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('rviz', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('bt_viewer', default_value='false', choices=['true', 'false'],
                              description='podgląd drzewa zachowań misji (py_trees_ros_viewer)'),
        DeclareLaunchArgument('localization', default_value='ground_truth',
                              choices=['gnss', 'ground_truth', 'none'],
                              description='ground_truth = poza z Gazebo (D3, seria główna od '
                                          '2026-10-25); gnss = korekcja GNSS (wariant)'),
        DeclareLaunchArgument('traversability_map', default_value=_variant(os.path.join(
            get_package_share_directory('ig_search'), 'config', 'maps', 'erc2025'),
            '_traversability.yaml'),
            description='mapa przejezdności: warstwa statyczna Nav2 i miejsca próbek '
                        "('' - bez niej)"),
        DeclareLaunchArgument('terrain_heightmap', default_value=os.path.join(
            get_package_share_directory('ig_search'), 'config', 'maps', 'erc2025_heightmap.yaml'),
            description="mapa wysokości terenu dla obstacle_cloud; '' - tylko lokalny grunt"),
        DeclareLaunchArgument('spawn_map', default_value=_variant(os.path.join(
            get_package_share_directory('ig_search'), 'config', 'maps', 'erc2025'),
            '_spawn.yaml'),
            description='miejsca na próbki: bez miejsc, w których próbka znika w podłożu '
                        "(mesh kolizyjny niżej niż wizualny; '' - bez tego warunku)"),
        DeclareLaunchArgument('upright_fraction', default_value='0.0',
                              description='udział próbek wbitych pionowo w grunt (jak na ERC)'),
        DeclareLaunchArgument('upright_keep_out', default_value=os.path.join(
            gazebo, 'models', 'erc2025_dressing', 'footprints.yaml'),
            description='kamienie i płyty, w które nie wbija się pionowych próbek'),
        DeclareLaunchArgument('area_min_x', default_value='1.0'),
        DeclareLaunchArgument('area_max_x', default_value='20.0'),
        DeclareLaunchArgument('area_min_y', default_value='-10.0'),
        DeclareLaunchArgument('area_max_y', default_value='8.0'),
        # kopia `rviz` przed dołączeniem symulacji: jej argumenty (rviz:=false dla widoku Nav2)
        # przeciekają do tego launcha; zamknięty zakres (GroupAction) psuje za to obsługę
        # zdarzeń w sirius_gazebo (spawnery kontrolerów szukają `manipulator`)
        SetLaunchConfiguration('ig_search_rviz', L('rviz')),
        sim, navigation, spawner, gt_localization, rviz, watch, store,
        TimerAction(period=25.0, actions=[spawn]),           # kontrolery i Gazebo gotowe
        TimerAction(period=40.0, actions=[search]),          # próbki ułożone, Nav2 aktywny
        TimerAction(period=45.0, actions=[bt_viewer]),       # po starcie węzła misji
    ])
