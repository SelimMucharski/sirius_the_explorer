"""
Nav2 dla epizodów ig_search: tylko serwery, których używa misja + chmura przeszkód.

Jak sirius_navigation/nav2.launch.py (te same parametry nav2_params.yaml, remapowania
i obstacle_cloud), ale bez route_server, docking_server i waypoint_follower - nie są
używane, a każdy zabiera ~10-15% CPU, którego brakuje symulacji (RTF).

Z `map` (mapa przejezdności w formacie nav2_map_server, jak kalman_nav2/maps) map_server
publikuje /map, a globalna mapa kosztów dostaje warstwę statyczną przed przeszkodami z kamery:
planer nie prowadzi przez teren nieprzejezdny (zbocza, uskoki, krawędź mapy), którego chmura
przeszkód nie oznacza (przeszkody ujemne).

  ros2 launch ig_search navigation.launch.py use_sim_time:=true
  ros2 launch ig_search navigation.launch.py map:=<mapa.yaml>

`linear_vel` / `angular_vel` nadpisują prędkość jazdy RPP (desired_linear_vel,
rotate_to_heading_angular_vel); granice kontrolera napędu i wygładzacza to 1,0 m/s i 1,0 rad/s
(sirius_gazebo/config/controllers.yaml). Planer ig_search ma tę samą prędkość w travel.v_max.

`terrain_heightmap` (mapa wysokości znana przed misją) daje obstacle_cloud drugi poziom
odniesienia: w usypisku najniższy punkt wokół wierzchu kamienia leży na sąsiednim kamieniu
(szczeliny są zasłonięte), więc sam lokalny grunt brał całe usypisko za teren (DESIGN §6).

`collision_horizon` nadpisuje max_allowed_time_to_collision_up_to_carrot RPP (czas jazdy, na
który kontroler sprawdza kolizje obrysu). Domyślnie 1,5 s jak w nav2_params: 0,9 s zmniejszało
liczbę „collision ahead” o ~25%, ale nie liczbę utknięć ani czasu misji (DESIGN §6).
"""
import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction, OpaqueFunction,
                            SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, SetParameter
from launch_ros.descriptions import ParameterFile
from nav2_common.launch import RewrittenYaml
import yaml

LIFECYCLE_NODES = ['controller_server', 'smoother_server', 'planner_server',
                   'behavior_server', 'velocity_smoother', 'collision_monitor', 'bt_navigator']
STATIC_LAYER = {'plugin': 'nav2_costmap_2d::StaticLayer', 'map_subscribe_transient_local': True}


def _with_static_layer(context):
    """
    Plik parametrów Nav2 z prędkością jazdy i horyzontem kolizji RPP z argumentów.

    Gdy podano mapę - także z warstwą statyczną w globalnej mapie kosztów.
    """
    src = LaunchConfiguration('nav2_params_file').perform(context)
    with open(src) as f:
        data = yaml.safe_load(f)
    follow = data['controller_server']['ros__parameters']['FollowPath']
    follow['desired_linear_vel'] = float(LaunchConfiguration('linear_vel').perform(context))
    follow['rotate_to_heading_angular_vel'] = float(
        LaunchConfiguration('angular_vel').perform(context))
    follow['max_allowed_time_to_collision_up_to_carrot'] = float(
        LaunchConfiguration('collision_horizon').perform(context))
    if LaunchConfiguration('map').perform(context):
        gc = data['global_costmap']['global_costmap']['ros__parameters']
        gc['plugins'] = ['static_layer'] + [p for p in gc['plugins'] if p != 'static_layer']
        gc['static_layer'] = dict(STATIC_LAYER)
    out = tempfile.NamedTemporaryFile('w', prefix='nav2_params_', suffix='.yaml', delete=False)
    yaml.safe_dump(data, out)
    out.close()
    context.launch_configurations['nav2_params_effective'] = out.name
    return []


def generate_launch_description():
    nav_share = get_package_share_directory('sirius_navigation')
    use_sim_time = LaunchConfiguration('use_sim_time')
    params = ParameterFile(RewrittenYaml(source_file=LaunchConfiguration('nav2_params_effective'),
                                         root_key='', param_rewrites={'autostart': 'true'},
                                         convert_types=True), allow_substs=True)
    remap = [('/tf', 'tf'), ('/tf_static', 'tf_static')]

    def server(package, executable, extra_remap=()):
        return Node(package=package, executable=executable, name=executable, output='screen',
                    parameters=[params], remappings=remap + list(extra_remap))

    nodes = GroupAction([
        SetParameter('use_sim_time', use_sim_time),
        server('nav2_controller', 'controller_server', [('cmd_vel', 'cmd_vel_nav')]),
        server('nav2_smoother', 'smoother_server'),
        server('nav2_planner', 'planner_server'),
        server('nav2_behaviors', 'behavior_server', [('cmd_vel', 'cmd_vel_nav')]),
        server('nav2_bt_navigator', 'bt_navigator'),
        server('nav2_velocity_smoother', 'velocity_smoother', [('cmd_vel', 'cmd_vel_nav')]),
        server('nav2_collision_monitor', 'collision_monitor'),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_navigation', output='screen',
             parameters=[{'autostart': True, 'node_names': LIFECYCLE_NODES}]),
    ])
    map_server = GroupAction([
        SetParameter('use_sim_time', use_sim_time),
        Node(package='nav2_map_server', executable='map_server', name='map_server',
             output='screen', parameters=[{'yaml_filename': LaunchConfiguration('map')}],
             remappings=remap),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_map', output='screen',
             parameters=[{'autostart': True, 'node_names': ['map_server']}]),
    ], condition=IfCondition(PythonExpression(["'", LaunchConfiguration('map'), "' != ''"])))
    obstacle_cloud = Node(package='sirius_navigation', executable='obstacle_cloud.py',
                          name='obstacle_cloud', namespace=LaunchConfiguration('obstacle_camera'),
                          output='screen',
                          parameters=[{'use_sim_time': use_sim_time, 'terrain_heightmap':
                                       LaunchConfiguration('terrain_heightmap')}])
    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'),
        DeclareLaunchArgument('nav2_params_file',
                              default_value=os.path.join(nav_share, 'config', 'nav2_params.yaml')),
        DeclareLaunchArgument('obstacle_camera', default_value='front'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('linear_vel', default_value='1.0',
                              description='prędkość jazdy RPP [m/s] (było 0,6)'),
        DeclareLaunchArgument('angular_vel', default_value='1.0',
                              description='prędkość obrotu w miejscu RPP [rad/s] (było 0,8)'),
        DeclareLaunchArgument('collision_horizon', default_value='1.5',
                              description='RPP: czas sprawdzania kolizji naprzód [s]'),
        DeclareLaunchArgument('map', default_value='',
                              description='mapa przejezdności (nav2_map_server); '' - bez niej'),
        DeclareLaunchArgument('terrain_heightmap', default_value='',
                              description='mapa wysokości terenu (.yaml) dla obstacle_cloud: '
                                          'przeszkody także względem znanego terenu (usypiska); '
                                          "'' - tylko lokalny grunt z kamery"),
        OpaqueFunction(function=_with_static_layer),
        map_server, obstacle_cloud, nodes,
    ])
