"""Nav2 for Sirius (replaces ROS 1 move_base / mbf_costmap_nav) + depth obstacle cloud.

Uses config/nav2_params.yaml (Nav2 format). Obstacles: /<obstacle_camera>/depth/image_raw
-> obstacle_cloud.py -> /<obstacle_camera>/obstacles (PointCloud2 in base_link).
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('sirius_navigation')
    nav2_launch = get_package_share_directory('nav2_bringup') + '/launch/navigation_launch.py'
    return LaunchDescription([
        DeclareLaunchArgument('nav2_params_file', default_value=share + '/config/nav2_params.yaml'),
        DeclareLaunchArgument('obstacle_camera', default_value='front', description='Depth camera namespace'),
        Node(package='sirius_navigation', executable='obstacle_cloud.py', name='obstacle_cloud',
             namespace=LaunchConfiguration('obstacle_camera'), output='screen'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_launch),
            launch_arguments={'params_file': LaunchConfiguration('nav2_params_file'),
                              'use_sim_time': 'false',
                              'autostart': 'true'}.items()),
    ])
