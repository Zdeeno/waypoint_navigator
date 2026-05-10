import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory('waypoint_follower')
    default_params = os.path.join(pkg_share, 'config', 'waypoint_follower_sim.yaml')

    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Path to the waypoint_follower parameter file.',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true.',
        ),
        Node(
            package='waypoint_follower',
            executable='waypoint_follower_node',
            name='waypoint_follower',
            output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time}],
        ),
    ])
