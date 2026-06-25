"""nav2_bringup.launch.py — Nav2 stack 시각장애인 8방향 햅틱 가이드

구성:
  - map_server: floor1_v6_map.pgm + yaml 로드 → /map publish
  - planner_server: SmacPlanner2D (cost-aware, 가운데 선호)
  - controller_server: RegulatedPurePursuitController (안정 path tracking)
  - bt_navigator: nav goal → BT 실행
  - behavior_server: spin/backup/wait recovery
  - waypoint_follower: multi-waypoint 지원 (optional)
  - lifecycle_manager: nav2 노드 생명주기 관리

전제:
  - localization.launch.py가 먼저 실행되어 map → odom → base_link TF chain 발행 중
  - track_unknown_space: false 적용 (nav2_params.yaml) → 매핑 free 0% 회피
  - InflationLayer radius 1.0m + scaling 3.0 → walls 멀리 (시각장애인 안전 마진)

사용:
  # 터미널 1: localization (RealSense + RTAB-Map)
  ros2 launch slam_mapping localization.launch.py \\
    database_path:=$HOME/maps/floor1_v6.db \\
    camera_height:=1.65

  # 터미널 2: nav2 stack
  ros2 launch slam_mapping nav2_bringup.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('slam_mapping')
    default_params = os.path.join(pkg_share, 'config', 'nav2_params.yaml')

    params_file = LaunchConfiguration('params_file')
    map_yaml = LaunchConfiguration('map')
    autostart = LaunchConfiguration('autostart')

    nav2_node_names = [
        'map_server',
        'planner_server',
        'controller_server',
        'behavior_server',
        'bt_navigator',
        'waypoint_follower',
    ]

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Nav2 stack 전체 파라미터 yaml',
        ),
        DeclareLaunchArgument(
            'map',
            default_value='/home/a/maps/floor1_v6_map.yaml',
            description='map_server가 로드할 yaml. db 매핑 외부 .pgm/.yaml',
        ),
        DeclareLaunchArgument(
            'autostart',
            default_value='true',
            description='lifecycle 자동 활성화',
        ),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='bag playback 시 true',
        ),

        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[
                params_file,
                {'yaml_filename': map_yaml,
                 'use_sim_time': LaunchConfiguration('use_sim_time')},
            ],
        ),

        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[params_file, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
        ),

        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[params_file, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
            remappings=[('cmd_vel', 'cmd_vel_nav')],
        ),

        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            parameters=[params_file, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
        ),

        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[params_file, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
        ),

        Node(
            package='nav2_waypoint_follower',
            executable='waypoint_follower',
            name='waypoint_follower',
            output='screen',
            parameters=[params_file, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
        ),

        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'autostart': autostart,
                'node_names': nav2_node_names,
            }],
        ),
    ])
