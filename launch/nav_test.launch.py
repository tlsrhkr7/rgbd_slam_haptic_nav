"""nav_test.launch.py — localization + viewer + Nav2 + goal sender 통합.

사용:
  ros2 launch slam_mapping nav_test.launch.py
  # 다른 목적지:
  ros2 launch slam_mapping nav_test.launch.py target:="학과사무실"

자동 동작:
  1. localization_view 모두 시작 (RealSense + RTAB + viewer + YOLO + POI 표시)
  2. 15초 후 Nav2 stack 시작 (RTAB 안정 + db.grid publish 대기)
  3. Nav2 활성화 + 추가 5초 후 goal_sender 가 POI → /goal_pose 발행
  4. Nav2 SmacPlanner2D 가 /plan 생성 → viewer 가 초록 굵은 선으로 표시

분리 launch 원하면 localization_view.launch.py 단독 사용 가능 (viewer 만, nav 없음).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('slam_mapping')
    localization_view_launch = os.path.join(
        pkg_share, 'launch', 'localization_view.launch.py')
    nav2_bringup_launch = os.path.join(
        pkg_share, 'launch', 'nav2_bringup.launch.py')
    default_params = os.path.join(pkg_share, 'config', 'nav2_params.yaml')

    target = LaunchConfiguration('target')
    poi_file = LaunchConfiguration('poi_file')

    return LaunchDescription([
        DeclareLaunchArgument(
            'target',
            default_value='텐서',
            description='POI 이름. 기본 텐서.',
        ),
        DeclareLaunchArgument(
            'poi_file',
            default_value='/home/a/maps/floor4_pois.yaml',
            description='POI yaml 경로',
        ),
        DeclareLaunchArgument(
            'camera_height',
            default_value='1.36',
            description='D435i 광학중심 바닥 위 높이 [m]',
        ),

        # 1) Localization + viewer + YOLO (모든 시각화 포함)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(localization_view_launch),
            launch_arguments={
                'camera_height': LaunchConfiguration('camera_height'),
            }.items(),
        ),

        # 2) Nav2 stack — RTAB / TF 안정화 충분히 대기 (15초)
        TimerAction(
            period=15.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(nav2_bringup_launch),
                    launch_arguments={
                        'params_file': default_params,
                        'map': '/home/a/maps/floor4_v3.yaml',
                        'autostart': 'true',
                    }.items(),
                ),
            ],
        ),

        # 3) Goal sender — Nav2 활성화 후 발행 (총 25초 후)
        TimerAction(
            period=25.0,
            actions=[
                Node(
                    package='slam_mapping',
                    executable='goal_sender_node',
                    name='goal_sender',
                    output='screen',
                    parameters=[{
                        'poi_file': poi_file,
                        'target': target,
                        'delay_s': 2.0,
                        'map_frame': 'map',
                        'use_projection': False,   # ★ yaml 좌표 그대로 (projection 끔)
                    }],
                ),
            ],
        ),
        # 4) Path → Haptic 진동 명령 변환
        TimerAction(
            period=20.0,
            actions=[
                Node(
                    package='slam_mapping',
                    executable='path_to_haptic_node',
                    name='path_to_haptic',
                    output='screen',
                    parameters=[{
                        'lookahead_dist': 2.0,   # 1.0→2.0: 코너 미리 안내 (벽 정면 도달 전 회전 유도)
                        'goal_tolerance': 0.5,
                        'rate': 5.0,
                        'user_frame': 'base_link',
                        'map_frame': 'map',
                    }],
                ),
                # 5) /haptic_motor_idx → Arduino 시리얼 송신 (USB ttyACM0)
                Node(
                    package='slam_mapping',
                    executable='serial_haptic_node',
                    name='serial_haptic',
                    output='screen',
                    parameters=[{
                        'port': '/dev/ttyACM0',
                        'baud': 115200,
                        'topic': '/haptic_motor_idx',
                        'keepalive_rate': 5.0,
                        'input_timeout_s': 0.5,
                    }],
                ),
            ],
        ),
    ])
