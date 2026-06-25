"""localization_view.launch.py — localization (db) + 외부 pgm 시각화 + OpenCV 뷰어.

전략:
  - localization db = 원본 .db (visual feature 매핑 직후 그대로, 매칭 정확)
  - 시각화 pgm    = reprocess .yaml (graph optimized 92%, ㄱ자 깔끔)
  - 둘 분리 → 매칭 품질 + 시각화 품질 동시 확보

사용:
  ros2 launch slam_mapping localization_view.launch.py
  # 인자 override:
  ros2 launch slam_mapping localization_view.launch.py \\
    database_path:=$HOME/maps/floor4.db \\
    map_yaml:=$HOME/maps/floor4_v2_map.yaml \\
    camera_height:=1.54

자동 띄움:
  - RealSense D435i + madgwick + base_link TF
  - rgbd_odometry + rtabmap (localization mode, db에 기록 X)
  - nav2 map_server + lifecycle_manager (pgm/yaml load → /map publish)
  - localization_viewer_node (OpenCV 단일 창, /map subscribe)

조작:
  q / ESC  종료    f  follow↔full 토글    r  궤적 reset    +/-  zoom
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('slam_mapping')
    localization_launch = os.path.join(pkg_share, 'launch', 'localization.launch.py')

    database_path = LaunchConfiguration('database_path')
    camera_height = LaunchConfiguration('camera_height')
    map_yaml = LaunchConfiguration('map_yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'database_path',
            default_value='/tmp/floor4_session.db',  # ★ 격리된 세션 db (원본 보호)
            description='RTAB-Map .db (격리 세션). launch 시 backup → /tmp 복사, v3.db 무손상',
        ),
        DeclareLaunchArgument(
            'map_yaml',
            default_value='/home/a/maps/floor4_v3.yaml',  # ★ GUI export (좌표계 정확)
            description='시각화용 .yaml/.pgm (GUI export 결과, db.grid origin 일치)',
        ),
        DeclareLaunchArgument(
            'camera_height',
            default_value='1.54',
            description='카메라 광학중심 z (m). 4바퀴 차량 기준 1.54',
        ),
        DeclareLaunchArgument(
            'enable_realsense',
            default_value='true',
            description='RealSense 자동 launch. bag playback 시 false 전달',
        ),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='bag playback 시 true',
        ),
        # 0) 격리 세션 db — backup → /tmp 복사. v3.db, backup.db 둘 다 영구 보호.
        #    RTAB 가 localization 모드에서도 graph link 추가/optimization 누적 → db write.
        #    /tmp 격리로 원본 db 완전 차단. launch 종료 시 /tmp 그대로 (다음 launch 가 덮어씀).
        ExecuteProcess(
            cmd=['bash', '-c',
                 'cp -f /home/a/maps/floor4_v3_backup.db /tmp/floor4_session.db '
                 '&& echo "[session] /tmp/floor4_session.db ← backup (격리 시작)"'],
            output='screen',
        ),
        # 1) localization (db) — rtabmap이 db 매칭으로 TF map→base_link 발행
        #    Race condition 방지: 339MB cp 완료 대기. Jetson eMMC ~100MB/s → 3.4s.
        #    2s 부족 → database is locked FATAL. 8s 로 여유 확보.
        TimerAction(
            period=8.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(localization_launch),
                    launch_arguments={
                        'database_path': database_path,
                        'camera_height': camera_height,
                        'rtabmap_viz': 'false',  # OFF — 노드별 cloud 누적으로 메모리 폭증→freeze. 3D 맵은 view_3dmap.sh (databaseViewer) 로 별도 확인
                        'enable_realsense': LaunchConfiguration('enable_realsense'),
                        'use_sim_time': LaunchConfiguration('use_sim_time'),
                    }.items(),
                ),
            ],
        ),
        # (map_server 제거 — viewer 는 /rtabmap/map (db.grid) 사용.
        #  nav_test.launch.py 의 nav2_bringup 이 자체 map_server 띄움. 중복 방지.)
        # 4) OpenCV 뷰어 — TF 안정화 + map_server 활성화 대기
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='slam_mapping',
                    executable='localization_viewer_node',
                    name='localization_viewer',
                    output='screen',
                    parameters=[{
                        'map_topic': '/rtabmap/map',  # ★ db.grid 직접 — 좌표계 100% 일치
                        'user_frame': 'base_link',
                        'map_frame': 'map',
                        'window_size': 700,
                        'follow_radius_m': 12.0,
                        'render_rate': 5.0,   # 20→10→5Hz: Jetson 렉 감소
                        'use_sim_time': LaunchConfiguration('use_sim_time'),
                    }],
                ),
            ],
        ),
        # 5) YOLO person detector — TF + camera 안정화 후 시작
        TimerAction(
            period=7.0,
            actions=[
                Node(
                    package='slam_mapping',
                    executable='yolo_person_node',
                    name='yolo_person',
                    output='screen',
                    parameters=[{
                        'model_path': '/home/a/ros2_ws/yolo11n.engine',  # TensorRT FP16
                        'conf_threshold': 0.5,
                        'camera_frame': 'camera_color_optical_frame',
                        'map_frame': 'map',
                        'infer_rate': 4.0,
                        'use_sim_time': LaunchConfiguration('use_sim_time'),
                    }],
                ),
            ],
        ),
    ])
