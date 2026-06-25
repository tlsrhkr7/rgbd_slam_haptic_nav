"""Phase 2 — Localization-only 모드.

매핑 끝난 .db 위에서 실시간 위치 추정만 수행. 새 keyframe 추가 안 함.

Usage:
    ros2 launch slam_mapping localization.launch.py \\
        database_path:=/home/a/maps/floor1.db

산출물:
    /odom              nav_msgs/Odometry      현재 위치 (사전 맵 기준)
    /map               OccupancyGrid          .db에서 로드한 사전 맵
    /rtabmap/cloud_map PointCloud2 (RGB)      3D 점군
    TF: map → odom → camera_link

설계 원칙:
    mapping.launch.py의 RealSense / madgwick / RTABMAP_PARAMS와 거의 동일.
    차이는 Mem/IncrementalMemory false (.db 변경 안 함) + localization=true 4개.
    이렇게 해야 매핑 때 검증한 odometry / loop closure 동작 그대로 재사용.
"""

import importlib.util
import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

from launch import LaunchDescription

# mapping.launch.py 에서 모든 튜닝을 단일 source of truth 로 임포트.
# 이전엔 손으로 복사해서 두 파일이 drift 됨 (LocalImmunizationRatio, Grid/*,
# GridGlobal/*, RawDescriptorsKept 등 20개 이상이 누락된 상태). importlib 로
# 파일명이 .launch.py 라 일반 import 가 안 되는 문제 우회.
_mapping_path = os.path.join(os.path.dirname(__file__), 'mapping.launch.py')
_spec = importlib.util.spec_from_file_location('_mapping_launch', _mapping_path)
_mapping_launch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mapping_launch)
RTABMAP_PARAMS_BASE = _mapping_launch.RTABMAP_PARAMS
RTABMAP_ODOM_PARAMS = _mapping_launch.RTABMAP_ODOM_PARAMS

# 2026-05-21: localization 시 odom cascade fail 차단 (빠른 회전 → graph 발산 방지)
# 2026-05-28: ResetCountdown 3→0. quality 14~31 저품질 프레임 3연속 시 오드메트리 강제
#   리셋 → odom→base_link TF 순간 변동 → 뷰어 빨간 원 텔레포트. 0=리셋 비활성.
RTABMAP_ODOM_LOCALIZATION_OVERRIDE = (
    ' --Odom/ResetCountdown 0 '       # 3→0: 자동 reset 비활성 (리셋이 텔레포트 원인)
    ' --Odom/GuessMotion false '      # 잘못된 guess 사용 X → cascade 차단
    ' --Odom/ImageBufferSize 5 '      # 이전 frame buffer 로 회복
    ' --Vis/MinInliers 4 '            # registration 통과 쉽게
)
RTABMAP_ODOM_FINAL = RTABMAP_ODOM_PARAMS + RTABMAP_ODOM_LOCALIZATION_OVERRIDE

# ───────────────────────────────────────────────────────────────────────────
# Localization 전용 오버라이드. RTAB-Map 인자 파싱은 **마지막 값 우선**이므로
# BASE 뒤에 APPEND 해야 함 (이전 prepend는 BASE의 IncrementalMemory true가
# OVERRIDE의 false를 덮어쓰는 버그 — db에 자동 Session 2 추가됨).
#   Mem/IncrementalMemory  true → false  (새 keyframe 추가 안 함)
#   Mem/InitWMWithAllNodes false → true  (시작 시 전체 .db 로드)
#   RGBD/StartAtOrigin     true → false  (.db 좌표계 그대로 시작)
# (localization=true 인자는 launch level 에서 별도 전달)
# ───────────────────────────────────────────────────────────────────────────
RTABMAP_LOCALIZATION_OVERRIDE = (
    # 핵심 localization 모드 강제 (4개)
    '--Mem/IncrementalMemory false '          # 매핑 모드 차단 (★)
    '--Mem/InitWMWithAllNodes true '          # 시작 시 db 전체 로드
    '--RGBD/StartAtOrigin false '             # db 좌표계 그대로 시작
    '--Reg/Force3DoF true '                   # 매핑과 동일 (2D 강제)
    # 2026-05-20 (옵션 B 롤백): 후속 Loop flash 안 터지는 문제 — LoopRatio/MinInliers/Iterations 복귀.
    '--Rtabmap/LoopThr 0.03 '                  # 0.05 → 0.03 (매칭 더 관대, 빠른 첫 트리거)
    '--RGBD/OptimizeMaxError 3 '               # 4→3: graph 와 불일치 큰 loop 거부 (틀어진 보정 차단 → 도착 정밀도 ↑)
    '--Rtabmap/MaxRetrieved 5 '                # retrieve 후보 ↑
    '--Optimizer/Iterations 20 '               # default 명시
    '--RGBD/LocalRadius 30 '                   # proximity 검색 반경 ↑
    '--Bayes/VirtualPlacePriorThr 0.3 '        # 0.5 → 0.3 (가상 prior 임계 더 ↓)
    '--Rtabmap/DetectionRate 2 '               # 4→2Hz 롤백: 4Hz+무거운 검사 → CPU 포화 → viewer freeze
    # ── 2026-05-28: loop closure 0-inlier 근본 수정 ──────────────────────────────────────
    # DB 파라미터 직접 쿼리 결과: Kp/DetectorStrategy=6, Vis/FeatureType=6 (GFTT/BRIEF).
    # Jetson custom rtabmap 기본값도 6 → ORB 강제는 오진이었음 (이전 수정에서 제거).
    #
    # [원인 A] Mem/UseOdomFeatures true(기본) → odom quality 14~166 불안정.
    #   quality=14면 feature 14개뿐 → VBoW 희박 → matches 6~23 → 6 inlier 달성 불가.
    #   UseOdomFeatures false → rtabmap이 카메라에서 Kp/MaxFeatures(750)만큼 균일 추출.
    '--Mem/UseOdomFeatures false '             # odom feature 의존 차단, rtabmap 직접 추출
    # ── 2026-06-01: 야간(어두움) 대응 A안 — 현재 프레임 feature 수/매칭 폭 확대 ──────────
    #   매핑은 아침 6~7시(밝음), 현재는 밤. BRIEF descriptor 는 조명 불변 X →
    #   현재 프레임에서 feature 를 최대한 많이/약한 코너까지 뽑아 매칭 후보를 늘림.
    #   (DB 는 노드당 ~814 feature 로 충분 → 현재 프레임 쪽만 보강)
    '--Kp/MaxFeatures 1200 '                  # 750→1200: 현재 프레임 vocab word ↑ → recall ↑
    '--Vis/MaxFeatures 2000 '                 # 1500→2000: RANSAC 대응점 후보 ↑
    '--GFTT/QualityLevel 0.0005 '             # 0.001→0.0005: 어두운 저대비 코너도 검출
    '--GFTT/MinDistance 3 '                   # feature 간 최소거리 ↓ → 더 조밀하게 추출
    # [원인 B] Vis/CorNNDR 0.8 너무 엄격 → tentative match 수 제한.
    '--Vis/CorNNDR 0.9 '                      # 0.8→0.9: match 허용 폭 ↑ → inlier 후보 ↑
    # [정밀도] Vis/InlierDistance — 작을수록 기하적으로 정확한 매칭만 inlier 인정 → 보정 정밀.
    #   0.8 은 너무 관대 → 부정확한 loop 수락 → 도착 시 위치 틀어짐. 정밀 우선으로 0.4.
    '--Vis/InlierDistance 0.4 '               # 0.8→0.4m: 정확한 inlier 만 → 도착 정밀도 ↑
    '--Vis/EstimationType 0 '                  # 3D-3D(0): DB 빌드 설정과 동일
    '--Vis/CorNNType 2 '                       # BruteForce Hamming: GFTT/BRIEF도 binary
    # DetectorStrategy 는 6(GFTT/BRIEF) 고정 — DB descriptor 타입과 일치해야 매칭됨 (변경 금지)
)

# 최종 인자 = base + override (★ override 뒤로 — 마지막 값 우선)
RTABMAP_LOCALIZATION_ARGS = RTABMAP_PARAMS_BASE + RTABMAP_LOCALIZATION_OVERRIDE


# RealSense 인자 — mapping과 완전 동일 (일관성)
REALSENSE_ARGS = {
    'align_depth.enable': 'true',
    'rgb_camera.color_profile': '640x480x30',
    'rgb_camera.color_format': 'BGR8',
    'rgb_camera.enable_auto_exposure': 'true',
    'depth_module.depth_profile': '640x480x30',
    'depth_module.enable_auto_exposure': 'true',
    'enable_sync': 'true',
    'enable_gyro': 'true',
    'enable_accel': 'true',
    'unite_imu_method': '2',
    'gyro_fps': '400',
    'accel_fps': '100',  # 250은 D435i 스펙상 가능하나 realsense2_camera Humble wrapper 거부 → 100 유지
    'depth_module.emitter_enabled': '1',
    'temporal_filter.enable': 'true',
    'spatial_filter.enable': 'false',
    'hole_filling_filter.enable': 'false',
    'decimation_filter.enable': 'false',
    'pointcloud.enable': 'false',
    'initial_reset': 'false',
}


def generate_launch_description():
    database_path = LaunchConfiguration('database_path')

    # 사람 마스킹: true 면 depth_mask_node 가 사람 영역을 0 으로 만든 image_masked 를
    # odometry/rtabmap 이 사용 → 사람 feature 로 인한 odom 드리프트 방지.
    DEPTH_RAW = '/camera/camera/aligned_depth_to_color/image_raw'
    DEPTH_MASKED = '/camera/camera/aligned_depth_to_color/image_masked'
    depth_topic = PythonExpression(
        ["'", DEPTH_MASKED, "' if '", LaunchConfiguration('mask_persons'),
         "' == 'true' else '", DEPTH_RAW, "'"])

    rs_launch = os.path.join(
        get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py'
    )

    rtabmap_launch = os.path.join(
        get_package_share_directory('rtabmap_launch'), 'launch', 'rtabmap.launch.py'
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'database_path',
                default_value='/home/a/maps/floor1.db',
                description='기존 매핑 .db 절대경로',
            ),
            # 사용자 카메라(D435i) 광학중심 높이 [m]. 머리에 장착 = 키 - 약 0.10m.
            # 매핑은 1.60m 4바퀴 차량 고정, 로컬라이제이션은 사용자별 가변.
            # 2D 보행 안내 목적상 ±20cm 오차는 매칭에 무영향, 기본 1.65(성인 중앙값).
            # 예: ros2 launch slam_mapping localization.launch.py camera_height:=1.80
            DeclareLaunchArgument(
                'camera_height',
                default_value='1.65',
                description='D435i 광학중심 바닥 위 높이 [m] — 사용자 키에 맞춰 지정',
            ),
            DeclareLaunchArgument(
                'rtabmap_viz',
                default_value='true',
                description='RTAB-Map GUI viewer 자동 띄움 여부. OpenCV 뷰어 쓸 땐 false',
            ),
            # bag playback 시 RealSense skip 용 — false 면 RealSense include 안 함
            DeclareLaunchArgument(
                'enable_realsense',
                default_value='true',
                description='RealSense 자동 launch 여부. bag playback 시 false 로 설정',
            ),
            DeclareLaunchArgument(
                'use_sim_time',
                default_value='false',
                description='bag playback 시 true (sim time 동기화)',
            ),
            DeclareLaunchArgument(
                'mask_persons',
                default_value='true',
                description='YOLO 사람 bbox 영역 depth 마스킹 → odometry 사람 드리프트 방지',
            ),
            # 1) RealSense (mapping과 완전 동일)
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(rs_launch),
                launch_arguments=REALSENSE_ARGS.items(),
                condition=IfCondition(LaunchConfiguration('enable_realsense')),
            ),
            # 1c) depth 마스킹 — 사람 bbox 영역 depth=0 (rgbd_odometry 가 image_masked 사용).
            #     bbox 없으면 depth 그대로 통과 → 사람 없을 땐 정상.
            Node(
                package='slam_mapping',
                executable='depth_mask_node',
                name='depth_mask',
                output='screen',
                condition=IfCondition(LaunchConfiguration('mask_persons')),
                parameters=[{
                    'depth_in': DEPTH_RAW,
                    'depth_out': DEPTH_MASKED,
                    'bbox_topic': '/yolo/person_bboxes',
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                }],
            ),
            # 1b) base_link → camera_link static TF — 사용자 키에 맞춰 동적.
            #     매핑 .db는 1.60m로 기록됐지만 시각 매칭은 카메라 절대좌표만
            #     보므로 사람 키 차이는 base_link z 값 차이로 흡수 (Force3DoF가 clamp).
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='base_to_camera_tf',
                arguments=['0', '0', LaunchConfiguration('camera_height'),
                           '0', '0', '0',
                           'base_link', 'camera_link'],
                parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            ),
            # 2) madgwick (mapping과 동일)
            Node(
                package='imu_filter_madgwick',
                executable='imu_filter_madgwick_node',
                name='imu_filter_madgwick',
                remappings=[
                    ('imu/data_raw', '/camera/camera/imu'),
                    ('imu/data', '/imu/data'),
                ],
                parameters=[
                    {
                        'use_mag': False,
                        'world_frame': 'enu',
                        'publish_tf': False,
                        'gain': 0.15,
                        'zeta': 0.005,
                        'use_sim_time': LaunchConfiguration('use_sim_time'),
                        # bag /camera/camera/imu 는 Best Effort 로 녹화됨.
                        # madgwick 기본 subscriber QoS 는 Reliable → FastRTPS 에서 불일치.
                        'qos_overrides./camera/camera/imu.subscription.reliability': 'best_effort',
                        'qos_overrides./camera/camera/imu.subscription.depth': 10,
                        'qos_overrides./camera/camera/imu.subscription.history': 'keep_last',
                    }
                ],
            ),
            # 3) RTAB-Map — Localization-only
            # 주: map_always_update=True 는 사용 안 함 (rtabmap maps_update 1.5s/cycle
            # 부하 폭주 → controller_server SIGABRT → /plan 불가). 대신 viewer 가
            # publish_map service 를 비동기 retry 호출 (localization_viewer_node._try_publish_map).
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(rtabmap_launch),
                launch_arguments={
                    'rgb_topic': '/camera/camera/color/image_raw',
                    'depth_topic': depth_topic,  # mask_persons=true 면 image_masked
                    'camera_info_topic': '/camera/camera/color/camera_info',
                    'imu_topic': '/imu/data',
                    'frame_id': 'base_link',
                    'odom_frame_id': 'odom',
                    'map_frame_id': 'map',
                    'publish_tf_map': 'true',
                    'approx_sync': 'true',
                    'approx_sync_max_interval': '0.05',
                    'queue_size': '30',
                    'qos': '2',  # Best Effort — RealSense 발행 QoS와 일치
                    'log_level': 'info',
                    'subscribe_rgbd': 'false',
                    # Localization은 IMU init 대기 (.db 좌표계 정확 정렬)
                    'wait_imu_to_init': 'true',
                    # ★ Localization 모드 ★
                    'localization': 'true',
                    'database_path': database_path,
                    'rtabmap_args': RTABMAP_LOCALIZATION_ARGS,
                    'odom_args': RTABMAP_ODOM_FINAL,  # cascade 차단 옵션 포함
                    'rviz': 'false',
                    'rtabmap_viz': LaunchConfiguration('rtabmap_viz'),
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                }.items(),
            ),
        ]
    )
