"""SLAM 매핑 (RTAB-Map) — 시각장애인 보행자 완전 튜닝.

이 launch는 매핑 품질을 극대화하기 위해 RealSense / madgwick / rgbd_odometry /
rtabmap 의 모든 단계를 보행자 환경에 맞춰 정밀 튜닝한다. 매핑 산출물(.db)은
이후 localization.launch.py가 그대로 재사용한다.

Usage:
    ros2 launch slam_mapping mapping.launch.py \\
        database_path:=/home/a/maps/floor1.db

인자:
    database_path : .db 저장 절대경로 (default /home/a/maps/floor1.db)
    delete_db     : true=새로 매핑, false=누적 매핑 (default true)
    use_rviz      : RViz 실행 여부 (default false — rtabmap_viz가 메인 GUI)

산출물:
    /odom              nav_msgs/Odometry        rgbd_odometry 출력
    /map               OccupancyGrid            2D grid (Nav2 입력)
    /rtabmap/cloud_map PointCloud2 (RGB)        3D 점군
    /rtabmap/info      rtabmap_msgs/Info        loop closure / 통계
    TF: map → odom → camera_link (RealSense 자체 TF는 camera_*_optical_frame까지)
    {database_path}                              매핑 .db (Localization 입력)
    rtabmap_viz GUI (실시간 시각화)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

from launch import LaunchDescription

# ═══════════════════════════════════════════════════════════════════════════
# RTAB-Map 매핑 파라미터 — 보행자 D435i 환경 완전 튜닝
#
# 각 파라미터는 docs/SLAM_ARCHITECTURE.md §4 근거 + RTAB-Map wiki 참고.
# default와 다른 값만 명시. default 좋은 건 그대로.
# ═══════════════════════════════════════════════════════════════════════════
# ───────────────────────────────────────────────────────────────────────────
# [A] rgbd_odometry — 시각-관성 odometry
#   F2M(Frame-to-Map)이 default. 보행 환경에선 F2M이 가장 robust.
#
# ⚠ Odom/* 와 OdomF2M/* 는 rgbd_odometry 노드만 인식. rtabmap (SLAM core)
# 에 같이 전달하면 ParameterNotDeclaredException 으로 crash. 그래서
# RTABMAP_ODOM_PARAMS 는 rtabmap_launch 의 odom_args 로만 전달하고
# RTABMAP_PARAMS (args) 와 분리.
# ───────────────────────────────────────────────────────────────────────────
RTABMAP_ODOM_PARAMS = (
    '--Odom/Strategy 0 '  # 0=F2M, 1=F2F. F2M 권장
    '--Odom/MinInliers 6 '  # 20 → 6. odom init 빠르게 (Vis/MinInliers 와 동일)
    '--Odom/GuessMotion true '  # 직전 motion으로 initial guess
    '--Odom/GuessSmoothingDelay 0 '  # IMU prior 즉시 사용
    '--Odom/ResetCountdown 0 '  # 자동 reset OFF
    '--Odom/FilteringStrategy 0 '  # 0=NoFilter (Kalman 은 평활 과해 이동 추종 막힘 → 제거)
    '--Odom/AlignWithGround false '
    '--Odom/Holonomic true '  # 보행자는 holonomic 모션
    # F2M(Frame-to-Map) 구체 옵션
    '--OdomF2M/MaxSize 1500 '  # 1000→1500: 무특징 구간 약간 보강 (3000은 이동 축소 부작용 → 1500)
    '--OdomF2M/BundleAdjustment 0 '  # OFF — BA가 이동량을 과하게 눌러(under-estimate) 실제보다 위치 덜 변함
    '--OdomF2M/MaxNewFeatures 0 '  # 0=무제한
)

RTABMAP_PARAMS = (
    # ───────────────────────────────────────────────────────────────────────
    # [B] 보행 동역학 — keyframe 추가 임계값 / odom buffer
    # ───────────────────────────────────────────────────────────────────────
    '--Reg/Force3DoF true '  # ★ 4축 구조물 고정-Z 매핑 — z/roll/pitch 0 강제 (미끄럼틀 차단)
    '--Reg/Strategy 0 '  # 0=Visual, 1=ICP, 2=Visual+ICP
    '--RGBD/AngularUpdate 0.03 '  # ~1.7°마다 keyframe (구조물 정적, 조밀 샘플링)
    '--RGBD/LinearUpdate 0.03 '  # 3cm마다 keyframe (정적 매핑 정밀도)
    '--RGBD/MaxOdomCacheSize 20 '  # odom buffer 크게 (빠른 회전 안정)
    '--RGBD/CreateOccupancyGrid true '  # grid 생성 ON
    # ───────────────────────────────────────────────────────────────────────
    # [C] Visual Feature — RGB-D 3D-3D estimation (가장 robust한 방식)
    #   FIX: 이전 PnP(EstimationType 1)는 inlier 0 으로 odom init 실패.
    #   RGB-D 환경에선 EstimationType 0 (3D-3D Procrustes)이 depth를 직접 활용
    #   해서 더 안정적. PnP는 외부 카메라 intrinsic 정확도에 민감하지만,
    #   3D-3D는 두 frame의 3D point cloud 간 직접 변환 추정.
    # ───────────────────────────────────────────────────────────────────────
    '--Vis/MinInliers 6 '  # 20 → 6. 보행 D435i 환경에서 충분히 동작. 10 으로
    # 올리면 bag 회귀 테스트에서 keyframe 18→1 로 급감. 단조 복도 false match
    # 위험은 있으나 실측상 6 이 가장 robust.
    '--Vis/MaxFeatures 500 '   # 1500→500: loop closure RAM/CPU 절감 (localization에 충분)
    '--Vis/EstimationType 0 '  # ★ PnP(1) → 3D-3D(0). RGB-D 직접
    '--Vis/CorNNType 1 '  # FLANN
    '--Vis/CorNNDR 0.8 '
    '--Vis/InlierDistance 0.1 '  # 3D-3D inlier 거리 (m)
    '--Vis/RefineIterations 5 '
    '--Vis/Iterations 300 '  # RANSAC iteration ↑
    # ───────────────────────────────────────────────────────────────────────
    # [D] Keypoint pool (loop closure용 vocabulary)
    # ───────────────────────────────────────────────────────────────────────
    '--Kp/MaxFeatures 300 '  # 750→300: BoW vocabulary 경량화
    '--Kp/DetectorStrategy 6 '  # 6=GFTT/BRIEF, 8=ORB
    '--Vis/FeatureType 6 '  # ★ Kp와 일치 (Mem/UseOdomFeatures 활성)
    '--Kp/NNStrategy 1 '  # FLANN
    '--Kp/IncrementalDictionary true '  # vocabulary 학습 ON
    '--Kp/IncrementalFlann true '  # FLANN index 학습
    '--GFTT/MinDistance 7 '  # feature 간 최소 7px
    '--GFTT/QualityLevel 0.001 '  # 약한 corner도 포함
    '--GFTT/UseHarrisDetector false '  # GFTT (Harris보다 안정)
    '--GFTT/BlockSize 3 '
    # ───────────────────────────────────────────────────────────────────────
    # [E] Loop Closure — Bayesian + Proximity
    #   Bayesian filter가 false positive 줄여서 복도 매핑에 유리.
    # ───────────────────────────────────────────────────────────────────────
    '--Rtabmap/LoopThr 0.11 '  # loop closure 임계 (default 0.11)
    '--Rtabmap/LoopRatio 0.0 '  # 0=비활성, ratio loop closure
    '--Rtabmap/DetectionRate 1 '  # 1Hz 검출 (천천히 보행 + RAM 35% 절감, 품질 유지)
    '--Rtabmap/TimeThr 700 '  # 처리 시간 제한 (ms)
    '--RGBD/ProximityBySpace true '  # 재방문 loop closure 강화
    '--RGBD/ProximityByTime false '  # 시간 기준 X
    '--RGBD/ProximityMaxGraphDepth 50 '  # graph 검색 깊이
    '--RGBD/ProximityPathMaxNeighbors 10 '
    '--RGBD/ProximityPathFilteringRadius 1.0 '
    '--RGBD/NeighborLinkRefining true '  # 인접 노드 변환 재추정
    '--RGBD/OptimizeMaxError 3 '  # graph optimize outlier 허용
    '--RGBD/LocalImmunizationRatio 0.25 '
    '--RGBD/LocalRadius 10 '  # local map 검색 반경 (m)
    # Bayes/PredictionLC 는 단일 값 아닌 분포 리스트 — default 사용 (수정 X)
    '--Bayes/VirtualPlacePriorThr 0.9 '
    '--Bayes/FullPredictionUpdate false '  # 속도 위해 partial update
    # ───────────────────────────────────────────────────────────────────────
    # [F] Optimizer — graph optimization (g2o)
    # ───────────────────────────────────────────────────────────────────────
    '--Optimizer/Strategy 1 '  # 1=g2o, 0=TORO, 2=GTSAM
    '--Optimizer/Robust true '  # robust kernel (Huber)
    '--Optimizer/Iterations 20 '
    '--Optimizer/Epsilon 0.0 '  # 무제한 수렴
    '--Optimizer/PriorsIgnored false '  # priors 사용 (IMU 등)
    '--Optimizer/LandmarksIgnored false '
    '--Optimizer/GravitySigma 0.1 '  # IMU gravity constraint (계단/경사 강한 제약)
    # ───────────────────────────────────────────────────────────────────────
    # [G] Memory Management
    # ───────────────────────────────────────────────────────────────────────
    # '--Mem/IncrementalMemory true' 는 BASE에서 제거. localization launch가 import해서
    # BASE+OVERRIDE 결합 시 마지막 값 우선이 안 먹는 케이스 발견 (db 변경됨).
    # mapping launch 는 아래 rtabmap_args 에 직접 명시, localization 은 OVERRIDE 의 false 만.
    '--Mem/STMSize 30 '  # 단기 메모리 30 keyframe
    '--Mem/RehearsalSimilarity 0.6 '  # 유사 location 인식
    '--Mem/RecentWmRatio 0.2 '
    '--Mem/UseOdomGravity true '  # gravity로 graph 정렬 보강
    '--Mem/SaveDepth16Format true '  # depth 16-bit 저장 (작음)
    '--Mem/RawDescriptorsKept true '  # 원본 descriptor 보존 (재처리 가능)
    '--Mem/BinDataKept true '  # binary 데이터 보존
    # ★ RAM freeze 방지 (8GB 환경, db 용량은 신경 X)
    '--Rtabmap/MemoryThr 700 '  # WM 노드 700개 cap, 도달 시 LTM(디스크)로 transfer
    '--Rtabmap/MaxRetrieved 2 '  # loop closure retrieval 시 RAM spike 차단
    # ★ free cell 강제 생성 강화 (z 정확성 무관하게 free 보장)
    '--Grid/Scan2dUnknownSpaceFilled true '  # unknown 공간을 ray로 자동 free 채움
    # ───────────────────────────────────────────────────────────────────────
    # [H] Grid (Nav2 입력) — 보행자 사용자 키 1.8m 가정
    # ───────────────────────────────────────────────────────────────────────
    '--Grid/FromDepth true '  # depth로 grid 생성
    '--Grid/CellSize 0.03 '  # 3cm 해상도 (길안내 벽 정밀도)
    '--Grid/RangeMax 3.5 '  # D435i depth noise σ<2cm 신뢰 범위
    '--Grid/RangeMin 0.3 '  # depth 최소
    # base_link frame 기준 (floor=z=0). 밴드 [MinGround,MaxGround]∩[MinObs,MaxObs]
    # 겹치면 NormalSegmentation이 normal로 분류 (수평=ground, 수직=obstacle).
    '--Grid/MaxObstacleHeight 1.8 '  # 사용자 키, 천장 무시
    '--Grid/MinObstacleHeight 0.03 '  # 3cm 턱·문턱 (시각장애인 안전 — 더 낮추면 노이즈)
    '--Grid/MinGroundHeight -0.20 '  # 바닥 -20cm 여유 (depth 노이즈·바닥 굴곡)
    '--Grid/MaxGroundHeight 0.20 '  # 바닥 +20cm 여유 (normals가 정밀 분류)
    '--Grid/MaxGroundAngle 45 '  # 경사 45° 이내 = 바닥
    '--Grid/RayTracing true '  # free space 명시
    '--Grid/Sensor 1 '  # 0=scan, 1=depth, 2=both (D435i RGB-D=depth)
    '--Grid/3D false '  # 3D voxel 비활성 → RAM 대폭 절감 (2D occupancy grid만 사용)
    '--Grid/DepthDecimation 4 '  # 1/4 다운샘플 (속도)
    '--Grid/NormalSegmentation true '  # 지면/장애물 normal로 분리
    '--Grid/FlatObstacleDetected true '  # 평평한 장애물도 검출
    '--Grid/MinClusterSize 10 '  # cluster 최소 크기
    '--Grid/ClusterRadius 0.1 '
    '--Grid/MapFrameProjection true '  # map frame에 projection
    '--GridGlobal/UpdateError 0.01 '  # global grid update 임계
    '--GridGlobal/MinSize 100 '
    '--GridGlobal/Eroded false '
)

# ═══════════════════════════════════════════════════════════════════════════
# RealSense 파라미터 — 보행자 환경 depth 품질 최적화
# ═══════════════════════════════════════════════════════════════════════════
REALSENSE_ARGS = {
    # 정렬된 depth (RGB 시점) — RTAB-Map 입력 표준
    'align_depth.enable': 'true',
    # 30Hz / 640x480 (실내 매핑 권장)
    'rgb_camera.color_profile': '640x480x30',
    'rgb_camera.color_format': 'BGR8',  # RTAB-Map BGR 가정
    'rgb_camera.enable_auto_exposure': 'true',  # 보행 시 조명 변화 대응
    'depth_module.depth_profile': '640x480x30',
    'depth_module.enable_auto_exposure': 'true',
    'enable_sync': 'true',  # RGB-D 시간 동기
    # IMU (raw) — madgwick으로 전달
    'enable_gyro': 'true',
    'enable_accel': 'true',
    'unite_imu_method': '2',  # linear interpolation
    'gyro_fps': '400',
    'accel_fps': '100',  # 250은 D435i 스펙상 가능하나 realsense2_camera Humble wrapper 거부 → 100 유지
    # IR emitter — 실내 depth 품질
    'depth_module.emitter_enabled': '1',  # ON
    # Depth filter — 시간 평균 ON, 나머지 OFF (노이즈 vs CPU 균형)
    'temporal_filter.enable': 'true',
    'spatial_filter.enable': 'false',
    'hole_filling_filter.enable': 'false',
    'decimation_filter.enable': 'false',  # 해상도 유지
    # pointcloud는 RTAB-Map가 만듦
    'pointcloud.enable': 'false',
    # USB busy 방지
    'initial_reset': 'false',
    # depth far clip 비활성 — clip이 가까운 객체에서 invalid depth 만들 수 있음
    # (RTAB-Map의 Grid/RangeMax 5.0 이 적절 cut 처리)
    # 'clip_distance': '5.0',
}


def generate_launch_description():
    database_path = LaunchConfiguration('database_path')
    delete_db = LaunchConfiguration('delete_db')
    use_rviz = LaunchConfiguration('use_rviz')

    rs_launch = os.path.join(
        get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py'
    )

    rtabmap_launch = os.path.join(
        get_package_share_directory('rtabmap_launch'), 'launch', 'rtabmap.launch.py'
    )

    rtabmap_args = PythonExpression(
        [
            "('--delete_db_on_start ' if '",
            delete_db,
            "' == 'true' else '')",
            " + '",
            RTABMAP_PARAMS,
            # ★ mapping launch 전용: IncrementalMemory true 직접 추가 (BASE 에서 제거됨).
            # localization launch 는 BASE 만 import 하므로 이 줄 영향 없음.
            ' --Mem/IncrementalMemory true ',
            "'",
        ]
    )
    # Odom/* 는 odom_args 로만 (rtabmap SLAM 노드는 Odom/* 못 받음)
    rtabmap_odom_args = RTABMAP_ODOM_PARAMS

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'database_path',
                default_value='/home/a/maps/floor1.db',
                description='저장할 .db 파일 경로 (절대)',
            ),
            DeclareLaunchArgument(
                'delete_db', default_value='true', description='true=새로, false=이어찍기'
            ),
            DeclareLaunchArgument(
                'use_rviz', default_value='false',
                description='경량 RViz2(2D 맵+TF+궤적). 기본은 use_monitor(OpenCV) 사용'
            ),
            DeclareLaunchArgument(
                'use_monitor', default_value='true',
                description='매핑 보강 모니터(OpenCV 2D): 맵+위치/방향+loop flash+새 노드(초록)'
            ),
            DeclareLaunchArgument(
                'camera_height', default_value='1.36',
                description='D435i 광학중심 바닥 위 높이 [m]. 웨어러블(가슴) 기준 1.36 '
                            '(nav_test localization 과 동일). 4바퀴 차량은 1.54.'
            ),
            # ═══════════════════════════════════════════════════════════════════
            # 1) RealSense D435i — RGB + aligned depth + IMU
            # ═══════════════════════════════════════════════════════════════════
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(rs_launch),
                launch_arguments=REALSENSE_ARGS.items(),
            ),
            # ═══════════════════════════════════════════════════════════════════
            # 2) base_link → camera_link static TF — Grid occupancy 정합 필수
            #    카메라 광학중심 z = camera_height (기본 1.36 웨어러블, nav_test 와 동일).
            #    ★ 매핑 시 실제 카메라 높이와 일치해야 바닥(z=0) 투영이 맞음.
            #      틀리면 바닥 점이 엉뚱한 z 로 잡혀 ground 분류/occupancy grid 왜곡.
            #    이 TF 없으면 RTAB-Map이 카메라를 바닥으로 간주 → ground 분류 실패
            #    → free space 0% (벽만 obstacle로 잡힘). frame_id='base_link' 와 한 셋.
            #    pitch (카메라 하향각) 있으면 6번째 인자 라디안 입력.
            # ═══════════════════════════════════════════════════════════════════
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='base_to_camera_tf',
                arguments=['0', '0', LaunchConfiguration('camera_height'),
                           '0', '0', '0',
                           'base_link', 'camera_link'],
            ),
            # ═══════════════════════════════════════════════════════════════════
            # 3) madgwick — raw IMU → /imu/data (orientation 포함)
            #
            # gain 0.1: default. 보행 흔들림에 민감하면 0.05까지 낮춤.
            # zeta 0.0: gyro bias drift 보정 OFF (단기 매핑이라 OK).
            #          매핑 30분 이상이면 0.001~0.005로 ON 권장.
            # ═══════════════════════════════════════════════════════════════════
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
                        'use_mag': False,  # D435i magnetometer 없음
                        'world_frame': 'enu',
                        'publish_tf': False,  # RealSense가 TF 발행
                        'gain': 0.15,
                        'zeta': 0.005,
                    }
                ],
            ),
            # ═══════════════════════════════════════════════════════════════════
            # 4) RTAB-Map (rgbd_sync + rgbd_odometry + rtabmap + rtabmap_viz)
            #    표준 launch include — 4개 노드 자동 띄움.
            #    위 RTABMAP_PARAMS로 모든 파라미터 정밀 튜닝.
            # ═══════════════════════════════════════════════════════════════════
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(rtabmap_launch),
                launch_arguments={
                    # 입력 토픽
                    'rgb_topic': '/camera/camera/color/image_raw',
                    'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
                    'camera_info_topic': '/camera/camera/color/camera_info',
                    'imu_topic': '/imu/data',
                    # TF frame — base_link 기준. 위의 base_to_camera_tf (z=1.54)이
                    # base_link → camera_link 체인을 채워 occupancy grid가 floor(z=0)
                    # 기준으로 투영됨. camera_link로 두면 floor가 z=-1.6에 와서
                    # MaxGroundHeight 0.1로 ground 분류 실패 → free=0%.
                    'frame_id': 'base_link',
                    'odom_frame_id': 'odom',
                    'map_frame_id': 'map',
                    'publish_tf_map': 'true',
                    # 동기화 — RGB-D 시간차 ~30ms 허용
                    'approx_sync': 'true',
                    'approx_sync_max_interval': '0.05',  # 0.02→0.05. 너무 빡빡해서 sync fail 방지
                    'queue_size': '30',
                    # ★ QoS: RealSense는 Best Effort (sensor_data)로 발행.
                    #   Reliable(1)로 두면 매칭 안 돼서 rtabmap이 데이터 못 받음
                    #   → "노드가 작동중인가요?" 메시지. 2(Best Effort)로 통일.
                    'qos': '2',
                    'log_level': 'info',
                    # rgbd_sync 자동 띄움
                    'subscribe_rgbd': 'false',
                    # IMU — true 로 두면 IMU 가 안정 orientation 보낼 때까지 대기 →
                    # 시작 좌표계가 gravity-align 되고 odometry 에 IMU prior 가 일관되게
                    # 적용됨. false 는 빠르지만 odom lost 후 회복 어려움. 공식
                    # rtabmap_examples/realsense_d435i_color.launch.py 도 true 사용.
                    'wait_imu_to_init': 'true',
                    # 매핑 모드 ON
                    'localization': 'false',
                    'database_path': database_path,
                    'rtabmap_args': rtabmap_args,
                    'odom_args': rtabmap_odom_args,
                    # GUI
                    'rviz': 'false',  # RViz는 별도
                    'rtabmap_viz': 'false',  # OFF — 3D cloud 노드별 누적 → RAM 폭증 → freeze. 경량 RViz2 로 대체
                    # ★ GUI 멈춤 방지: odom을 TF가 아닌 topic 직접 구독
                    'subscribe_odom_info': 'true',
                    'subscribe_odom': 'true',
                }.items(),
            ),
            # ═══════════════════════════════════════════════════════════════════
            # 5) 경량 RViz2 — 매핑 실시간 확인 (2D 맵+TF+궤적, 3D cloud 없음 → freeze 방지)
            # ═══════════════════════════════════════════════════════════════════
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                condition=IfCondition(use_rviz),
                arguments=['-d', os.path.join(
                    get_package_share_directory('slam_mapping'),
                    'config', 'mapping_view.rviz')],
                output='screen',
            ),
            # ═══════════════════════════════════════════════════════════════════
            # 6) 매핑 보강 모니터 (OpenCV 2D) — 맵+위치/방향+loop flash+새노드(초록)
            #    누적 없는 경량 뷰 → rtabmap_viz freeze 와 무관.
            # ═══════════════════════════════════════════════════════════════════
            Node(
                package='slam_mapping',
                executable='mapping_monitor_node',
                name='mapping_monitor',
                condition=IfCondition(LaunchConfiguration('use_monitor')),
                output='screen',
                parameters=[{
                    'map_topic': '/rtabmap/map',
                    'map_frame': 'map',
                    'base_frame': 'base_link',
                    'window_size': 800,
                    'follow_radius_m': 12.0,
                    'render_rate': 10.0,
                }],
            ),
        ]
    )
