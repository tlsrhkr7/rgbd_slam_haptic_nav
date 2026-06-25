# slam_mapping — 시각장애인 보조 웨어러블 내비게이션 시스템

> **캡스톤 디자인 2 (지능로봇공학과)** · Depth 카메라 기반 실내 장애물 회피 및 8방향 햅틱 피드백 내비게이션

Depth 카메라(Intel RealSense D435i)로 실내 위치를 SLAM으로 추정하고, **복도 중심(medial-axis)을 따르는 안전 경로**를 생성해, **8방향 진동 햅틱**으로 시각장애인에게 방향을 안내하는 웨어러블 시스템입니다. 전방 보행자는 YOLO로 검출해 자동 회피·경고합니다.

- **플랫폼:** ROS 2 Humble · NVIDIA Jetson Orin Nano · Arduino Uno
- **팀:** 신동준 (2021042026) · 박성호 (2023042044)

---

## 하드웨어 구성

| 단계 | 장치 | 역할 |
|---|---|---|
| INPUT | RealSense D435i | 전방 RGB + 정렬 depth + IMU(raw) 획득 |
| PROCESS | Jetson Orin Nano | SLAM·경로·검출 연산 |
| CONTROL | Arduino Uno | sector 명령 수신 → 진동 모터 구동 |
| OUTPUT | 8방향 진동 모터 | 진동으로 경로 방향 안내 |
| POWER | 리튬 배터리 | 전 모듈 공급 |

조끼형 하네스에 D435i(가슴 정면) · Jetson·Arduino 모듈 · 배터리 · 햅틱 모터(벨트 라인)를 일체화.

---

## 시스템 아키텍처 (데이터 흐름)

```
RealSense D435i  (RGB 640x480x30 · aligned depth · IMU raw, Best-Effort QoS)
   │  /camera/.../color, aligned_depth, imu
   ├─► imu_filter_madgwick ──► /imu/data (orientation 포함)
   ├─► depth_mask_node      ──► 사람 bbox 영역 depth=0 마스킹 (odometry drift 완화)
   ├─► yolo_person_node     ──► YOLOv11n + ByteTrack 사람 검출 → /yolo/persons_map
   └─► RTAB-Map (localization)  기존 .db 위에서 현재 프레임 매칭
            │  TF map→odom→base_link · /rtabmap/map
            ▼
      localization_viewer_node
        - /rtabmap/map 점유격자 → medial-axis ridge(복도 중심선) 추출
        - robot ↔ 목적지(POI) BFS/Dijkstra 경로, 사람 검출 시 우회
        - /user_path (nav_msgs/Path) 발행 + OpenCV 뷰어(맵·경로·8방향 패널)
            │  /user_path
            ▼
      path_to_haptic_node
        - TF로 사용자 위치·heading → lookahead 앞 점 방향 계산
        - 8방향 sector(0~7) 결정, 전방 사람 시 경고
            │  /haptic_motor_idx (std_msgs/Int32, -1=정지)
            ▼
      serial_haptic_node ──(USB 시리얼)──► Arduino Uno ──► 8방향 진동 모터
```

### 8방향 sector
`0=전(N) · 1=전우(NE) · 2=우(E) · 3=후우(SE) · 4=후(S) · 5=후좌(SW) · 6=좌(W) · 7=전좌(NW)` · `-1=정지`

---

## 노드

| 노드 | 역할 |
|---|---|
| `localization_viewer_node` | medial-axis ridge 경로 생성 + OpenCV 뷰어 + `/user_path` 발행 |
| `path_to_haptic_node` | `/user_path` + TF heading → 8방향 sector → `/haptic_motor_idx` |
| `serial_haptic_node` | `/haptic_motor_idx` → Arduino USB 시리얼 송신 |
| `yolo_person_node` | YOLOv11n + ByteTrack 사람 검출 → `/yolo/persons_map` |
| `depth_mask_node` | 사람 bbox depth 마스킹 (visual odometry drift 완화) |
| `goal_sender_node` | POI yaml → 목적지 발행 (Nav2 연동 시) |
| `mapping_monitor_node` | 매핑 중 2D 맵·위치·loop closure 경량 모니터(OpenCV) |

---

## 주요 토픽

| 토픽 | 타입 | 발행 → 구독 |
|---|---|---|
| `/imu/data` | sensor_msgs/Imu | madgwick → rtabmap/odom |
| `/rtabmap/map` | nav_msgs/OccupancyGrid | rtabmap → viewer |
| `/yolo/persons_map` | visualization_msgs/MarkerArray | yolo → viewer, path_to_haptic |
| `/user_path` | nav_msgs/Path | viewer → path_to_haptic |
| `/haptic_motor_idx` | std_msgs/Int32 | path_to_haptic → serial_haptic |

---

## 빌드

```bash
source /opt/ros/humble/setup.bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select slam_mapping
source install/setup.bash
```

> `setuptools<70` 권장 (egg-link 충돌 회피).

## 실행

**보행 안내 (localization + 경로 + 햅틱):**
```bash
ros2 launch slam_mapping nav_test.launch.py target:="320호"
```

**새 지도 매핑:**
```bash
ros2 launch slam_mapping mapping.launch.py database_path:=$HOME/maps/floor.db
```

- `target` 은 POI yaml(`/home/a/maps/floor4_pois.yaml`)의 이름.
- 런타임 의존: 사전 매핑 `.db`, 시각화 `.yaml/.pgm`, POI `.yaml` (경로는 launch arg로 override).

---

## 데모 시나리오 (E8-7 건물)

`A. E8-7 엘리베이터` (부팅·localization 시작) → 웨어러블 착용 보행 (통로 중심 경로를 8방향 햅틱으로 안내) → `B. E8-7 320호` (목적지 POI 도착·안내 종료).

---

## 디렉토리 구조

```
slam_mapping/
├── launch/        nav_test · mapping · localization · localization_view · nav2_bringup
├── slam_mapping/  ROS 2 노드 (7개)
├── config/        nav2_params.yaml · mapping_view.rviz
├── firmware/      Arduino 진동 모터 펌웨어 (haptic_arduino)
├── package.xml · setup.py
└── README.md
```

## 의존성

ROS 2 Humble · realsense2_camera · rtabmap_launch/slam/odom/sync/viz · imu_filter_madgwick · nav2 (map_server/planner/controller/bt_navigator/lifecycle_manager) · ultralytics(YOLOv11) · OpenCV · NumPy · SciPy

## 펌웨어 (Arduino)

`firmware/haptic_arduino/haptic_arduino.ino` — `/haptic_motor_idx`(sector 0~7, -1=정지)를 USB 시리얼로 수신해 해당 방향 진동 모터를 구동. `test_motors.py`는 모터 단독 테스트 스크립트.
