"""localization_viewer_node — 미니멀 OpenCV 시각화 + RTAB 메타 정보.

차량 내비 모드: 2D 맵 + 현재 위치 + 이동 흔적 + RTAB localization 신호.

토픽:
    /rtabmap/map                  OccupancyGrid              회색조 맵
    /rtabmap/localization_pose    PoseWithCovarianceStamped  공분산 ellipse
    /rtabmap/mapPath              Path                       매핑 trajectory (배경)
    /rtabmap/info                 rtabmap_msgs/Info          loop closure, proc time
    TF map→base_link                                          현재 위치 + heading

확장 (nav 단계 추가 예정):
    /plan, /goal_pose, /haptic_motor_idx

조작:
    q / ESC  종료
    f        follow 토글     r  궤적 reset     +/-  zoom
"""

import math
import os
import threading
import time

import cv2
import numpy as np
import rclpy
try:
    from PIL import Image, ImageDraw, ImageFont
    _KFONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    _KFONT = ImageFont.truetype(_KFONT_PATH, 14) if os.path.exists(_KFONT_PATH) else None
except Exception:
    _KFONT = None
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import Int32
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import Image as RosImage   # PIL Image 와 충돌 방지 alias
try:
    from cv_bridge import CvBridge
    _HAS_CVBRIDGE = True
except ImportError:
    _HAS_CVBRIDGE = False
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray

try:
    from rtabmap_msgs.msg import Info as RtabInfo
    _HAS_RTAB_INFO = True
except ImportError:
    _HAS_RTAB_INFO = False

try:
    from rtabmap_msgs.srv import PublishMap
    _HAS_PUBLISH_MAP_SRV = True
except ImportError:
    _HAS_PUBLISH_MAP_SRV = False

try:
    from rtabmap_msgs.msg import MapGraph
    _HAS_MAPGRAPH = True
except ImportError:
    _HAS_MAPGRAPH = False


class LocalizationViewer(Node):
    def __init__(self):
        super().__init__('localization_viewer')

        # 파라미터
        self.declare_parameter('map_topic', '/rtabmap/map')
        self.declare_parameter('user_frame', 'base_link')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('window_size', 900)
        self.declare_parameter('follow_radius_m', 15.0)
        self.declare_parameter('render_rate', 15.0)

        map_topic = self.get_parameter('map_topic').value
        self.user_frame = self.get_parameter('user_frame').value
        self.map_frame = self.get_parameter('map_frame').value
        self.window_size = self.get_parameter('window_size').value
        self.follow_radius = self.get_parameter('follow_radius_m').value
        render_rate = self.get_parameter('render_rate').value

        # State — 시각화 데이터
        self.map_img = None
        self.map_info = None
        self.trajectory = []                # 사용자 이동 흔적 (px)
        self.mapping_graph_world = []       # 매핑 trajectory (world coords)
        self.cov_2d = None                  # 2x2 위치 공분산
        self.loop_flash_until = 0.0         # loop closure 깜빡임 종료 시각
        self.loop_pulse_until = 0.0         # 사용자 위치 펄스 종료 시각
        self.last_loop_id = 0
        self.last_prox_id = 0
        self.loops_count = 0
        self._localized = False             # 첫 loop closure 전엔 길안내 보류
        # 음성 안내 (mp3) — wait(localize 전 5초마다) / start / 10m / finish
        import shutil as _sh
        self.declare_parameter('voice_dir', '/home/a/Downloads')
        self._voice_dir = self.get_parameter('voice_dir').value
        self._voice_player = ('gst-play-1.0' if _sh.which('gst-play-1.0')
                              else ('mpg123' if _sh.which('mpg123') else None))
        self._voice_procs = []
        self._wait_last = 0.0
        self._voice_start_done = False
        self._voice_10m_done = False
        self._voice_finish_done = False
        self.proximity_count = 0
        self.proc_time_ms = 0.0
        self.ref_id = 0                     # 현재 처리 중인 reference keyframe
        self.last_info_time = 0.0           # /rtabmap/info 마지막 수신 시각
        self.follow_mode = True
        self.show_trajectory = False  # ★ default OFF (t 키로 토글)
        self.show_path = True         # 파란 경로선 표시 (p 키로 토글 — ridge 디버그용)
        # YOLO persons: list of (x_world, y_world, last_seen_ts, track_id)
        self.persons = []
        self.persons_last_msg = 0.0
        self._dist_to_target = None   # 목적지까지 거리(m) — haptic panel 표시
        self._committed_side = {}     # {track_id: +1(좌)/-1(우)} 사람 우회 방향 커밋(히스테리시스)
        # ridge 비동기 계산 — render 스레드 블로킹 방지
        self._ridge_lock = threading.Lock()
        self._ridge_computing = False
        self._ridge_pending_key = None
        self._ridge_done_t = 0.0           # 마지막 ridge 완료 시각 (디바운스용)
        self.declare_parameter('ridge_min_interval', 0.5)  # 1.0→0.5: 회피 반응 ↑ (재계산 잦아짐)
        self._ridge_min_interval = float(
            self.get_parameter('ridge_min_interval').value)
        # 매칭 진단용 (HUD 표시)
        self.last_matched_id = 0          # 마지막 매칭된 keyframe ID
        # (viewer 는 클릭 publish 안 함. POI 등록은 poi_editor_node 별도 툴.)

        # Subscriptions
        qos_map = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(OccupancyGrid, map_topic, self._map_cb, qos_map)
        self.create_subscription(
            PoseWithCovarianceStamped, '/rtabmap/localization_pose',
            self._loc_cb, 10,
        )
        self.create_subscription(Path, '/rtabmap/mapPath', self._graph_cb, 1)
        if _HAS_RTAB_INFO:
            self.create_subscription(RtabInfo, '/rtabmap/info', self._info_cb, 10)
        else:
            self.get_logger().warn('rtabmap_msgs 못 찾음 — loop closure 시각화 비활성')

        # YOLO persons
        self.create_subscription(MarkerArray, '/yolo/persons_map', self._yolo_cb, 10)

        # POI yaml 로드 (목적지 목록)
        self.declare_parameter('poi_file', '/home/a/maps/floor4_pois.yaml')
        self.poi_file = self.get_parameter('poi_file').value
        self.pois = []   # [{name, x, y, yaw, ref_node_id?, offset_x?, offset_y?}]
        self._load_pois()

        # 목표 POI 이름 — viewer 가 POI yaml 에서 이 이름의 좌표 직접 사용.
        # _mapgraph_cb 가 ref_node_id 기반 self.pois[i].x/y 를 graph update 마다
        # 자동 갱신하므로 loop closure 후에도 정확한 텐서 위치 유지.
        self.declare_parameter('target', '텐서')
        self.target = self.get_parameter('target').value

        # mapGraph subscribe — keyframe ID → pose 매핑 (POI 동적 갱신용)
        self.node_poses = {}  # {node_id: (x, y)}
        if _HAS_MAPGRAPH:
            self.create_subscription(MapGraph, '/rtabmap/mapGraph', self._mapgraph_cb, 10)

        # Nav2 /plan 제거 — 우리 ridge polyline + target_link 만 사용 (자체 path).
        self.nav_plan = []
        # 자체 path 를 nav_msgs/Path 로 publish (path_to_haptic 가 subscribe).
        self.user_path_pub = self.create_publisher(Path, '/user_path', 10)

        # 8방향 진동 모터 시각화 — /haptic_motor_idx subscribe
        self.declare_parameter('panel_width', 350)
        self.panel_width = self.get_parameter('panel_width').value
        self.active_motor = -1  # -1 = 미수신 또는 정지
        self.create_subscription(Int32, '/haptic_motor_idx',
                                 self._haptic_cb, 10)

        # 카메라 실시간 영상 — 우측 패널 하단(햅틱 아래)에 표시. 마운트 각도 확인용.
        self.declare_parameter('camera_topic', '/camera/camera/color/image_raw')
        cam_topic = self.get_parameter('camera_topic').value
        self._cam_frame = None
        self._cv_bridge = CvBridge() if _HAS_CVBRIDGE else None
        if self._cv_bridge is not None:
            qos_cam = QoSProfile(depth=1,
                                 reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                 durability=QoSDurabilityPolicy.VOLATILE)
            self.create_subscription(RosImage, cam_topic, self._cam_cb, qos_cam)
        # /goal_pose subscribe 제거 — viewer 가 POI yaml 의 target 좌표 직접 사용.
        # 이유: goal_sender 발행 timing/ref_node_id 갱신 race 로 loop closure 후
        # X 가 엉뚱한 위치로 튀는 문제 회피.

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Render timer
        self.create_timer(1.0 / render_rate, self._render)

        # rtabmap localization 모드는 graph 정적이라 /rtabmap/map 자동 발행 0회.
        # map_always_update=True 는 rtabmap CPU 폭주(maps_update 1.5s/cycle) 야기 →
        # controller_server SIGABRT 死 → /plan 불가. 그래서 viewer 가 *스스로*
        # publish_map service 를 비동기 retry 호출. service 호출 자체는 한 번씩만
        # 일어나므로 grid 매 cycle 재생성 부하 없음 (latched 1회로 충분).
        self._pubmap_client = None
        if _HAS_PUBLISH_MAP_SRV:
            self._pubmap_client = self.create_client(PublishMap, '/rtabmap/publish_map')
            self._pubmap_inflight = False
            self.create_timer(5.0, self._try_publish_map)

        cv2.namedWindow('Localization', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Localization',
                         self.window_size + self.panel_width, self.window_size)

        self.get_logger().info(f'Localization viewer up. Listening: {map_topic}')

    @staticmethod
    def _cluster_polyline(cluster_mask):
        """1-px cluster cell들을 8-conn BFS 로 정렬해 polyline 좌표 list 반환.
        endpoint 부터 farthest cell 까지 BFS path. branch 있으면 longest path."""
        from collections import deque
        ys, xs = np.where(cluster_mask)
        if len(xs) == 0:
            return []
        if len(xs) == 1:
            return [(int(xs[0]), int(ys[0]))]
        H, W = cluster_mask.shape
        # endpoints = 8-conn 이웃 1개 이하인 cell
        endpoints = []
        for y, x in zip(ys.tolist(), xs.tolist()):
            cnt = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and cluster_mask[ny, nx]:
                        cnt += 1
            if cnt <= 1:
                endpoints.append((int(x), int(y)))
        if not endpoints:
            endpoints = [(int(xs[0]), int(ys[0]))]
        start = endpoints[0]
        parent = {start: None}
        q = deque([start])
        last = start
        while q:
            x, y = q.popleft()
            last = (x, y)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H and cluster_mask[ny, nx] and (nx, ny) not in parent:
                        parent[(nx, ny)] = (x, y)
                        q.append((nx, ny))
        path = []
        n = last
        while n is not None:
            path.append(n)
            n = parent[n]
        path.reverse()
        return path

    @staticmethod
    def _resample_polyline(poly, step_px):
        """폴리라인을 step_px 간격으로 리샘플 — 직선 구간에도 균등 노드 유지.
        approxPolyDP 가 직선을 끝점 2개로 붕괴시켜 robot 이 먼 노드로 snap →
        벽 근처 출발 시 벽 따라 비스듬히 이동하는 문제 해결. 끝점은 보존."""
        pts = poly.reshape(-1, 2).astype(np.float64)
        if len(pts) < 2 or step_px < 1:
            return poly
        out = [pts[0]]
        for i in range(len(pts) - 1):
            p, q = pts[i], pts[i + 1]
            seg = q - p
            L = float(math.hypot(seg[0], seg[1]))
            if L >= 1e-6:
                n = int(L // step_px)
                for k in range(1, n + 1):
                    t = (k * step_px) / L
                    if t < 0.999:
                        out.append(p + seg * t)
            out.append(q)
        return np.array(out, dtype=np.int32).reshape(-1, 1, 2)

    def _compute_ridge_bg(self, m, persons_snap, map_info, map_key):
        """백그라운드 스레드에서 ridge 계산 — render 루프 블로킹 없음."""
        try:
            free_mask = (m == 254).astype(np.uint8)
            if not free_mask.any():
                with self._ridge_lock:
                    self._skel_polylines = []
                    self._skel_map_key = map_key
                    self._ridge_computing = False
                return
            not_wall = (m != 0).astype(np.uint8)
            if persons_snap and map_info is not None:
                danger_r_px = max(1, int(0.80 / map_info.resolution))
                now_p = time.time()
                for x_w, y_w, ts, tid in persons_snap:
                    if (now_p - ts) > 3.0:
                        continue
                    pxw, pyw = self._world_to_px(x_w, y_w)
                    cv2.circle(not_wall, (pxw, pyw), danger_r_px, 0, -1)
                    cv2.circle(free_mask, (pxw, pyw), danger_r_px, 0, -1)
            dt = cv2.distanceTransform(not_wall, cv2.DIST_L2, 5)
            dilated = cv2.dilate(dt, np.ones((3, 3), np.uint8))
            ridge = ((dt > 1.5) & (dt >= dilated - 0.5)).astype(np.uint8)
            ridge = ridge & free_mask
            ridge = cv2.morphologyEx(ridge, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            num, labels, stats, _ = cv2.connectedComponentsWithStats(ridge, connectivity=8)
            polylines = []
            endpoints = []
            for i in range(1, num):
                if stats[i, cv2.CC_STAT_AREA] < 18:
                    continue
                cluster = (labels == i)
                path = self._cluster_polyline(cluster)
                if len(path) >= 2:
                    arr = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
                    simp = cv2.approxPolyDP(arr, epsilon=3.0, closed=False)
                    # 직선 구간에도 ~2m 간격 노드 유지 — 벽 근처 출발 시 가까운
                    #   중앙선 노드로 snap → 즉시 중앙 유도(벽 충돌↓). approxPolyDP
                    #   만으론 직선이 끝점 2개로 붕괴 → robot 이 먼 노드로 snap됨.
                    step_px = max(2, int(2.0 / self.map_info.resolution))
                    simp = self._resample_polyline(simp, step_px)
                    polylines.append(simp)
                    pts2 = simp.reshape(-1, 2)
                    endpoints.append(tuple(pts2[0]))
                    endpoints.append(tuple(pts2[-1]))
            GAP_BRIDGE_PX = 100   # 50→100: 끊긴 둘레 ridge 클러스터 연결 (BFS unreachable→직선 fallback 방지)
            bridges = []
            n_eps = len(endpoints)
            for i in range(n_eps):
                cluster_i = i // 2
                for j in range(i + 1, n_eps):
                    if cluster_i == j // 2:
                        continue
                    p1 = np.array(endpoints[i])
                    p2 = np.array(endpoints[j])
                    d = float(np.linalg.norm(p1 - p2))
                    if 0 < d <= GAP_BRIDGE_PX:
                        bridges.append((d, i, j, endpoints[i], endpoints[j]))
            bridges.sort(key=lambda b: b[0])
            used = set()
            for d, i, j, p1, p2 in bridges:
                if i in used or j in used:
                    continue
                used.add(i); used.add(j)
                polylines.append(np.array([p1, p2], np.int32).reshape(-1, 1, 2))
            # dt 는 _build_user_path 에서 사용 — 결과 커밋 전 업데이트
            self._dt = dt
            total_pts = sum(int(p.shape[0]) for p in polylines)
            self.get_logger().info(
                f'[MEDIAL AXIS bg] clusters={len(polylines)}  pts={total_pts}  '
                f'map={m.shape}')
        except Exception as e:
            self.get_logger().warn(f'[MEDIAL AXIS bg] 오류: {e}')
            polylines = getattr(self, '_skel_polylines', []) or []
        finally:
            with self._ridge_lock:
                self._skel_polylines = polylines
                self._skel_map_key = map_key
                self._ridge_computing = False
                self._ridge_done_t = time.time()  # 디바운스 기준 시각

    def _play_voice(self, fname):
        """mp3 비차단 재생 (gst-play-1.0). 콜백 블로킹 없음."""
        if self._voice_player is None:
            return
        import os as _os
        import subprocess as _sp
        path = _os.path.join(self._voice_dir, fname)
        if not _os.path.exists(path):
            self.get_logger().warn(f'voice 파일 없음: {path}', throttle_duration_sec=10.0)
            return
        # 끝난 프로세스 reap (zombie 방지)
        self._voice_procs = [p for p in self._voice_procs if p.poll() is None]
        try:
            if self._voice_player == 'gst-play-1.0':
                cmd = ['gst-play-1.0', '--quiet', path]
            else:
                cmd = ['mpg123', '-q', path]
            self._voice_procs.append(
                _sp.Popen(cmd, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL))
        except Exception as e:
            self.get_logger().warn(f'voice 재생 실패: {e}', throttle_duration_sec=10.0)

    def _voice_update(self):
        """상태 기반 음성 트리거 — render 에서 매 프레임 호출 (값 변화 시 1회만)."""
        now = time.time()
        if not self._localized:
            if now - self._wait_last >= 5.0:   # localize 전 5초마다 wait
                self._wait_last = now
                self._play_voice('wait.mp3')
            return
        if not self._voice_start_done:         # 첫 localize → 길안내 시작
            self._voice_start_done = True
            self._play_voice('start.mp3')
        d = self._dist_to_target
        if d is None:
            return
        if not self._voice_finish_done and d <= 3.0:        # 3m 이내 도착
            self._voice_finish_done = True
            self._play_voice('finish.mp3')
        elif not self._voice_10m_done and 3.0 < d <= 10.0:  # 10m 진입 (1회)
            self._voice_10m_done = True
            self._play_voice('10m.mp3')

    def _cam_cb(self, msg: RosImage):
        try:
            self._cam_frame = self._cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            pass

    def _build_camera_panel(self, w, h):
        """우측 하단 카메라 패널 — 종횡비 유지(letterbox) + 수평/수직 기준선(기울기 확인)."""
        panel = np.zeros((h, w, 3), dtype=np.uint8)
        frame = self._cam_frame
        if frame is None:
            self._put_text(panel, 'no camera', (10, h // 2))
        else:
            fh, fw = frame.shape[:2]
            scale = min(w / fw, h / fh)
            nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
            resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            x0, y0 = (w - nw) // 2, (h - nh) // 2
            panel[y0:y0 + nh, x0:x0 + nw] = resized
            # 기준선: 중앙 수평/수직 (마운트 좌우 기울기·회전 눈으로 판단)
            cv2.line(panel, (0, h // 2), (w, h // 2), (0, 255, 0), 1)
            cv2.line(panel, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
        self._put_text(panel, 'CAMERA', (8, 18))
        return panel

    def _haptic_cb(self, msg: Int32):
        """path_to_haptic 가 보내는 8방향 sector index (-1 = 정지)."""
        self.active_motor = int(msg.data)

    def _draw_haptic_panel(self, panel, persons_active=False, pulse_on=True,
                           dist_m=None):
        """오른쪽 panel 에 8방향 진동 모터 시각화. 활성 모터 = 빨간 fill.
        중앙 robot 아이콘 + 위쪽 화살표 (사용자 전방 = 항상 위).
        persons_active=True 면 PERSON DETECTED 표시 + 활성 모터가 pulse_on 에 따라 점멸.
        dist_m: 목적지까지 거리(m) — 상단에 크게 표시."""
        h, w = panel.shape[:2]
        cx, cy = w // 2, h // 2
        R = min(w, h) // 3   # 8 motor 배치 반경
        motor_r = 30
        # 배경 외곽선 + 헤더 텍스트
        cv2.rectangle(panel, (2, 2), (w - 3, h - 3), (50, 50, 50), 1)
        self._put_text(panel, 'HAPTIC 8-DIR', (10, 24))
        sector_txt = f'sector: {self.active_motor}'
        if self.active_motor == -1:
            sector_txt += '  (STOP)'
        elif persons_active:
            sector_txt += '  (PULSE)'
        self._put_text(panel, sector_txt, (10, 48))
        # 목적지 거리 — 큰 글씨로 강조 (도착 임박 <1m 면 초록)
        dist_txt = 'DEST: -- m' if dist_m is None else f'DEST: {dist_m:.1f} m'
        dcol = (0, 230, 0) if (dist_m is not None and dist_m < 1.0) else (0, 220, 255)
        cv2.putText(panel, dist_txt, (10, 86),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, dcol, 2, cv2.LINE_AA)
        # 사람 감지 경고 배너 (panel 하단)
        if persons_active:
            band_y = h - 56
            warn_col = (0, 0, 230) if pulse_on else (0, 0, 110)  # 점멸
            cv2.rectangle(panel, (8, band_y), (w - 9, h - 12), warn_col, -1)
            cv2.rectangle(panel, (8, band_y), (w - 9, h - 12), (255, 255, 255), 2)
            (tw, _), _ = cv2.getTextSize('PERSON DETECTED',
                                         cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
            cv2.putText(panel, 'PERSON DETECTED',
                        ((w - tw) // 2, band_y + 32),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        # 중앙 robot 아이콘 + 전방 화살표
        cv2.circle(panel, (cx, cy), 16, (60, 60, 220), -1, cv2.LINE_AA)
        cv2.circle(panel, (cx, cy), 16, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.arrowedLine(panel, (cx, cy), (cx, cy - 50),
                        (60, 60, 220), 4, cv2.LINE_AA, tipLength=0.35)
        # 8 motor — sector 0=전(위), 1=전우, 2=우, 3=후우, 4=후, 5=후좌, 6=좌, 7=전좌
        labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        # 사람 감지 시 펄스 OFF phase 면 활성 모터도 꺼진 것처럼 표시
        lit = self.active_motor if (pulse_on or not persons_active) else -1
        for i in range(8):
            angle = -math.pi / 2 + i * math.pi / 4   # 0=-pi/2(위), 시계방향
            mx = cx + int(R * math.cos(angle))
            my = cy + int(R * math.sin(angle))
            if i == lit:
                cv2.circle(panel, (mx, my), motor_r, (0, 0, 240), -1, cv2.LINE_AA)
                cv2.circle(panel, (mx, my), motor_r + 5, (0, 0, 240), 2, cv2.LINE_AA)
            else:
                cv2.circle(panel, (mx, my), motor_r, (70, 70, 70), -1, cv2.LINE_AA)
            cv2.circle(panel, (mx, my), motor_r, (255, 255, 255), 2, cv2.LINE_AA)
            # 숫자 + 방향 라벨
            self._put_text(panel, str(i), (mx - 6, my + 6))
            self._put_text(panel, labels[i],
                           (mx - 12, my + motor_r + 18))

    def _build_user_path(self, polylines, robot_xy, target_xy):
        """robot ↔ TARGET 한 줄 path. graph + path 모두 cache (실시간성 강화).
        - graph: id(polylines) 동일하면 재사용
        - path: robot/target 양자화 좌표 (5px 격자) 동일하면 재사용
        """
        if not polylines:
            return None
        # 동적 장애물 원 (사람 30cm) — graph node 차단용
        danger_circles = []  # [(tid, p_px, p_py, r_px), ...]
        danger_r_px = 1
        if self.map_img is not None:
            danger_r_px = max(1, int(0.80 / self.map_info.resolution))
            now_ts = time.time()
            for x_w, y_w, ts, tid in self.persons:
                if (now_ts - ts) > 3.0:
                    continue
                pxw, pyw = self._world_to_px(x_w, y_w)
                danger_circles.append((tid, pxw, pyw, danger_r_px))

        # graph cache key — polyline 만 (사람 원 차단 X, path 후처리에서 우회)
        graph_key = id(polylines)
        if getattr(self, '_graph_key', None) != graph_key:
            nodes = []
            node_to_idx = {}
            adj = {}
            for poly in polylines:
                pts = poly.reshape(-1, 2)
                prev_n = None
                for p in pts:
                    key = (int(p[0]), int(p[1]))
                    if key not in node_to_idx:
                        node_to_idx[key] = len(nodes)
                        nodes.append(key)
                    n = node_to_idx[key]
                    if prev_n is not None and prev_n != n:
                        adj.setdefault(prev_n, set()).add(n)
                        adj.setdefault(n, set()).add(prev_n)
                    prev_n = n
            if not nodes:
                return None
            self._graph_nodes = nodes
            self._graph_adj = adj
            self._graph_arr = np.array(nodes, dtype=np.float64)
            self._graph_key = graph_key
        nodes = self._graph_nodes
        adj = self._graph_adj
        arr = self._graph_arr

        rx, ry = robot_xy
        tx, ty = target_xy
        from collections import deque

        # ── 사이드 커밋 + 히스테리시스 ─────────────────────────────────────────
        #   사람 정중앙 fork 에서 좌/우 경로가 매 프레임 뒤집히는 펄럭임 방지.
        #   사람별(track_id) 우회 방향을 한 번 정하고 고수 → 그 쪽이 막힐 때만 전환.
        ux, uy = (tx - rx), (ty - ry)
        un = math.hypot(ux, uy)
        ux, uy = (ux / un, uy / un) if un > 1e-6 else (1.0, 0.0)
        BLOCK_R = danger_r_px * 1.6
        active_tids = {tid for tid, _, _, _ in danger_circles}
        self._committed_side = {k: v for k, v in self._committed_side.items()
                                if k in active_tids}

        def _make_blocked(flip=False):
            blocked = np.zeros(len(nodes), dtype=bool)
            for tid, cxp, cyp, rp in danger_circles:
                near = ((arr[:, 0] - cxp) ** 2 + (arr[:, 1] - cyp) ** 2) <= (BLOCK_R * BLOCK_R)
                if not near.any():
                    continue
                cross = ux * (arr[:, 1] - cyp) - uy * (arr[:, 0] - cxp)  # >0 = 좌
                s = self._committed_side.get(tid)
                if s is None:   # 첫 결정: 노드 더 많은(넓은) 쪽으로 커밋
                    s = 1.0 if (near & (cross >= 0)).sum() >= (near & (cross < 0)).sum() else -1.0
                    self._committed_side[tid] = s
                if flip:
                    s = -s
                    self._committed_side[tid] = s
                # 커밋 안 한 쪽(near) 노드 차단
                blocked |= near & ((cross >= 0) != (s > 0))
            return blocked

        def _snap_bfs(blocked):
            dR = np.hypot(arr[:, 0] - rx, arr[:, 1] - ry)
            dT = np.hypot(arr[:, 0] - tx, arr[:, 1] - ty)
            dRb = np.where(blocked, np.inf, dR)
            dTb = np.where(blocked, np.inf, dT)
            if not np.isfinite(dRb).any():
                dRb = dR
            if not np.isfinite(dTb).any():
                dTb = dT
            # snap_R = robot 최근접 노드 (forward_mask 제거).
            #   직선거리 forward_mask 는 ㄷ자/평행 복도에서 robot 근처 노드를 탈락시켜
            #   엉뚱한 노드로 snap → robot→snap_R 연결선이 벽 관통. 방향은 Dijkstra 가 처리.
            iR = int(np.argmin(dRb))
            iT = int(np.argmin(dTb))
            # Dijkstra (유클리드 가중치) — hop 수가 아닌 실제 거리 최소 → 원 빙 도는 경로 제거
            import heapq
            dist = {iR: 0.0}
            parent = {iR: None}
            pq = [(0.0, iR)]
            while pq:
                d, n = heapq.heappop(pq)
                if n == iT:
                    break
                if d > dist.get(n, float('inf')):
                    continue
                axn, ayn = arr[n, 0], arr[n, 1]
                for nb in adj.get(n, []):
                    if blocked[nb]:
                        continue
                    w = math.hypot(axn - arr[nb, 0], ayn - arr[nb, 1])
                    nd = d + w
                    if nd < dist.get(nb, float('inf')):
                        dist[nb] = nd
                        parent[nb] = n
                        heapq.heappush(pq, (nd, nb))
            if iT not in parent:
                return None
            mp = []
            n = iT
            while n is not None:
                mp.append(nodes[n]); n = parent[n]
            mp.reverse()
            return mp

        mid_path = _snap_bfs(_make_blocked(flip=False))      # 1차: 커밋된 쪽
        if mid_path is None and danger_circles:
            mid_path = _snap_bfs(_make_blocked(flip=True))    # 커밋 쪽 막힘 → 반대로 전환
        if mid_path is None:
            mid_path = _snap_bfs(np.zeros(len(nodes), dtype=bool))  # 차단 없이 (최후)
        if mid_path is None:
            iR = int(np.argmin(np.hypot(arr[:, 0] - rx, arr[:, 1] - ry)))
            iT = int(np.argmin(np.hypot(arr[:, 0] - tx, arr[:, 1] - ty)))
            mid_path = [nodes[iR], nodes[iT]]

        # 선두 backward 노드 제거 — robot 이 이미 지난 노드(뒤로 무는 hook) 잘라냄.
        #   robot 이 mid_path[0]보다 mid_path[1]에 더 가까우면 [0]은 지나친 것 → drop.
        while len(mid_path) >= 2:
            d01 = math.hypot(mid_path[0][0] - mid_path[1][0],
                             mid_path[0][1] - mid_path[1][1])
            d_r1 = math.hypot(rx - mid_path[1][0], ry - mid_path[1][1])
            if d_r1 < d01:
                mid_path.pop(0)
            else:
                break

        full = [(int(rx), int(ry))] + mid_path + [(int(tx), int(ty))]

        # 경로 스무딩 — 사람(danger_circles) 있을 때만 적용. 없으면 원본 ridge path 그대로.
        #   _los_clear/_free 는 아래 최종 벽 검증에서 항상 사용 (사람 유무 무관).
        if len(full) > 2 and self.map_img is not None:
            H, W = self.map_img.shape[:2]
            CLEAR_PX = 4   # 벽 안전거리(~20cm). 벽관통 잦으면↑, 좁은통로 못지나면↓

            def _free(x, y):
                return 0 <= x < W and 0 <= y < H and self.map_img[y, x] == 254

            def _los_clear(p0, p1):
                x0, y0 = p0
                x1, y1 = p1
                L = math.hypot(x1 - x0, y1 - y0)
                if L < 1e-6:
                    return True
                n = max(2, int(L))
                pxn, pyn = -(y1 - y0) / L, (x1 - x0) / L   # 선에 수직 단위벡터
                for k in range(n + 1):
                    t = k / n
                    cx_ = x0 + (x1 - x0) * t
                    cy_ = y0 + (y1 - y0) * t
                    # 중심 + 좌우 CLEAR_PX 까지 free (벽 스침 차단)
                    for s in (0, CLEAR_PX, -CLEAR_PX):
                        if not _free(int(round(cx_ + pxn * s)),
                                     int(round(cy_ + pyn * s))):
                            return False
                    for _t, cxp, cyp, rp in danger_circles:
                        if (cx_ - cxp) ** 2 + (cy_ - cyp) ** 2 < rp * rp:
                            return False
                return True

            # 사람 있을 때만 string-pulling 으로 회피경로 지그재그 단축.
            #   사람 없으면 원본 ridge path 그대로 (중앙선 유지).
            if danger_circles:
                sm = [full[0]]
                i = 0
                while i < len(full) - 1:
                    j = len(full) - 1
                    while j > i + 1 and not _los_clear(full[i], full[j]):
                        j -= 1
                    sm.append(full[j])
                    i = j
                full = sm

            # 최종 벽 검증(경로 중간 끊김) 제거 — 사용자 요청. ridge 기반이라 경로가
            #   벽을 통과할 일이 거의 없다는 판단. dist_to_target 도 full 경로 길이로
            #   측정되므로 더이상 잘린 길이 → 10m 플래그 오발 안 함.
            #   (복원하려면 백업 .bak_* 또는 이 주석 참고해 validated 루프 되살리기.)

        self._path_cache = full
        self._path_key = (graph_key, int(rx) // 5, int(ry) // 5, int(tx), int(ty),
                          tuple(sorted(self._committed_side.items())))
        return full

    @staticmethod
    def _corner_cut(path, cut_dist=15):
        """각 corner 의 prev/next 방향으로 cut_dist 안쪽 두 점을 대각선으로 잇기.
        segment 길이의 절반 초과 안 함. 결과 길이 = len(path) + corner 수."""
        if len(path) < 3:
            return path
        result = [path[0]]
        for i in range(1, len(path) - 1):
            p_prev = np.array(path[i - 1], dtype=np.float64)
            p_curr = np.array(path[i], dtype=np.float64)
            p_next = np.array(path[i + 1], dtype=np.float64)
            v_in = p_prev - p_curr
            v_out = p_next - p_curr
            len_in = float(np.linalg.norm(v_in))
            len_out = float(np.linalg.norm(v_out))
            d = min(cut_dist, len_in / 2.0, len_out / 2.0)
            if d <= 1:
                result.append((int(p_curr[0]), int(p_curr[1])))
                continue
            cut_a = p_curr + v_in / len_in * d
            cut_b = p_curr + v_out / len_out * d
            result.append((int(cut_a[0]), int(cut_a[1])))
            result.append((int(cut_b[0]), int(cut_b[1])))
        result.append(path[-1])
        return result

    def _target_perpendicular_cut(self, polylines, gx, gy, dt):
        """[DEPRECATED] _build_user_path 로 대체됨. 호환 위해 stub 유지."""
        H, W = dt.shape
        if not (0 <= gx < W and 0 <= gy < H):
            return polylines, None
        # 모든 polyline cell 중 TARGET 에 가장 가까운 point 찾음 → 직선 연결.
        # unknown 깊은 곳에서도 무관 (gradient 의존 X).
        all_pts = []  # (poly_idx, seg_idx, x, y)
        for pi, poly in enumerate(polylines):
            pts = poly.reshape(-1, 2)
            for si, p in enumerate(pts):
                all_pts.append((pi, si, int(p[0]), int(p[1])))
        if not all_pts:
            return polylines, None
        arr = np.array([[p[2], p[3]] for p in all_pts], dtype=np.float64)
        dists = np.hypot(arr[:, 0] - gx, arr[:, 1] - gy)
        i_min = int(np.argmin(dists))
        pi, si, ix_p, iy_p = all_pts[i_min]
        best = (float(dists[i_min]), pi, si, float(ix_p), float(iy_p))

        d_t, pi, si, ix, iy = best
        ix_i, iy_i = int(ix), int(iy)
        # 만난 polyline 통째로 제거 (양쪽 다 지움). TARGET → nearest point 직선만 남김.
        new_polylines = list(polylines)
        new_polylines.pop(pi)
        target_link = ((int(gx), int(gy)), (ix_i, iy_i))
        return new_polylines, target_link

    @staticmethod
    def _seg_intersect(p1, p2, p3, p4):
        """Line segment p1-p2 vs p3-p4 intersection. Returns (x, y) or None."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-9:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        if 0 <= t <= 1 and 0 <= u <= 1:
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        return None

    def _try_publish_map(self):
        """rtabmap publish_map service 비동기 retry — grid 받기 전까지 5초마다."""
        if self.map_img is not None:
            return  # 이미 grid 받음 → 더 호출 불필요
        if self._pubmap_client is None or self._pubmap_inflight:
            return
        if not self._pubmap_client.service_is_ready():
            return  # rtabmap 아직 ready 안 됨 → 다음 tick에 재시도
        req = PublishMap.Request()
        req.global_map = True
        req.optimized = True
        req.graph_only = False
        self._pubmap_inflight = True
        fut = self._pubmap_client.call_async(req)

        def _done(_f):
            self._pubmap_inflight = False
            self.get_logger().info('publish_map service 응답 수신 — grid 메시지 곧 옴')
        fut.add_done_callback(_done)
        self.get_logger().info('publish_map service 호출 (grid 발행 요청)')

    # ─────────────── Callbacks ───────────────

    def _map_cb(self, msg: OccupancyGrid):
        # origin lock 제거 — mapPath/TF 와 좌표계 동기화 (graph 변동 시 같이 따라감).
        # 캔버스 흔들림 vs 좌표계 정합 trade-off 에서 정합 선택 (사용자 요청 2026-05-20).
        w, h = msg.info.width, msg.info.height
        data = np.array(msg.data, dtype=np.int8).reshape(h, w)

        # Trinary 변환: 0 free → 254, 100 occupied → 0 (65%+), 그 외 → 200 (unknown)
        img = np.full((h, w), 200, dtype=np.uint8)
        img[data == 0] = 254
        img[(data >= 65) & (data <= 100)] = 0

        self.map_img = cv2.flip(img, 0)
        self.map_info = msg.info
        self._map_bgr_cache = None  # 맵 갱신 → BGR 캐시 무효화

    def _loc_cb(self, msg: PoseWithCovarianceStamped):
        c = msg.pose.covariance  # 6x6 row-major
        self.cov_2d = np.array([[c[0], c[1]], [c[6], c[7]]])

    def _graph_cb(self, msg: Path):
        self.mapping_graph_world = [
            (p.pose.position.x, p.pose.position.y) for p in msg.poses
        ]

    def _mapgraph_cb(self, msg):
        """RTAB 그래프 노드 위치 갱신 + POI 좌표 동적 재계산 + 자동 마이그레이션."""
        self.node_poses = {
            msg.poses_id[i]: (msg.poses[i].position.x, msg.poses[i].position.y)
            for i in range(min(len(msg.poses_id), len(msg.poses)))
        }
        if not self.node_poses:
            return

        # 자동 마이그레이션: ref_node_id 없는 POI → nearest keyframe 찾아서 등록 + yaml save
        migrated = False
        for p in self.pois:
            if 'ref_node_id' in p:
                # 이미 변환됨 → 좌표 갱신만
                ref = p['ref_node_id']
                if ref in self.node_poses:
                    nx, ny = self.node_poses[ref]
                    p['x'] = nx + p.get('offset_x', 0.0)
                    p['y'] = ny + p.get('offset_y', 0.0)
            else:
                # 마이그레이션 — nearest node 찾음
                wx, wy = p.get('x'), p.get('y')
                if wx is None or wy is None:
                    continue
                min_d2 = float('inf')
                nid = None
                for k, (nx, ny) in self.node_poses.items():
                    d2 = (nx - wx) ** 2 + (ny - wy) ** 2
                    if d2 < min_d2:
                        min_d2 = d2
                        nid = k
                if nid is not None:
                    nx, ny = self.node_poses[nid]
                    p['ref_node_id'] = int(nid)
                    p['offset_x'] = float(wx - nx)
                    p['offset_y'] = float(wy - ny)
                    migrated = True
                    self.get_logger().info(
                        f"[migrate] '{p.get('name')}' → ref=#{nid} "
                        f"off=({p['offset_x']:.2f}, {p['offset_y']:.2f})")
        if migrated:
            self._save_pois()

    def _save_pois(self):
        try:
            import yaml as _yaml
            # 백업 한 번만
            bak = self.poi_file + '.bak'
            import os as _os, shutil as _shutil
            if _os.path.exists(self.poi_file) and not _os.path.exists(bak):
                _shutil.copy(self.poi_file, bak)
                self.get_logger().info(f'[backup] {bak}')
            with open(self.poi_file, 'w') as f:
                _yaml.safe_dump(
                    {'pois': self.pois}, f, allow_unicode=True, sort_keys=False)
            self.get_logger().info(f'[save] {len(self.pois)} POIs → {self.poi_file}')
        except Exception as e:
            self.get_logger().warn(f'[save] 실패: {e}')

    def _yolo_cb(self, msg: MarkerArray):
        now = time.time()
        MERGE_M = 0.35   # 이 거리 내 새 검출 = 같은 사람 (ByteTrack ID switch 흡수)
        EMA_A = 0.5      # 위치 평활 계수 (정지 떨림 ↓)
        SNAP_M = 0.5     # 이보다 크게 움직이면 실제 이동 → 평활 X, 즉시 반영(지연 방지)
        touched = set()  # 이번 메시지에서 갱신된 index (한 메시지 내 두 마커 병합 방지)

        def _ema(ox, oy, nx, ny):
            # 작은 변화(정지 떨림)만 평활, 큰 변화(실제 이동)는 snap → 추종 지연 없음
            if math.hypot(nx - ox, ny - oy) > SNAP_M:
                return nx, ny
            return ox + EMA_A * (nx - ox), oy + EMA_A * (ny - oy)

        for m in msg.markers:
            if m.action == 2:  # DELETE
                self.persons = [(x, y, ts, t) for x, y, ts, t in self.persons if t != m.id]
                continue
            mx, my, tid = m.pose.position.x, m.pose.position.y, m.id
            # 1) 같은 track_id 갱신 (EMA 평활)
            updated = False
            for i, (x, y, ts, t) in enumerate(self.persons):
                if t == tid:
                    ex, ey = _ema(x, y, mx, my)
                    self.persons[i] = (ex, ey, now, tid)
                    touched.add(i)
                    updated = True
                    break
            if updated:
                continue
            # 2) 위치 기반 병합 — ID 바뀌어도 0.35m 내 기존 사람이면 그걸 갱신(기존 tid 유지).
            #    → ID switch 유령 마커 제거 + 사이드 커밋(track_id 기반) 유지.
            best_i, best_d = -1, MERGE_M
            for i, (x, y, ts, t) in enumerate(self.persons):
                if i in touched:
                    continue
                d = math.hypot(mx - x, my - y)
                if d < best_d:
                    best_d, best_i = d, i
            if best_i >= 0:
                ox, oy, _, kept_tid = self.persons[best_i]
                ex, ey = _ema(ox, oy, mx, my)
                self.persons[best_i] = (ex, ey, now, kept_tid)  # 평활 위치, 기존 tid 유지
                touched.add(best_i)
                continue
            # 3) 진짜 새 사람
            self.persons.append((mx, my, now, tid))
            touched.add(len(self.persons) - 1)
        # ghost 정리 — 6초 이상 갱신 없는 track 제거
        self.persons = [(x, y, ts, t) for x, y, ts, t in self.persons if (now - ts) <= 6.0]
        self.persons_last_msg = now

    def _info_cb(self, msg):
        # rtabmap_msgs/Info: snake_case in ROS2
        lid = getattr(msg, 'loop_closure_id', 0)
        pid = getattr(msg, 'proximity_detection_id', 0)
        matched = lid if lid > 0 else pid
        if matched > 0:
            self.last_matched_id = matched
        # 매칭 발생 시마다 flash (cooldown 1.5초). 같은 ID 반복이라도 1.5s 후 다시 flash.
        # = 매칭 활발 = 위치 정확 보정 중 = 사용자에게 시각 피드백
        now_ts = time.time()
        if matched > 0:
            if not self._localized:
                self._localized = True   # 첫 loop flash → 이제부터 길안내 시작
                self.get_logger().info('[LOCALIZED] 첫 loop closure — 길안내 시작')
            if now_ts > self.loop_flash_until - 0.3:  # 직전 flash 거의 끝났으면
                self.loop_flash_until = now_ts + 1.0
        if lid > 0 and lid != self.last_loop_id:
            self.last_loop_id = lid
            self.loops_count += 1
        # ref_id 갱신
        rid = getattr(msg, 'ref_id', 0)
        if rid > 0:
            self.ref_id = rid
        self.proc_time_ms = getattr(msg, 'time_total', 0.0) * 1000.0

    # ─────────────── Helpers ───────────────


    def _load_pois(self):
        """POI yaml 로드 (목적지 마커 표시용). 파일 없으면 빈 list."""
        import os
        try:
            import yaml
        except ImportError:
            self.get_logger().warn('PyYAML 없음 — POI 표시 비활성')
            return
        if not os.path.exists(self.poi_file):
            self.get_logger().info(f'POI file 없음: {self.poi_file}')
            return
        try:
            with open(self.poi_file, 'r') as f:
                data = yaml.safe_load(f) or {}
            self.pois = data.get('pois', [])
            self.get_logger().info(f'POI 로드: {len(self.pois)}개')
        except Exception as e:
            self.get_logger().warn(f'POI 로드 실패: {e}')

    def _world_to_px(self, x: float, y: float):
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        h = self.map_img.shape[0]
        px = int(round((x - ox) / res))
        py = h - 1 - int(round((y - oy) / res))
        return px, py

    def _px_to_world(self, px: int, py: int):
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        h = self.map_img.shape[0]
        x = ox + px * res
        y = oy + (h - 1 - py) * res
        return x, y

    def _get_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.user_frame,
                rclpy.time.Time(), timeout=Duration(seconds=0.5),
            )
        except Exception:
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return t.x, t.y, math.atan2(siny, cosy)

    # ─────────────── Render ───────────────

    def _render(self):
        if self.map_img is None or self.map_info is None:
            return

        if getattr(self, '_map_bgr_cache', None) is None:
            self._map_bgr_cache = cv2.cvtColor(self.map_img, cv2.COLOR_GRAY2BGR)
        canvas = self._map_bgr_cache.copy()

        # 1) 매핑 graph 배경 — 제거 (렉 감소: 439노드 좌표변환 + line 반복 스킵)

        pose = self._get_pose()
        px = py = None

        if pose is not None:
            x, y, yaw = pose
            px, py = self._world_to_px(x, y)

            # 2) 사용자 궤적 누적
            if not self.trajectory or (px, py) != self.trajectory[-1]:
                self.trajectory.append((px, py))
                if len(self.trajectory) > 5000:
                    self.trajectory.pop(0)

            if self.show_trajectory and len(self.trajectory) > 1:
                pts_arr = np.array(self.trajectory, np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [pts_arr], False, (0, 220, 0), 2, cv2.LINE_AA)

            # 3) Covariance ellipse (위치 신뢰도)
            # covariance ellipse 제거 (eigenvalue 분해 매 렌더 → 렉)

            # 4) 빨간 원 + heading 화살표
            cv2.circle(canvas, (px, py), 10, (0, 0, 255), -1)
            cv2.circle(canvas, (px, py), 10, (255, 255, 255), 2)
            arrow_len = 35
            end_px = px + int(arrow_len * math.cos(yaw))
            end_py = py - int(arrow_len * math.sin(yaw))
            cv2.arrowedLine(canvas, (px, py), (end_px, end_py),
                            (0, 0, 255), 3, tipLength=0.35, line_type=cv2.LINE_AA)

        # 5a) POI 마커 — 원은 매 render cv2 (빠름), 이름은 매 1초마다만 PIL (CPU 절약)
        if self.pois:
            poi_pxs = []
            for p in self.pois:
                poi_px, poi_py = self._world_to_px(p['x'], p['y'])
                cv2.circle(canvas, (poi_px, poi_py), 10, (0, 200, 220), -1, cv2.LINE_AA)
                cv2.circle(canvas, (poi_px, poi_py), 10, (40, 40, 40), 2, cv2.LINE_AA)
                poi_pxs.append((poi_px, poi_py, p.get('name', '')))
            # PIL 변환은 BGR↔RGB 전체 변환이라 비싸다 → 매 1초만 갱신
            now_ts = time.time()
            need_refresh = (not hasattr(self, '_poi_label_cache')
                            or (now_ts - getattr(self, '_poi_label_ts', 0)) > 1.0
                            or self._poi_label_cache is None
                            or self._poi_label_cache.shape != canvas.shape)
            if _KFONT is not None and need_refresh:
                # 빈 RGBA 오버레이 만들어 텍스트만 그림 → 매 render alpha-blend 로 가벼움
                from PIL import Image as _PI
                overlay = _PI.new('RGBA', (canvas.shape[1], canvas.shape[0]), (0, 0, 0, 0))
                d = ImageDraw.Draw(overlay)
                for px_, py_, name in poi_pxs:
                    d.text((px_ + 13, py_ - 9), name,
                           font=_KFONT, fill=(0, 0, 0, 255), stroke_width=2,
                           stroke_fill=(255, 255, 255, 255))
                self._poi_label_cache = np.array(overlay)  # RGBA H×W×4
                self._poi_label_ts = now_ts
            if _KFONT is not None and getattr(self, '_poi_label_cache', None) is not None:
                # mask-based copy (alpha blend 보다 수 배 빠름)
                ov = self._poi_label_cache
                if ov.shape[:2] == canvas.shape[:2]:
                    mask = ov[:, :, 3] > 0  # alpha channel binary mask
                    canvas[mask] = ov[mask][:, [2, 1, 0]]  # RGB → BGR copy
            elif _KFONT is None:
                for px_, py_, name in poi_pxs:
                    self._put_text(canvas, name, (px_ + 12, py_ - 4))

        # 5a2) Robot → Target POI 굵은 초록 점선 (직선).
        #    target POI 좌표는 self.pois 에서 직접 찾음 (loop closure 후
        #    _mapgraph_cb 가 ref_node_id 기반 자동 갱신 → 항상 정확 위치).
        target_pos = None
        self._dist_to_target = None   # 이번 프레임 거리(m) — panel 표시용 (매 프레임 갱신)
        for _p in self.pois:
            if _p.get('name') == self.target:
                target_pos = (_p['x'], _p['y'])
                break

        # 5a2) Medial axis — 사람 carving 포함 (복도 폭 안에서 ridge 가 휘어 회피).
        #   사람은 위치 노이즈로 자주 바뀌므로 person 유발 재계산은 디바운스:
        #   맵 자체(shape/sum) 변경은 즉시, 사람만 변경이면 min_interval 마다만 재계산.
        #   → 끊김이 "사람 움직일 때마다 연속" → "최대 N초마다 1회 짧은 hitch" 로 완화.
        if self.map_img is not None:
            m = self.map_img
            persons_key = tuple(
                (int(x * 2), int(y * 2))
                for x, y, ts, _ in self.persons if (time.time() - ts) <= 3.0
            )
            static_key = (m.shape, int(m.sum()))
            map_key = (static_key, persons_key)
            cur = getattr(self, '_skel_map_key', None)
            static_changed = (cur is None) or (cur[0] != static_key)
            now_r = time.time()
            since = now_r - getattr(self, '_ridge_done_t', 0.0)
            allow = static_changed or (since >= self._ridge_min_interval)
            if cur != map_key and allow:
                with self._ridge_lock:
                    if not self._ridge_computing and self._ridge_pending_key != map_key:
                        self._ridge_pending_key = map_key
                        self._ridge_computing = True
                        threading.Thread(
                            target=self._compute_ridge_bg,
                            args=(m.copy(), list(self.persons), self.map_info, map_key),
                            daemon=True,
                        ).start()
            polylines = getattr(self, '_skel_polylines', None)
            if polylines:
                # (디버그 ridge 드로잉 제거 — 경량화. 계산/BFS/햅틱은 유지)
                # robot → TARGET 한 줄 path — 첫 loop closure(localized) 후에만 안내 시작
                if (self._localized and target_pos is not None
                        and pose is not None and px is not None):
                    gx_t, gy_t = self._world_to_px(*target_pos)
                    user_path = self._build_user_path(
                        polylines, (px, py), (gx_t, gy_t))
                    if user_path is not None and len(user_path) >= 2:
                        if self.show_path:   # p 키로 토글 (그리기만; 계산·publish 는 항상)
                            path_arr = np.array(user_path, np.int32).reshape(-1, 1, 2)
                            cv2.polylines(canvas, [path_arr], False,
                                          (255, 200, 0), 6, cv2.LINE_AA)  # 청록 굵게
                        # 경로 길이(m) = 픽셀 세그먼트 합 × resolution → 실제 걸어야 할 거리
                        seg = np.diff(np.array(user_path, np.float64), axis=0)
                        path_px = float(np.hypot(seg[:, 0], seg[:, 1]).sum())
                        self._dist_to_target = path_px * self.map_info.resolution
                        # /user_path 토픽으로 publish — path_to_haptic 가 받아 진동 계산
                        msg = Path()
                        msg.header.frame_id = 'map'
                        msg.header.stamp = self.get_clock().now().to_msg()
                        for px_, py_ in user_path:
                            ps = PoseStamped()
                            ps.header = msg.header
                            wx, wy = self._px_to_world(int(px_), int(py_))
                            ps.pose.position.x = float(wx)
                            ps.pose.position.y = float(wy)
                            ps.pose.orientation.w = 1.0
                            msg.poses.append(ps)
                        self.user_path_pub.publish(msg)


        # 5a3) Target POI 강조 — 큰 빨간 fill circle + 'TARGET' 라벨.
        #    POI 일반 노란 원과 구분. X 형태 제거 (사용자 요구).
        if target_pos is not None:
            gx, gy = self._world_to_px(*target_pos)
            cv2.circle(canvas, (gx, gy), 10, (0, 0, 220), -1, cv2.LINE_AA)
            cv2.circle(canvas, (gx, gy), 10, (255, 255, 255), 2, cv2.LINE_AA)
            self._put_text(canvas, f'TARGET ({self.target})', (gx + 22, gy - 14))
            # 경로 거리 못 구했으면 직선거리(m)로 fallback
            if self._dist_to_target is None and pose is not None:
                self._dist_to_target = math.hypot(
                    target_pos[0] - pose[0], target_pos[1] - pose[1])

        # 5) Follow 모드 crop
        cropped = False
        crop_x0 = crop_y0 = 0
        if self.follow_mode and px is not None:
            half = int(self.follow_radius / self.map_info.resolution)
            x0, y0 = px - half, py - half
            x1, y1 = px + half, py + half
            H, W = canvas.shape[:2]
            pad_top = max(0, -y0)
            pad_left = max(0, -x0)
            pad_bot = max(0, y1 - H)
            pad_right = max(0, x1 - W)
            if any([pad_top, pad_left, pad_bot, pad_right]):
                canvas = cv2.copyMakeBorder(
                    canvas, pad_top, pad_bot, pad_left, pad_right,
                    cv2.BORDER_CONSTANT, value=(80, 80, 80),
                )
                x0 += pad_left
                x1 += pad_left
                y0 += pad_top
                y1 += pad_top
            canvas = canvas[y0:y1, x0:x1]
            cropped = True
            crop_x0, crop_y0 = x0, y0
        # (canvas mapping 불필요 — viewer 는 클릭 안 받음)

        # 5b) YOLO persons — 정사각형 박스 + 30cm 위험 영역 (점선 빨간 원).
        #      path BFS 는 그 원 안 polyline node 자동 제외 → robot 회피.
        now = time.time()
        if self.persons:
            self.persons = [(x, y, ts, t) for x, y, ts, t in self.persons
                            if (now - ts) <= 3.0]
            box_half = 8
            color_fill = (0, 165, 255)
            color_edge = (255, 255, 255)
            # 30cm world → 픽셀
            danger_r_px = max(1, int(0.80 / self.map_info.resolution))
            for x_w, y_w, ts, tid in self.persons:
                p_px, p_py = self._world_to_px(x_w, y_w)
                if self.follow_mode and px is not None:
                    p_px = p_px - (px - int(self.follow_radius / self.map_info.resolution))
                    p_py = p_py - (py - int(self.follow_radius / self.map_info.resolution))
                # 박스
                cv2.rectangle(canvas, (p_px - box_half, p_py - box_half),
                              (p_px + box_half, p_py + box_half),
                              color_fill, -1, cv2.LINE_AA)
                cv2.rectangle(canvas, (p_px - box_half, p_py - box_half),
                              (p_px + box_half, p_py + box_half),
                              color_edge, 1, cv2.LINE_AA)
                # 30cm 점선 원 — 16개 segment 로 dashed
                n_seg = 16
                for k in range(0, n_seg, 2):  # 짝수 segment 만 = 점선
                    a1 = 2 * math.pi * k / n_seg
                    a2 = 2 * math.pi * (k + 1) / n_seg
                    x1 = int(p_px + danger_r_px * math.cos(a1))
                    y1 = int(p_py + danger_r_px * math.sin(a1))
                    x2 = int(p_px + danger_r_px * math.cos(a2))
                    y2 = int(p_py + danger_r_px * math.sin(a2))
                    cv2.line(canvas, (x1, y1), (x2, y2),
                             (60, 60, 220), 2, cv2.LINE_AA)

        # 6) HUD (텍스트)
        mode = 'FOLLOW' if self.follow_mode else 'FULL'
        traj = 'T:ON' if self.show_trajectory else 'T:OFF'
        left = f'[{mode}] [{traj}]  POIs:{len(self.pois)}  q:quit f:follow t:traj r:reset +/-:zoom'
        if pose is not None:
            x, y, yaw = pose
            left += f'   x={x:.2f} y={y:.2f} yaw={math.degrees(yaw):+.0f}°'
        else:
            left += '   (no TF)'
        self._put_text(canvas, left, (10, 22))

        # 진단 HUD: Loops + Match kf# + Ref# + Proc time + Persons
        right = (f'Loops:{self.loops_count} '
                 f'Match:#{self.last_matched_id} '
                 f'Ref:#{self.ref_id} '
                 f'Proc:{self.proc_time_ms:.0f}ms '
                 f'P:{len(self.persons)}')
        if not _HAS_RTAB_INFO:
            right = '(rtabmap_msgs 없음)'
        self._put_text(canvas, right, (canvas.shape[1] - 480, 22))

        # 7) Loop closure flash (중앙 상단, 1초 fadeout)
        now = time.time()
        if now < self.loop_flash_until:
            alpha = max(0.0, self.loop_flash_until - now)
            color = (0, int(255 * alpha), int(255 * alpha))  # yellow→fade
            text = '*** LOOP CLOSURE ***'
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
            tx = (canvas.shape[1] - tw) // 2
            cv2.putText(canvas, text, (tx, 55),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA)

        # 7a2) 음성 안내 트리거 (wait/start/10m/finish)
        self._voice_update()

        # 7b) 아직 localize 안 됨 → 길안내 대기 표시 (첫 loop closure 전)
        if not self._localized:
            wtext = 'LOCALIZING...'
            (tw, _), _ = cv2.getTextSize(wtext, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
            cv2.putText(canvas, wtext, ((canvas.shape[1] - tw) // 2, 80),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

        # 8) Resize + 오른쪽 panel 합치기 + 키 입력
        display = cv2.resize(canvas, (self.window_size, self.window_size),
                             interpolation=cv2.INTER_AREA)
        # haptic panel 캐시 — serial_haptic_node 와 동일한 펄스 로직 (0.25s) 반영.
        #   사람 감지(2초 hold) + 안내 sector != STOP → 활성 모터 점멸 + PERSON DETECTED.
        now_t = time.time()
        persons_active = (
            self.active_motor != -1
            and any((now_t - ts) <= 2.0 for _x, _y, ts, _t in self.persons)
        )
        pulse_on = (int(now_t / 0.25) % 2 == 0)
        # 거리(m) — 0.1m 양자화하여 key 에 포함 (값 바뀔 때만 재렌더)
        dist_m = self._dist_to_target
        dist_q = None if dist_m is None else round(dist_m, 1)
        # 점멸 중엔 phase 도 key 에 포함 → 매 toggle 시 재렌더 (그 외엔 캐시)
        panel_key = (self.active_motor, persons_active,
                     pulse_on if persons_active else True, dist_q)
        # 우측 패널: 위=햅틱(절반 높이, 캐시), 아래=카메라(실시간, 매 프레임)
        haptic_h = self.window_size // 2
        cam_h = self.window_size - haptic_h
        if getattr(self, '_panel_cache', None) is None or self._panel_cache_key != panel_key:
            _panel = np.zeros((haptic_h, self.panel_width, 3), dtype=np.uint8)
            self._draw_haptic_panel(_panel, persons_active=persons_active,
                                    pulse_on=pulse_on, dist_m=dist_m)
            self._panel_cache = _panel
            self._panel_cache_key = panel_key
        cam_panel = self._build_camera_panel(self.panel_width, cam_h)
        right_col = np.vstack([self._panel_cache, cam_panel])
        display = np.hstack([display, right_col])
        cv2.imshow('Localization', display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            self.get_logger().info('quit by user')
            rclpy.shutdown()
        elif key == ord('f'):
            self.follow_mode = not self.follow_mode
        elif key == ord('t'):
            self.show_trajectory = not self.show_trajectory
        elif key == ord('p'):
            self.show_path = not self.show_path   # 파란 경로선 표시 토글 (ridge 보기용)
        elif key == ord('r'):
            self.trajectory.clear()
        elif key == ord('l'):
            # POI 재로드 (편집 후 yaml 갱신됐을 때)
            self._load_pois()
        elif key == ord('+') or key == ord('='):
            self.follow_radius = max(2.0, self.follow_radius * 0.8)
        elif key == ord('-'):
            self.follow_radius = min(100.0, self.follow_radius * 1.25)

    @staticmethod
    def _put_text(canvas, text, org):
        # 한글 포함 시 PIL 사용 (cv2.putText 는 한글 ???? 깨짐).
        # 영문/숫자만이면 cv2 (빠름).
        has_ko = any('가' <= c <= '힣' for c in text)
        if has_ko and _KFONT is not None:
            # 텍스트 bbox 계산 → ROI 만 PIL 처리 (전체 변환 부하 회피).
            try:
                bbox = _KFONT.getbbox(text)  # (x0, y0, x1, y1)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except Exception:
                tw, th = len(text) * 14, 18
            pad = 4
            x0 = max(org[0] - pad, 0)
            y0 = max(org[1] - th - pad, 0)
            x1 = min(org[0] + tw + pad, canvas.shape[1])
            y1 = min(org[1] + pad, canvas.shape[0])
            if x1 > x0 and y1 > y0:
                roi = canvas[y0:y1, x0:x1]
                pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                d = ImageDraw.Draw(pil)
                d.text((org[0] - x0, org[1] - th - y0), text,
                       font=_KFONT, fill=(255, 255, 255),
                       stroke_width=2, stroke_fill=(0, 0, 0))
                canvas[y0:y1, x0:x1] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            return
        cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)        # 외곽 검정
        cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)  # 흰색


def main():
    rclpy.init()
    node = LocalizationViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
