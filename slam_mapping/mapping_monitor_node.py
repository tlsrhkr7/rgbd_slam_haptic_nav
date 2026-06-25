"""mapping_monitor_node — 추가(보강) 매핑 실시간 2D 모니터.

목적: 기존 맵 위에서 매핑이 보강되는 걸 한눈에 확인.
  - 기존 맵(2D occupancy) 회색조 배경
  - 기존 키프레임 노드 = 흐린 파랑 점 / 이번 세션 새 노드 = 밝은 초록 점 (★ 강화되는 느낌)
  - 내 위치(빨간 원) + 바라보는 방향(화살표) — TF map→base_link
  - loop closure 발생 시 중앙 노란 'LOOP CLOSURE' 플래시 + 카운트
  - HUD: 전체 노드 / 이번 세션 추가 노드 / loop 수

토픽:
  /rtabmap/map        OccupancyGrid          (mapping 중 갱신되는 2D grid)
  /rtabmap/mapGraph   rtabmap_msgs/MapGraph  (노드 id→pose — 강화 시각화 핵심)
  /rtabmap/info       rtabmap_msgs/Info      (loop closure)
  TF map→base_link                            현재 위치 + heading

조작: q/ESC 종료, f follow↔full, +/- 줌
"""

import math
import time

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from tf2_ros import Buffer, TransformListener

try:
    from rtabmap_msgs.msg import Info as RtabInfo
    _HAS_INFO = True
except ImportError:
    _HAS_INFO = False

try:
    from rtabmap_msgs.msg import MapGraph
    _HAS_GRAPH = True
except ImportError:
    _HAS_GRAPH = False


class MappingMonitor(Node):
    def __init__(self):
        super().__init__('mapping_monitor')

        self.declare_parameter('map_topic', '/rtabmap/map')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('window_size', 800)
        self.declare_parameter('follow_radius_m', 12.0)
        self.declare_parameter('render_rate', 10.0)

        map_topic = self.get_parameter('map_topic').value
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.window_size = int(self.get_parameter('window_size').value)
        self.follow_radius = float(self.get_parameter('follow_radius_m').value)
        render_rate = float(self.get_parameter('render_rate').value)

        self.map_img = None
        self.map_info = None
        self._map_bgr = None
        self.node_poses = {}        # {id: (x, y)}
        self.base_node_ids = None   # 모니터 시작 시점의 기존 노드 (None=아직 미수신)
        self.loop_flash_until = 0.0
        self.last_loop_id = 0
        self.loops_count = 0
        self.last_node_id = 0
        self.follow_mode = True

        qos_map = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(OccupancyGrid, map_topic, self._map_cb, qos_map)
        if _HAS_INFO:
            self.create_subscription(RtabInfo, '/rtabmap/info', self._info_cb, 10)
        if _HAS_GRAPH:
            self.create_subscription(MapGraph, '/rtabmap/mapGraph', self._graph_cb, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_timer(1.0 / render_rate, self._render)
        cv2.namedWindow('Mapping Monitor', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Mapping Monitor', self.window_size, self.window_size)
        self.get_logger().info(f'Mapping monitor up. map={map_topic}')

    # ─── Callbacks ───
    def _map_cb(self, msg: OccupancyGrid):
        w, h = msg.info.width, msg.info.height
        data = np.array(msg.data, dtype=np.int8).reshape(h, w)
        img = np.full((h, w), 200, dtype=np.uint8)   # unknown=회색
        img[data == 0] = 254                          # free=흰색
        img[(data >= 65) & (data <= 100)] = 0         # occupied=검정
        self.map_img = cv2.flip(img, 0)
        self.map_info = msg.info
        self._map_bgr = None   # 맵 갱신 → 배경 캐시 무효화 (= 강화되는 모습)

    def _info_cb(self, msg):
        lid = getattr(msg, 'loop_closure_id', 0)
        pid = getattr(msg, 'proximity_detection_id', 0)
        matched = lid if lid > 0 else pid
        now = time.time()
        if matched > 0 and now > self.loop_flash_until - 0.3:
            self.loop_flash_until = now + 1.0
        if lid > 0 and lid != self.last_loop_id:
            self.last_loop_id = lid
            self.loops_count += 1

    def _graph_cb(self, msg):
        n = min(len(msg.poses_id), len(msg.poses))
        poses = {int(msg.poses_id[i]):
                 (msg.poses[i].position.x, msg.poses[i].position.y)
                 for i in range(n)}
        self.node_poses = poses
        if poses:
            self.last_node_id = max(poses.keys())
            if self.base_node_ids is None:
                # 첫 수신 = 기존 맵 노드 (이게 "기존", 이후 새로 생기는 게 "보강")
                self.base_node_ids = set(poses.keys())

    # ─── Helpers ───
    def _world_to_px(self, x, y):
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        h = self.map_img.shape[0]
        return (int(round((x - ox) / res)), h - 1 - int(round((y - oy) / res)))

    def _get_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time(),
                timeout=Duration(seconds=0.3))
        except Exception:
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return t.x, t.y, math.atan2(siny, cosy)

    # ─── Render ───
    def _render(self):
        if self.map_img is None or self.map_info is None:
            return
        if self._map_bgr is None:
            self._map_bgr = cv2.cvtColor(self.map_img, cv2.COLOR_GRAY2BGR)
        canvas = self._map_bgr.copy()

        # 1) 키프레임 노드 — 기존(흐린 파랑) vs 이번 세션 새 노드(밝은 초록 = 강화)
        new_count = 0
        base = self.base_node_ids or set()
        for nid, (x, y) in self.node_poses.items():
            px, py = self._world_to_px(x, y)
            if nid in base:
                cv2.circle(canvas, (px, py), 2, (180, 120, 60), -1, cv2.LINE_AA)  # 기존: 흐린 파랑
            else:
                cv2.circle(canvas, (px, py), 4, (0, 255, 0), -1, cv2.LINE_AA)      # 새: 밝은 초록
                new_count += 1

        # 2) 내 위치 + 바라보는 방향
        pose = self._get_pose()
        if pose is not None:
            x, y, yaw = pose
            rx, ry = self._world_to_px(x, y)
            cv2.circle(canvas, (rx, ry), 8, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (rx, ry), 8, (255, 255, 255), 2, cv2.LINE_AA)
            ar = 28
            hx = int(rx + ar * math.cos(yaw))
            hy = int(ry - ar * math.sin(yaw))   # 이미지 y 반전
            cv2.arrowedLine(canvas, (rx, ry), (hx, hy), (0, 0, 255), 3,
                            cv2.LINE_AA, tipLength=0.35)

        # 3) Follow crop (로봇 중심)
        if self.follow_mode and pose is not None:
            half = int(self.follow_radius / self.map_info.resolution)
            H, W = canvas.shape[:2]
            x0, y0, x1, y1 = rx - half, ry - half, rx + half, ry + half
            pt, pl = max(0, -y0), max(0, -x0)
            pb, pr = max(0, y1 - H), max(0, x1 - W)
            if any([pt, pl, pb, pr]):
                canvas = cv2.copyMakeBorder(canvas, pt, pb, pl, pr,
                                            cv2.BORDER_CONSTANT, value=(80, 80, 80))
                x0 += pl; x1 += pl; y0 += pt; y1 += pt
            canvas = canvas[y0:y1, x0:x1]

        canvas = cv2.resize(canvas, (self.window_size, self.window_size),
                            interpolation=cv2.INTER_AREA)

        # 4) HUD
        total = len(self.node_poses)
        hud = f'nodes:{total}  +new:{new_count}  loops:{self.loops_count}'
        cv2.putText(canvas, hud, (10, 24), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                    (0, 255, 0), 1, cv2.LINE_AA)

        # 5) Loop closure flash
        now = time.time()
        if now < self.loop_flash_until:
            a = max(0.0, self.loop_flash_until - now)
            col = (0, int(255 * a), int(255 * a))
            txt = '*** LOOP CLOSURE ***'
            (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
            cv2.putText(canvas, txt, ((self.window_size - tw) // 2, 60),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, col, 2, cv2.LINE_AA)

        cv2.imshow('Mapping Monitor', canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            rclpy.shutdown()
        elif key == ord('f'):
            self.follow_mode = not self.follow_mode
        elif key in (ord('+'), ord('=')):
            self.follow_radius = max(2.0, self.follow_radius * 0.8)
        elif key == ord('-'):
            self.follow_radius = min(100.0, self.follow_radius * 1.25)


def main():
    rclpy.init()
    node = MappingMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
