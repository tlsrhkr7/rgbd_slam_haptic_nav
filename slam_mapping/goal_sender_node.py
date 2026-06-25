"""goal_sender_node — POI 좌표 → medial axis projection → /goal_pose 발행.

시각장애인 안내:
  사용자가 POI 를 벽 안에 등록해도 자동으로 가장 가까운 통로 중앙으로 보정.
  → 시각장애인 시나리오에서 안전한 도달 위치.

흐름:
  1. /rtabmap/map subscribe (db.grid)
  2. POI yaml load + target 선택
  3. cv2.distanceTransform: 모든 free cell 의 nearest wall 거리 계산
  4. POI 주변 N m 반경 ROI 에서 max-distance free cell 찾음 = 통로 중앙
  5. 그 cell 의 world 좌표 → /goal_pose publish

사용:
  ros2 run slam_mapping goal_sender_node --ros-args -p target:="텐서"
"""

import math
import os
import sys

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

try:
    from rtabmap_msgs.msg import MapGraph
    _HAS_MAPGRAPH = True
except ImportError:
    _HAS_MAPGRAPH = False


class GoalSender(Node):
    def __init__(self):
        super().__init__('goal_sender')

        self.declare_parameter('poi_file', '/home/a/maps/floor4_pois.yaml')
        self.declare_parameter('target', '텐서')
        self.declare_parameter('delay_s', 5.0)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('search_radius_m', 1.5)   # POI 주변 검색 반경 (좁은 통로 fit 방지)
        self.declare_parameter('use_projection', True)   # False 면 yaml 좌표 그대로 발행

        self.poi_file = self.get_parameter('poi_file').value
        self.target = self.get_parameter('target').value
        self.delay_s = float(self.get_parameter('delay_s').value)
        self.map_frame = self.get_parameter('map_frame').value
        self.search_radius_m = float(self.get_parameter('search_radius_m').value)
        self.use_projection = bool(self.get_parameter('use_projection').value)

        # POI load
        if not os.path.exists(self.poi_file):
            self.get_logger().error(f'POI 파일 없음: {self.poi_file}')
            sys.exit(1)
        with open(self.poi_file, 'r') as f:
            data = yaml.safe_load(f) or {}
        pois = data.get('pois', [])
        match = next((p for p in pois if p.get('name') == self.target), None)
        if match is None:
            available = [p.get('name') for p in pois]
            self.get_logger().error(
                f'목적지 "{self.target}" 없음. 가능: {available}')
            sys.exit(1)
        self.target_pose = match
        self.get_logger().info(
            f'목적지 "{self.target}": world=({match["x"]:.2f}, {match["y"]:.2f})')

        # State
        self.map_msg = None
        self.node_poses = {}   # mapGraph: keyframe_id → (x, y)

        # /rtabmap/map subscribe (transient_local)
        qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(OccupancyGrid, '/rtabmap/map', self._map_cb, qos)

        # mapGraph subscribe — ref_node_id 동적 좌표 계산
        if _HAS_MAPGRAPH:
            self.create_subscription(MapGraph, '/rtabmap/mapGraph', self._mapgraph_cb, 10)

        # publisher
        self.pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        self.get_logger().info(f'{self.delay_s:.1f}초 후 projection + 발행...')
        self.create_timer(self.delay_s, self._send_once)
        self._sent = False

    def _map_cb(self, msg: OccupancyGrid):
        self.map_msg = msg

    def _mapgraph_cb(self, msg):
        first_receive = not self.node_poses
        self.node_poses = {
            msg.poses_id[i]: (msg.poses[i].position.x, msg.poses[i].position.y)
            for i in range(min(len(msg.poses_id), len(msg.poses)))
        }
        if first_receive:
            self.get_logger().info(f'mapGraph 받음: {len(self.node_poses)} nodes')
        # delay 끝났지만 mapGraph 못 받아서 아직 발행 안 한 경우 → 지금 발행
        if getattr(self, '_delay_elapsed', False) and not self._sent:
            self._send_now()

    def _project_to_corridor_center(self, target_x, target_y):
        """POI world 좌표 → 통로 중앙 free cell 좌표 변환."""
        if self.map_msg is None:
            self.get_logger().warn('/rtabmap/map 못 받음 — projection 없이 원본 사용')
            return target_x, target_y

        msg = self.map_msg
        w, h = msg.info.width, msg.info.height
        res = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y

        data = np.array(msg.data, dtype=np.int8).reshape(h, w)

        # Binary: free (0) 만 free_mask. occupied (100) + unknown (-1) 모두 0.
        # 매핑 안 한 외부 unknown (44m 거리) 잘못 fit 방지.
        # 통로 가능 영역 = 실제 free 만.
        free_mask = np.zeros((h, w), dtype=np.uint8)
        free_mask[data == 0] = 255

        # 모든 free cell 의 nearest wall 거리
        dist = cv2.distanceTransform(free_mask, cv2.DIST_L2, 5)

        # POI cell 좌표
        cx = int(round((target_x - ox) / res))
        cy_world_up = int(round((target_y - oy) / res))
        # ROS map y 위, OpenCV array y 아래 — distance transform 도 array indexing
        cy = h - 1 - cy_world_up

        # 주변 ROI
        radius_px = int(self.search_radius_m / res)
        x0 = max(0, cx - radius_px)
        y0 = max(0, cy - radius_px)
        x1 = min(w, cx + radius_px)
        y1 = min(h, cy + radius_px)
        roi = dist[y0:y1, x0:x1]

        if roi.size == 0:
            self.get_logger().warn('ROI 비어있음 — projection 없이 원본 사용')
            return target_x, target_y

        # ROI 안 max-distance cell = 가장 안전한 통로 중앙 점
        idx = np.unravel_index(np.argmax(roi), roi.shape)
        proj_cy = y0 + idx[0]
        proj_cx = x0 + idx[1]

        # cell → world (y flip)
        proj_world_x = proj_cx * res + ox
        proj_world_y = (h - 1 - proj_cy) * res + oy

        max_dist = float(roi.max())
        self.get_logger().info(
            f'projection: ({target_x:.2f},{target_y:.2f}) → '
            f'({proj_world_x:.2f},{proj_world_y:.2f}) '
            f'[wall dist={max_dist * res:.2f}m]')

        return proj_world_x, proj_world_y

    def _send_once(self):
        # delay timer 첫 호출 시 _delay_elapsed 표시. mapGraph 있으면 즉시 발행, 없으면 대기.
        self._delay_elapsed = True
        if self._sent:
            return
        if not self.node_poses:
            ref = self.target_pose.get('ref_node_id')
            if ref is not None:
                self.get_logger().warn(
                    f'mapGraph 아직 미수신 — 받으면 자동 발행 (ref_node_id #{ref} 대기)')
                return
        self._send_now()

    def _send_now(self):
        if self._sent:
            return
        p = self.target_pose
        yaw = float(p.get('yaw', 0.0))

        ref = p.get('ref_node_id')
        if ref is not None and ref in self.node_poses:
            nx, ny = self.node_poses[ref]
            base_x = nx + float(p.get('offset_x', 0.0))
            base_y = ny + float(p.get('offset_y', 0.0))
            self.get_logger().info(
                f'ref_node_id #{ref} 동적 좌표: ({base_x:.2f}, {base_y:.2f}) '
                f'[yaml static: ({p["x"]:.2f}, {p["y"]:.2f})]')
        else:
            base_x, base_y = float(p['x']), float(p['y'])
            self.get_logger().warn(
                f'ref_node_id 없음 또는 mapGraph 미수신 → yaml static 사용: '
                f'({base_x:.2f}, {base_y:.2f})')

        if self.use_projection:
            tx, ty = self._project_to_corridor_center(base_x, base_y)
        else:
            tx, ty = base_x, base_y

        msg = PoseStamped()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(tx)
        msg.pose.position.y = float(ty)
        msg.pose.position.z = 0.0
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        self.pub.publish(msg)
        self.get_logger().info(
            f'/goal_pose 발행: ({tx:.2f}, {ty:.2f}) yaw={yaw:.2f}')
        self._sent = True


def main():
    rclpy.init()
    node = GoalSender()
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
