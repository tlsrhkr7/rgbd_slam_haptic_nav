"""path_to_haptic_node — Nav2 /plan + 사용자 위치/yaw → 8방향 진동 모터 명령.

입력:
  /plan                       nav_msgs/Path           Nav2 전역 경로 (map frame)
  TF map → base_link                                  사용자 현재 위치 + heading

출력:
  /haptic_motor_idx           std_msgs/Int32          활성 sector 0..7 (-1 = 정지)
  /haptic_visualization       MarkerArray             viewer 시각화용

8방향 sector:
  0 = 전방
  1 = 전-우
  2 = 우
  3 = 후-우
  4 = 후
  5 = 후-좌
  6 = 좌
  7 = 전-좌
  -1 = goal 도달 (정지)

알고리즘:
  1. /plan 받음 → path 목록 저장
  2. 매 N Hz: TF lookup → 사용자 (x, y, yaw)
  3. path 상 사용자 lookahead_dist (예: 1.0m) 앞 점 찾기
  4. 그 점 방향 vector = desired heading
  5. 사용자 yaw - desired heading = 상대 각도 θ
  6. sector = round(θ / 45°) mod 8
  7. publish
"""

import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import Int32
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


class PathToHaptic(Node):
    def __init__(self):
        super().__init__('path_to_haptic')

        self.declare_parameter('lookahead_far', 2.0)    # m (사람 미검출 시)
        self.declare_parameter('lookahead_near', 0.5)   # m (사람 검출 시)
        self.declare_parameter('person_hold_sec', 1.5)  # 마지막 검출 후 near 유지 시간
        self.declare_parameter('goal_tolerance', 0.5)   # m (도달 판정)
        self.declare_parameter('rate', 5.0)             # Hz
        self.declare_parameter('user_frame', 'base_link')
        self.declare_parameter('map_frame', 'map')

        self.lookahead_far = float(self.get_parameter('lookahead_far').value)
        self.lookahead_near = float(self.get_parameter('lookahead_near').value)
        self.person_hold = float(self.get_parameter('person_hold_sec').value)
        self.goal_tol = float(self.get_parameter('goal_tolerance').value)
        rate = float(self.get_parameter('rate').value)
        self.user_frame = self.get_parameter('user_frame').value
        self.map_frame = self.get_parameter('map_frame').value

        # State
        self.plan_points = []          # list of (x, y) in map frame
        self.last_person_time = None   # 마지막 사람 검출 시각 (rclpy Time, sim time 반영)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Sub/Pub — viewer 의 자체 path /user_path subscribe (이전 nav2 /plan 대체)
        self.create_subscription(Path, '/user_path', self._plan_cb, 10)
        # 사람 검출 상태 → lookahead 전환 (검출 시 near, 평소 far)
        self.create_subscription(
            MarkerArray, '/yolo/persons_map', self._persons_cb, 10)
        self.pub_motor = self.create_publisher(Int32, '/haptic_motor_idx', 10)
        self.pub_marker = self.create_publisher(MarkerArray, '/haptic_visualization', 10)

        # Timer
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f'path_to_haptic up. lookahead far={self.lookahead_far}m / '
            f'near={self.lookahead_near}m (person), rate={rate}Hz')

    def _plan_cb(self, msg: Path):
        self.plan_points = [
            (p.pose.position.x, p.pose.position.y) for p in msg.poses
        ]
        if self.plan_points:
            self.get_logger().info(
                f'plan received: {len(self.plan_points)} points, '
                f'goal=({self.plan_points[-1][0]:.2f}, {self.plan_points[-1][1]:.2f})')

    def _persons_cb(self, msg: MarkerArray):
        # ADD action 마커가 하나라도 있으면 사람 검출 상태 갱신.
        # (미검출 프레임은 빈 MarkerArray → last_person_time 미갱신 → hold 후 far 복귀)
        for m in msg.markers:
            if m.action == Marker.ADD:
                self.last_person_time = self.get_clock().now()
                return

    def _person_detected(self):
        if self.last_person_time is None:
            return False
        dt = (self.get_clock().now() - self.last_person_time).nanoseconds / 1e9
        return dt < self.person_hold

    def _get_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.user_frame,
                rclpy.time.Time(), timeout=Duration(seconds=0.2),
            )
        except Exception:
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return t.x, t.y, math.atan2(siny, cosy)

    def _tick(self):
        if not self.plan_points:
            return
        pose = self._get_pose()
        if pose is None:
            return
        ux, uy, uyaw = pose

        # 사람 검출 상태 → lookahead 조건부 (검출 0.5m, 평소 2m)
        lookahead = (
            self.lookahead_near if self._person_detected()
            else self.lookahead_far)

        # Goal 도달 판정
        gx, gy = self.plan_points[-1]
        if math.hypot(gx - ux, gy - uy) < self.goal_tol:
            self.pub_motor.publish(Int32(data=-1))
            self._publish_markers(ux, uy, uyaw, None, sector=-1)
            return

        # 사용자에서 lookahead_dist 앞의 path 점 찾기
        target = None
        # 가장 가까운 점부터 시작해서 lookahead 거리 이상인 첫 점
        # (단순 알고리즘: 모든 점 중 거리 lookahead 이상 + 가장 가까운 거)
        nearest_i = 0
        nearest_d = float('inf')
        for i, (px, py) in enumerate(self.plan_points):
            d = math.hypot(px - ux, py - uy)
            if d < nearest_d:
                nearest_d = d
                nearest_i = i
        # TARGET 의 robot frame 위치 — 전방/후방 판단
        cos_y = math.cos(uyaw)
        sin_y = math.sin(uyaw)
        tgt_dx, tgt_dy = gx - ux, gy - uy
        target_x_robot = tgt_dx * cos_y + tgt_dy * sin_y
        target_forward = target_x_robot >= 0

        if target_forward:
            # TARGET 전방 → forward filter 적용 (stale path 의 robot 뒤편 점 skip)
            for i in range(nearest_i, len(self.plan_points)):
                px, py = self.plan_points[i]
                dx, dy = px - ux, py - uy
                if dx * cos_y + dy * sin_y < 0:
                    continue  # robot 뒤편 점 skip
                if math.hypot(dx, dy) >= lookahead:
                    target = (px, py)
                    break
            if target is None:
                target = self.plan_points[-1]  # TARGET 전방이라 안전
        else:
            # TARGET 후방 → 사용자 뒤돌아본 상태. filter 끄고 정상 안내 (후방 sector 출력).
            for i in range(nearest_i, len(self.plan_points)):
                px, py = self.plan_points[i]
                if math.hypot(px - ux, py - uy) >= lookahead:
                    target = (px, py)
                    break
            if target is None:
                target = self.plan_points[-1]

        tx, ty = target
        # 사용자 → target 방향 (world frame)
        desired_yaw = math.atan2(ty - uy, tx - ux)
        # 사용자 yaw 대비 상대 각도 (좌/우 보정)
        rel = self._normalize_angle(desired_yaw - uyaw)
        # 8 sector quantize: sector 0 = 전방 (rel 가 0), 시계 방향
        # rel 양수 = 좌, 음수 = 우 (ROS REP-103 convention).
        # 모터 인덱스: 0 전방, 1 전-우, 2 우, 3 후-우, 4 후, 5 후-좌, 6 좌, 7 전-좌
        # → sector_float = -rel / (pi/4)  (시계 방향 sector)
        sector_float = -rel / (math.pi / 4.0)
        sector = int(round(sector_float)) % 8

        self.pub_motor.publish(Int32(data=sector))
        self._publish_markers(ux, uy, uyaw, target, sector=sector)

    @staticmethod
    def _normalize_angle(a):
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a

    def _publish_markers(self, ux, uy, uyaw, target, sector):
        markers = MarkerArray()
        # 사용자 heading 화살표 (빨강)
        m1 = Marker()
        m1.header.frame_id = self.map_frame
        m1.header.stamp = self.get_clock().now().to_msg()
        m1.ns = 'haptic'
        m1.id = 0
        m1.type = Marker.ARROW
        m1.action = Marker.ADD
        m1.pose.position.x = ux
        m1.pose.position.y = uy
        m1.pose.position.z = 0.0
        m1.pose.orientation.z = math.sin(uyaw / 2)
        m1.pose.orientation.w = math.cos(uyaw / 2)
        m1.scale.x = 0.8
        m1.scale.y = 0.15
        m1.scale.z = 0.15
        m1.color.r = 1.0
        m1.color.a = 0.9
        markers.markers.append(m1)
        # Target 직선 (녹색)
        if target is not None:
            m2 = Marker()
            m2.header.frame_id = self.map_frame
            m2.header.stamp = m1.header.stamp
            m2.ns = 'haptic'
            m2.id = 1
            m2.type = Marker.LINE_STRIP
            m2.action = Marker.ADD
            m2.scale.x = 0.05
            m2.color.g = 1.0
            m2.color.a = 0.9
            from geometry_msgs.msg import Point
            p1 = Point(); p1.x = ux; p1.y = uy
            p2 = Point(); p2.x = target[0]; p2.y = target[1]
            m2.points = [p1, p2]
            markers.markers.append(m2)
        self.pub_marker.publish(markers)


def main():
    rclpy.init()
    node = PathToHaptic()
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
