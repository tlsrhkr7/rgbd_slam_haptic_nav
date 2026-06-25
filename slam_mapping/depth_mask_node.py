"""depth_mask_node — 사람 영역을 depth 에서 0(무효)으로 마스킹.

목적: visual odometry(rgbd_odometry)가 사람(동적/비강체) feature 로 오염돼
      pose 가 드리프트하는 문제 해결. 사람 bbox 영역 depth 를 0 으로 만들면
      그 영역에 3D feature 가 안 생겨 → odometry 가 정지 배경만 사용.

흐름:
    /yolo/person_bboxes (Int32MultiArray, x1,y1,x2,y2 flatten)  ← yolo_person_node (4Hz)
    /camera/.../aligned_depth_to_color/image_raw (16UC1, 30fps) ← RealSense
      → bbox(여유 dilate) 영역 depth=0 → image_masked publish (30fps)
    rgbd_odometry/rtabmap 가 image_masked 를 depth 로 사용 (launch 에서 remap)

안전:
    - bbox 없거나 오래되면(stale) depth 를 **그대로 통과** → 사람 없을 땐 정상 동작
    - 헤더(timestamp/frame) 보존 → rgbd_odometry 동기화 유지
    - cv2 직접 호출 0 (cv_bridge + numpy 만)
"""

import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray


class DepthMaskNode(Node):
    def __init__(self):
        super().__init__('depth_mask')

        self.declare_parameter('depth_in',
                               '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('depth_out',
                               '/camera/camera/aligned_depth_to_color/image_masked')
        self.declare_parameter('bbox_topic', '/yolo/person_bboxes')
        self.declare_parameter('dilate_px', 24)        # bbox 여유 (이동/staleness 대비)
        self.declare_parameter('bbox_timeout_s', 0.6)  # 이 시간 지난 bbox 는 무시

        depth_in = self.get_parameter('depth_in').value
        depth_out = self.get_parameter('depth_out').value
        bbox_topic = self.get_parameter('bbox_topic').value
        self.dilate = int(self.get_parameter('dilate_px').value)
        self.bbox_timeout = float(self.get_parameter('bbox_timeout_s').value)

        self.bridge = CvBridge()
        self.bboxes = []     # [[x1,y1,x2,y2], ...]
        self.bbox_t = 0.0

        # RealSense depth = Best Effort → 맞춰 구독/발행
        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(Image, depth_in, self._depth_cb, qos)
        self.create_subscription(Int32MultiArray, bbox_topic, self._bbox_cb, 10)
        self.pub = self.create_publisher(Image, depth_out, qos)

        self.get_logger().info(
            f'depth_mask up: {depth_in} → {depth_out} (dilate={self.dilate}px)')

    def _bbox_cb(self, msg: Int32MultiArray):
        d = list(msg.data)
        self.bboxes = [d[i:i + 4] for i in range(0, len(d) - 3, 4)]
        self.bbox_t = time.time()

    def _depth_cb(self, msg: Image):
        # bbox 없음/stale → 그대로 통과 (사람 없을 때 정상)
        if not self.bboxes or (time.time() - self.bbox_t) > self.bbox_timeout:
            self.pub.publish(msg)
            return
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception:
            self.pub.publish(msg)
            return
        depth = depth.copy()
        H, W = depth.shape[:2]
        m = self.dilate
        for x1, y1, x2, y2 in self.bboxes:
            xa = max(0, x1 - m); ya = max(0, y1 - m)
            xb = min(W, x2 + m); yb = min(H, y2 + m)
            if xb > xa and yb > ya:
                depth[ya:yb, xa:xb] = 0   # 사람 영역 depth 무효화
        out = self.bridge.cv2_to_imgmsg(depth, encoding=msg.encoding)
        out.header = msg.header   # ★ timestamp/frame 보존 (동기화 유지)
        self.pub.publish(out)


def main():
    rclpy.init()
    node = DepthMaskNode()
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
