"""yolo_person_node — YOLOv11n person detection + map frame 위치 추정.

흐름:
    /camera/color/image_raw          → YOLOv11n inference (class=0 person만)
    /camera/aligned_depth_to_color/* → bbox 중심 depth 추출
    /camera/color/camera_info        → intrinsics
    TF camera_color_optical_frame → map → 사람 위치 변환
    publish /yolo/persons_map (MarkerArray, lifetime 0.5s)

viewer 가 MarkerArray subscribe 해서 맵 위에 깜빡이는 마커 표시.
"""

import os
import sys

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

# tf2_geometry_msgs import 는 .transform 동작 위한 plugin 등록용 (사용 안 해도 import 필수)
try:
    import tf2_geometry_msgs  # noqa: F401
except ImportError:
    pass


class YoloPersonNode(Node):
    def __init__(self):
        super().__init__('yolo_person_node')

        # 파라미터
        # TensorRT FP16 엔진(.engine) 기본 — GPU(Orin) 추론 ~32ms. .pt 도 가능(자동 fallback).
        self.declare_parameter('model_path', '/home/a/ros2_ws/yolo11n.engine')
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('infer_rate', 5.0)  # Hz — 매 frame 안 함 (CPU 부담)

        model_path = self.get_parameter('model_path').value
        self.conf_th = float(self.get_parameter('conf_threshold').value)
        self.camera_frame = self.get_parameter('camera_frame').value
        self.map_frame = self.get_parameter('map_frame').value
        infer_period = 1.0 / float(self.get_parameter('infer_rate').value)

        # YOLO 로드. .engine(TensorRT) 은 task 추론 불가 → 명시 필요.
        #   엔진 로드 실패 시(미생성/하드웨어 변경) yolo11n.pt 로 자동 fallback.
        from ultralytics import YOLO
        try:
            if str(model_path).endswith('.engine'):
                self.model = YOLO(model_path, task='detect')
            else:
                self.model = YOLO(model_path)
            self.get_logger().info(f'YOLO loaded: {model_path}')
        except Exception as e:
            self.get_logger().warn(f'{model_path} 로드 실패 ({e}) → yolo11n.pt fallback')
            try:
                self.model = YOLO('yolo11n.pt')
                self.get_logger().info('YOLO loaded: yolo11n.pt (fallback)')
            except Exception as e2:
                self.get_logger().error(f'YOLO load 실패: {e2}')
                sys.exit(1)

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.depth_img = None
        self.K = None        # 3x3 intrinsics
        self.color_msg = None  # 최신 color frame

        # Subscriptions
        self.create_subscription(
            Image, '/camera/camera/color/image_raw', self._color_cb, 10)
        self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self._depth_cb, 10)
        self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self._info_cb, 10)

        # Publisher
        self.marker_pub = self.create_publisher(MarkerArray, '/yolo/persons_map', 10)
        # 사람 bbox(이미지 좌표 x1,y1,x2,y2 flatten) — depth_mask_node 가 odometry 마스킹에 사용
        from std_msgs.msg import Int32MultiArray
        self._Int32MultiArray = Int32MultiArray
        self.bbox_pub = self.create_publisher(Int32MultiArray, '/yolo/person_bboxes', 10)

        # Infer timer (5Hz)
        self.create_timer(infer_period, self._infer)

        self.get_logger().info(
            f'YOLO person detector up. conf={self.conf_th}, rate={1.0/infer_period:.1f}Hz')

    def _color_cb(self, msg):
        self.color_msg = msg

    def _depth_cb(self, msg):
        # 16UC1 (mm) 또는 32FC1 (m) — RealSense aligned_depth = 16UC1
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _info_cb(self, msg):
        self.K = np.array(msg.k).reshape(3, 3)

    def _infer(self):
        if self.color_msg is None or self.depth_img is None or self.K is None:
            return

        try:
            img = self.bridge.imgmsg_to_cv2(self.color_msg, 'bgr8')
        except Exception:
            return

        # ByteTrack 재도입 — 사용자 이동 시 같은 사람이 여러 마커로 분열되는 문제 해결.
        #   위치-hash ID 는 카메라 이동에 ID 가 바뀌어 잔상 누적 → 트래킹으로 ID 영속화.
        #   ByteTrack 자체는 CPU ~1-2ms (렉 무관). 이전 렉 원인은 YOLO CPU 추론 → 지금 GPU.
        try:
            results = self.model.track(
                img, persist=True, classes=[0],
                conf=self.conf_th, verbose=False, tracker='bytetrack.yaml',
            )
        except Exception as e:
            self.get_logger().error(f'inference 실패: {e}')
            return

        markers = MarkerArray()
        bboxes_flat = []   # 마스킹용 — 모든 검출 사람 bbox (id/depth 무관)
        depth_h, depth_w = self.depth_img.shape[:2]
        fx, fy = self.K[0, 0], self.K[1, 1]
        ppx, ppy = self.K[0, 2], self.K[1, 2]

        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            # 마스킹용 bbox — id/depth 검사 전에 수집 (검출된 모든 사람 가림)
            bboxes_flat.extend([int(x1), int(y1), int(x2), int(y2)])
            # 트랙 미확정(id None) 프레임은 마커 skip — viewer 3s hold 로 끊김 방어됨.
            if box.id is None:
                continue
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            track_id = int(box.id[0])   # ByteTrack 영속 ID → 같은 사람 = 같은 마커

            # bbox 중심이 아니라 약간 위쪽 (사람 가슴 위치) 으로 — 발 아닌 몸통 depth
            cy_body = int(y1 + (y2 - y1) * 0.3)
            cy_body = max(0, min(depth_h - 1, cy_body))
            cx = max(0, min(depth_w - 1, cx))

            # depth 중앙값 (5x5 patch, 0/invalid 제외)
            patch = self.depth_img[
                max(0, cy_body - 2):cy_body + 3,
                max(0, cx - 2):cx + 3,
            ]
            valid = patch[(patch > 100) & (patch < 8000)]  # 0.1m~8m 유효
            if valid.size < 3:
                continue
            depth_mm = float(np.median(valid))
            depth_m = depth_mm / 1000.0

            # Pixel → camera 3D
            cam_x = (cx - ppx) * depth_m / fx
            cam_y = (cy_body - ppy) * depth_m / fy
            cam_z = depth_m

            # Camera → map TF (latest 사용 — 시점 정밀도 < 0.1s)
            pt = PointStamped()
            pt.header.frame_id = self.camera_frame
            pt.header.stamp = self.color_msg.header.stamp
            pt.point.x = cam_x
            pt.point.y = cam_y
            pt.point.z = cam_z

            try:
                pt_map = self.tf_buffer.transform(
                    pt, self.map_frame, timeout=Duration(seconds=0.05))
            except Exception:
                # TF 못 받으면 latest 시도
                try:
                    pt.header.stamp = rclpy.time.Time().to_msg()
                    pt_map = self.tf_buffer.transform(
                        pt, self.map_frame, timeout=Duration(seconds=0.05))
                except Exception:
                    continue

            # Marker — 정사각형 box, ByteTrack ID 사용 (frame 간 동일 ID 유지)
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = self.color_msg.header.stamp
            m.ns = 'persons'
            m.id = track_id   # ★ ByteTrack ID — 같은 사람은 같은 marker 갱신 (깜빡임 X)
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = pt_map.point.x
            m.pose.position.y = pt_map.point.y
            m.pose.position.z = pt_map.point.z
            m.pose.orientation.w = 1.0
            m.scale.x = 0.5
            m.scale.y = 0.5
            m.scale.z = 0.5
            m.color.r = 1.0
            m.color.g = 0.4
            m.color.b = 0.0
            m.color.a = 0.9
            m.lifetime = Duration(seconds=1.5).to_msg()  # 0.5 → 1.5s (끊김 ↓)
            markers.markers.append(m)

        self.marker_pub.publish(markers)
        # 사람 bbox publish — depth_mask_node 가 odometry 입력 depth 마스킹
        bb = self._Int32MultiArray()
        bb.data = bboxes_flat
        self.bbox_pub.publish(bb)


def main():
    rclpy.init()
    node = YoloPersonNode()
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
