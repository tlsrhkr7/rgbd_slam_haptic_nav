"""serial_haptic_node — /haptic_motor_idx 구독 → Arduino 시리얼 1바이트 송신.

흐름:
    /haptic_motor_idx (Int32)
        0~7  → 해당 sector 모터 ON
        -1   → 정지 (0xFF 송신)
    → /dev/ttyACM0 115200 baud

펌웨어: firmware/haptic_arduino/haptic_arduino.ino

안전:
    - shutdown 시 0xFF 명시 송신 (마지막 값 유지 방지)
    - 200ms 동안 토픽 무수신이면 자동 0xFF 송출 (노드는 살아있지만 path 끊김 대비)
    - Arduino 펌웨어가 500ms watchdog → 노드 죽어도 모터 OFF 보장
    - 같은 sector 연속 수신이면 송신 생략 (시리얼 트래픽 절약)
"""

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from visualization_msgs.msg import MarkerArray

try:
    import serial
except ImportError:
    serial = None


STOP_BYTE = 0xFF


class SerialHapticNode(Node):
    def __init__(self):
        super().__init__('serial_haptic')

        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('topic', '/haptic_motor_idx')
        self.declare_parameter('keepalive_rate', 10.0)   # Hz — 펄스 토글 위해 ↑
        self.declare_parameter('input_timeout_s', 0.5)   # 이 시간 무수신 → 정지
        # 사람 감지 → 같은 방향이 ON/OFF 펄스
        self.declare_parameter('persons_topic', '/yolo/persons_map')
        self.declare_parameter('pulse_half_period_s', 0.25)  # 250ms ON, 250ms OFF
        self.declare_parameter('persons_hold_s', 2.0)        # 마지막 사람 감지 후 N초 유지

        port = self.get_parameter('port').value
        baud = int(self.get_parameter('baud').value)
        topic = self.get_parameter('topic').value
        persons_topic = self.get_parameter('persons_topic').value
        keepalive_rate = float(self.get_parameter('keepalive_rate').value)
        self.input_timeout_s = float(self.get_parameter('input_timeout_s').value)
        self.pulse_half = float(self.get_parameter('pulse_half_period_s').value)
        self.persons_hold_s = float(self.get_parameter('persons_hold_s').value)

        if serial is None:
            self.get_logger().error(
                'pyserial 미설치. `pip install --user pyserial` 또는 '
                '`sudo apt install python3-serial` 필요.'
            )
            raise SystemExit(1)

        try:
            self.ser = serial.Serial(port, baud, timeout=0, write_timeout=0.05)
        except Exception as e:
            self.get_logger().error(f'시리얼 포트 열기 실패: {port} — {e}')
            raise SystemExit(1)

        # Arduino reset 대기 (DTR 토글 후 부팅 ~2s)
        time.sleep(2.0)
        # 시작 시 강제 정지 송출
        self._write(STOP_BYTE)
        self._last_byte = STOP_BYTE       # 실제로 시리얼에 마지막 송신한 바이트
        self._cmd_byte = STOP_BYTE        # path_to_haptic 이 보낸 안내 sector
        self._last_recv = time.time()     # haptic_motor_idx 마지막 수신
        self._persons_last_seen = 0.0     # YOLO persons 마지막 감지 시각

        self.create_subscription(Int32, topic, self._cb, 10)
        self.create_subscription(MarkerArray, persons_topic, self._persons_cb, 10)
        self.create_timer(1.0 / keepalive_rate, self._keepalive)

        self.get_logger().info(
            f'serial_haptic up. port={port}@{baud} subscribe={topic}'
        )

    def _cb(self, msg: Int32):
        self._last_recv = time.time()
        v = int(msg.data)
        self._cmd_byte = STOP_BYTE if (v < 0 or v > 7) else v

    def _persons_cb(self, msg: MarkerArray):
        # DELETE 아닌 마커가 하나라도 있으면 "사람 감지" 상태로 갱신
        for m in msg.markers:
            if m.action != 2:
                self._persons_last_seen = time.time()
                return

    def _keepalive(self):
        now = time.time()
        # 입력 timeout — 토픽 끊겼으면 강제 STOP
        if now - self._last_recv > self.input_timeout_s:
            if self._last_byte != STOP_BYTE:
                self._write(STOP_BYTE)
                self._last_byte = STOP_BYTE
            return

        target = self._cmd_byte
        # 사람 감지 활성 + 안내 sector 가 STOP 아니면 펄스 패턴
        persons_active = (now - self._persons_last_seen) <= self.persons_hold_s
        if persons_active and target != STOP_BYTE:
            phase = int(now / self.pulse_half) % 2  # 0=ON, 1=OFF
            target = self._cmd_byte if phase == 0 else STOP_BYTE

        if target != self._last_byte:
            self._write(target)
            self._last_byte = target
        else:
            # 같은 값이라도 주기적 재송신 → Arduino watchdog (500ms) 갱신
            self._write(target)

    def _write(self, b: int):
        try:
            self.ser.write(bytes([b & 0xFF]))
        except Exception as e:
            self.get_logger().warn(f'시리얼 송신 실패: {e}', throttle_duration_sec=2.0)

    def shutdown_stop(self):
        try:
            self.ser.write(bytes([STOP_BYTE]))
            self.ser.flush()
        except Exception:
            pass
        try:
            self.ser.close()
        except Exception:
            pass


def main():
    rclpy.init()
    node = SerialHapticNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
