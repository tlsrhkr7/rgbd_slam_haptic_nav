// haptic_arduino.ino — 8채널 진동모터 컨트롤러 (Arduino Uno R3)
//
// 프로토콜 (115200 baud):
//   1바이트 수신
//     0~7   : 해당 sector 모터만 ON (그 외 OFF)
//     0xFF  : 전체 OFF (정지)
//
// 핀 매핑 (path_to_haptic sector → Arduino pin):
//   0=N (전)   → pin 3   HW PWM
//   1=NE       → pin 5   HW PWM
//   2=E (우)   → pin 6   HW PWM
//   3=SE       → pin 4   SW PWM (디지털 핀 — loop 에서 토글)
//   4=S (후)   → pin 7   SW PWM (디지털 핀 — loop 에서 토글)
//   5=SW       → pin 9   HW PWM
//   6=W (좌)   → pin 10  HW PWM
//   7=NW       → pin 11  HW PWM
//
// 변경: 디지털 핀(4,7)을 고정 HIGH 대신 소프트웨어 PWM 으로 구동.
//   구동 회로가 오실레이션 신호를 요구하는 경우(고정 DC 면 안 켜짐) 대응.
//   PWM 채널과 동일하게 동작 + 강도(PWM_INTENSITY) 적용됨.
//
// Watchdog: 500ms 동안 시리얼 입력 없으면 전체 OFF (노드 죽어도 모터 정지 보장).

const uint8_t PIN_MAP[8] = { 3, 5, 6, 4, 7, 9, 10, 11 };
const bool    IS_PWM[8]  = { true, true, true, false, false, true, true, true };
// 254 = 최대 세기(오실레이션 유지). 255 는 HW/SW 둘 다 고정 HIGH(DC)가 되어
//   회로가 오실레이션을 요구할 경우 진동이 멈춤 → 일부러 254 로 둠.
const uint8_t PWM_INTENSITY = 254;  // 0~255 (254=실질 최대)

// 소프트웨어 PWM 파라미터 (디지털 핀용)
const unsigned long SW_PERIOD_US = 1000;  // 1kHz
// 항상 최소 30us 의 OFF 구간 보장 → 듀티가 높아도 신호가 계속 토글(오실레이션 유지).
const unsigned long SW_ON_RAW = (SW_PERIOD_US * (unsigned long)PWM_INTENSITY) / 255UL;
const unsigned long SW_ON_US =
    (SW_ON_RAW > SW_PERIOD_US - 30) ? (SW_PERIOD_US - 30) : SW_ON_RAW;

const unsigned long WATCHDOG_MS = 500;
unsigned long last_recv_ms = 0;
int8_t active = -1;  // -1 = 정지

void set_all_off() {
  for (int i = 0; i < 8; i++) {
    if (IS_PWM[i]) analogWrite(PIN_MAP[i], 0);
    else           digitalWrite(PIN_MAP[i], LOW);
  }
}

// HW PWM 채널만 즉시 설정. 디지털(SW PWM) 채널은 loop 가 토글.
void set_active(int8_t sector) {
  for (int i = 0; i < 8; i++) {
    bool on = (i == sector);
    if (IS_PWM[i]) {
      analogWrite(PIN_MAP[i], on ? PWM_INTENSITY : 0);
    } else if (!on) {
      digitalWrite(PIN_MAP[i], LOW);  // 비활성 디지털 핀은 끔
    }
    // 활성 디지털 핀은 loop 의 SW PWM 이 구동
  }
}

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < 8; i++) pinMode(PIN_MAP[i], OUTPUT);
  set_all_off();
  last_recv_ms = millis();
}

void loop() {
  while (Serial.available() > 0) {
    uint8_t b = Serial.read();
    last_recv_ms = millis();
    if (b == 0xFF) {
      active = -1;
      set_all_off();
    } else if (b <= 7) {
      active = b;
      set_active(active);
    }
  }

  // Watchdog — 500ms 무신호 시 강제 OFF
  if (active != -1 && (millis() - last_recv_ms) > WATCHDOG_MS) {
    active = -1;
    set_all_off();
  }

  // 소프트웨어 PWM — 활성 sector 가 디지털 핀이면 토글로 PWM 신호 생성
  if (active != -1 && !IS_PWM[active]) {
    unsigned long phase = micros() % SW_PERIOD_US;
    digitalWrite(PIN_MAP[active], (phase < SW_ON_US) ? HIGH : LOW);
  }
}
