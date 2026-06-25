#!/usr/bin/env python3
# test_motors.py — 8 sector 모터를 하나씩 켜며 어느 게 동작하는지 확인.
#   sector 0~2,5~7 = HW PWM(3,5,6,9,10,11) / sector 3,4 = SW PWM 디지털(4,7)
# 사용: python3 test_motors.py   (각 sector 2초씩, 어느 게 떨리는지 느껴보기)
import serial, time, sys

PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyACM0'
NAMES = ['0=N(전) pin3 PWM', '1=NE pin5 PWM', '2=E(우) pin6 PWM',
         '3=SE pin4 DIGITAL(SW PWM)', '4=S(후) pin7 DIGITAL(SW PWM)',
         '5=SW pin9 PWM', '6=W(좌) pin10 PWM', '7=NW pin11 PWM']

s = serial.Serial(PORT, 115200)
time.sleep(2.0)  # Arduino reset 대기
print('=== 모터 테스트 시작 (각 2초) ===')
try:
    for i in range(8):
        print(f'  sector {NAMES[i]} ... ON')
        s.write(bytes([i]))
        time.sleep(2.0)
        s.write(bytes([0xFF]))  # OFF
        time.sleep(0.5)
    print('=== 끝. 전체 OFF ===')
    s.write(bytes([0xFF]))
finally:
    s.close()
