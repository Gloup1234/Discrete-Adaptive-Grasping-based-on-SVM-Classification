import numpy as np
import time
import serial
import os

# --- 电机极值限制参数 ---
P_MIN = -95.5; P_MAX = 95.5       
V_MIN = -45.0; V_MAX = 45.0
KP_MIN = 0.0;  KP_MAX = 500.0
KD_MIN = 0.0;  KD_MAX = 5.0
T_MIN = -13.0; T_MAX = 13.0

def float_to_uint(x, x_min, x_max, bits):
    span = x_max - x_min
    offset = x_min
    return np.uint16((x - offset) * ((1 << bits) - 1) / span)

def LIMIT_MIN_MAX(x, min_val, max_val):
    if x <= min_val: return min_val
    elif x > max_val: return max_val
    return x

class DM_USB2CAN_Serial:
    def __init__(self, port="/dev/ttyACM0", baudrate=921600):
        os.system(f"sudo chmod 777 {port}")
        self.ser = serial.Serial(port, baudrate, timeout=0.1)
        if self.ser.isOpen():
            print(f"成功打开 USB2CAN 串口: {port}")
            self.init_can_baudrate()

    def init_can_baudrate(self):
        """配置 1Mbps 波特率并清空历史缓存"""
        send_data = np.array([0x55, 0x05, 0x00, 0xAA, 0x55], np.uint8)
        self.ser.write(bytes(send_data.T))
        time.sleep(0.1)
        self.clear_buffer() # 开局清空垃圾数据
        print("=> 已将 USB 盒子底层 CAN 波特率配置为 1Mbps")

    def clear_buffer(self):
        """极其关键：疯狂抽干接收缓存，防止盒子红灯死机"""
        count = self.ser.in_waiting
        if count > 0:
            self.ser.read(count)

    def CanComm_ControlCmd(self, cmd, motor_id):
        """发送系统指令"""
        send_data = np.array([0x55,0xAA,0x1e,0x01,0x01,0x00,0x00,0x00,0x0a,0x00,0x00,0x00,0x00,0,0,0,0,0x00,0x08,0x00,0x00,0,0,0,0,0,0,0,0,0x88], np.uint8)
        buf = np.array([motor_id, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00], np.uint8)
        
        if cmd == "ENABLE": buf[8] = 0xFC
        elif cmd == "DISABLE": buf[8] = 0xFD
        elif cmd == "ZERO": buf[8] = 0xFE
            
        send_data[13] = motor_id
        send_data[21:29] = buf[1:9]
        self.ser.write(bytes(send_data.T))
        time.sleep(0.05)
        self.clear_buffer() # 发完立刻清缓存

    def CanComm_SendControlPara(self, f_p, f_v, f_kp, f_kd, f_t, motor_id):
        """发送 MIT 模式运动指令"""
        send_data = np.array([0x55,0xAA,0x1e,0x01,0x01,0x00,0x00,0x00,0x0a,0x00,0x00,0x00,0x00,0,0,0,0,0x00,0x08,0x00,0x00,0,0,0,0,0,0,0,0,0x88], np.uint8)
        buf = np.zeros(9, np.uint8)

        f_p = LIMIT_MIN_MAX(f_p, P_MIN, P_MAX)
        f_v = LIMIT_MIN_MAX(f_v, V_MIN, V_MAX)
        f_kp = LIMIT_MIN_MAX(f_kp, KP_MIN, KP_MAX)
        f_kd = LIMIT_MIN_MAX(f_kd, KD_MIN, KD_MAX)
        f_t = LIMIT_MIN_MAX(f_t, T_MIN, T_MAX)

        p = float_to_uint(f_p, P_MIN, P_MAX, 16)            
        v = float_to_uint(f_v, V_MIN, V_MAX, 12)
        kp = float_to_uint(f_kp, KP_MIN, KP_MAX, 12)
        kd = float_to_uint(f_kd, KD_MIN, KD_MAX, 12)
        t = float_to_uint(f_t, T_MIN, T_MAX, 12)

        buf[0] = motor_id
        buf[1] = p >> 8
        buf[2] = p & 0xFF
        buf[3] = v >> 4
        buf[4] = ((v & 0xF) << 4) | (kp >> 8)
        buf[5] = kp & 0xFF
        buf[6] = kd >> 4
        buf[7] = ((kd & 0xF) << 4) | (t >> 8)
        buf[8] = t & 0xff

        send_data[13] = motor_id
        send_data[21:29] = buf[1:9]
        
        self.ser.write(bytes(send_data.T))
        self.clear_buffer() # 每次下发控制指令后，立刻抽干反馈数据

if __name__ == "__main__":
    MOTOR_ID = 1  
    
    try:
        motor = DM_USB2CAN_Serial()
        
        print("1. 使能电机...")
        motor.CanComm_ControlCmd("ENABLE", MOTOR_ID)
        time.sleep(1)
        
        print("2. 执行暴力位置控制：正向弹簧 (位置 +2.0, KP=10)...")
        # 这里的 KP=10 相当于一根很强的弹簧，会把电机强行拉到 2.0 弧度的位置
        for _ in range(50):
            motor.CanComm_SendControlPara(2.0, 0.0, 10.0, 1.0, 0.0, MOTOR_ID)
            time.sleep(0.02)
            
        print("3. 执行暴力位置控制：反向弹簧 (位置 -2.0, KP=10)...")
        for _ in range(50):
            motor.CanComm_SendControlPara(-2.0, 0.0, 10.0, 1.0, 0.0, MOTOR_ID)
            time.sleep(0.02)
            
        print("4. 归位到 0...")
        for _ in range(50):
            motor.CanComm_SendControlPara(0.0, 0.0, 10.0, 1.0, 0.0, MOTOR_ID)
            time.sleep(0.02)
            
    except Exception as e:
        print(f"出现错误: {e}")
        
    finally:
        print("5. 失能电机...")
        motor.CanComm_ControlCmd("DISABLE", MOTOR_ID)
        print("测试结束！")
