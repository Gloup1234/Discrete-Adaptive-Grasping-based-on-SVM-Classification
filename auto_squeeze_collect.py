#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电机自动化挤压与材料特征采集系统 (安全增强版)
实现：张开 -> 试探闭合 -> 接触检测 -> 柔性加载(防断裂) -> 稳态挤压录制 -> 保持(防掉落)
"""

import sys
import time
import serial
import numpy as np
import csv
import os
import keyboard
import threading
from collections import deque

# 引入达妙 SDK 与项目配置
from damiao import *
import Config

# ================== 硬件配置 ==================
# 传感器配置
SENSOR_PORT = '/dev/ttyACM1'  # Linux 下的串口
BAUD = 115200

# 电机配置
CAN_ID = 0x01
MST_ID = 0x11
OPEN_TORQUE = -0.25    # 张开时的力矩
APPROACH_TORQUE = 0.25 # 寻找物体时的试探闭合力矩
SQUEEZE_TORQUE = 0.50  # 接触后维持挤压的稳定力矩 (最高力矩)

# ================== 采集与安全参数 ==================
SAMPLE_RATE = 100
FEATURE_WIN = 100      # 1秒钟滑动窗口
RECORD_DURATION = 1.5  # 挤压录制时长
EPS = 1e-6
FZ_CONTACT = 0.5       # 接触触发阈值 (N)

# --- 新增：安全与柔顺控制参数 ---
RAMP_DURATION = 0.4    # 软启动时间(秒)：用0.4秒平滑过渡到最大扭矩，消除机械冲击
SAFE_FZ_LIMIT = 15.0   # 极限保护力(N)：超过此力瞬间强制卸力，保护3D打印件和传感器

# ================== 数据缓冲 ==================
fx_buf, fy_buf, fz_buf = deque(maxlen=FEATURE_WIN), deque(maxlen=FEATURE_WIN), deque(maxlen=FEATURE_WIN)
ft_buf, mu_buf, dfz_buf = deque(maxlen=FEATURE_WIN), deque(maxlen=FEATURE_WIN), deque(maxlen=FEATURE_WIN)
recording_data = []

# ================== 数据保存配置 ==================
# ⚠️ 测试新材料时修改这里
BASE_MATERIAL_NAME = "Test" 
MATERIAL_LABEL = f"Material_{BASE_MATERIAL_NAME}"
os.makedirs(Config.RAW_DIR, exist_ok=True)

# ================== 全局状态 ==================
current_state = "IDLE"  # 状态: IDLE, APPROACH, SQUEEZE, HOLD
target_torque = 0.0
motor_running = True

# ==================== 电机控制线程 ====================
def motor_control_loop(control, canid):
    """独立线程：带动态限流与底层保护的力矩下发"""
    global target_torque, motor_running, current_state
    
    time.sleep(1.0) # 初始化缓冲
    
    while motor_running:
        try:
            if control is not None:
                control.control_mit(control.getMotor(canid), 0.0, 0.0, 0.0, 0.0, target_torque)
        except Exception:
            pass 
            
        if current_state == "IDLE":
            time.sleep(0.1) # 待命时降频，防拥堵
        else:
            time.sleep(0.02) # 动作时50Hz丝滑下发

    try:
        if control is not None:
            control.control_mit(control.getMotor(canid), 0.0, 0.0, 0.0, 0.0, 0.0)
    except:
        pass

# ==================== 主流程 ====================
def main():
    global current_state, target_torque, motor_running
    
    print("\n" + "="*50)
    print(" 🤖 自动化挤压采集系统 (防冲击安全版) 🤖 ".center(46))
    print("="*50)
    
    try:
        sensor = serial.Serial(SENSOR_PORT, BAUD, timeout=0.1)
        time.sleep(2)
        sensor.write(b"100 Hz,1")
        time.sleep(1)
        print("✅ 传感器连接成功！")
    except Exception as e:
        print(f"❌ 传感器连接失败: {e}")
        return

    try:
        init_data = [DmActData(motorType=DM_Motor_Type.DM4310, mode=Control_Mode.MIT_MODE, can_id=CAN_ID, mst_id=MST_ID)]
        control = Motor_Control(1000000, 5000000, "91F3E4730611D4E059B68138C08043B3", init_data)
        control.switchControlMode(control.getMotor(CAN_ID), Control_Mode_Code.MIT)
        control.enable_all()
        
        motor_thread = threading.Thread(target=motor_control_loop, args=(control, CAN_ID), daemon=True)
        motor_thread.start()
        print("✅ 电机初始化并使能成功！已进入纯力矩模式。")
    except Exception as e:
        print(f"❌ 电机初始化失败: {e}")
        sensor.close()
        return

    last_fz = None
    start_record_time = 0
    csv_file, csv_writer = None, None
    csv_filepath = ""
    
    target_torque = OPEN_TORQUE 
    time.sleep(0.5)
    target_torque = 0.0 
    
    print(f"📂 当前录制目标: {MATERIAL_LABEL}")
    print(f"🛡️  安全阈值设定: 极限力 {SAFE_FZ_LIMIT}N | 柔性爬坡 {RAMP_DURATION}秒")
    print("\n操作指南:")
    print("  [s] 启动一次标准化夹取")
    print("  [r] 手动/强制张开夹爪 (Release)")
    print("  [q] 退出程序")

    try:
        while True:
            # --- 键盘状态机 ---
            if keyboard.is_pressed('s') and current_state == "IDLE":
                # 如果当前受力超过 1N，拒绝执行闭合，防止带着大负载启动
                if len(fz_buf) > 0 and abs(fz_buf[-1]) > 1.0:
                    print(f"\n⚠️ [警告] 当前传感器仍受力 {abs(fz_buf[-1]):.2f}N，请先按 [r] 张开或移除物体！")
                    time.sleep(0.5)
                else:
                    current_state = "APPROACH"
                    target_torque = APPROACH_TORQUE
                    print("\n⚙️ [动作] 开始寻找物体...")
                    time.sleep(0.3)
                
            if keyboard.is_pressed('r'):
                current_state = "IDLE"
                target_torque = OPEN_TORQUE 
                print("\n⚙️ [动作] 强制张开/复位夹爪...")
                time.sleep(0.5)
                target_torque = 0.0
                print("💤 [待命] 已准备好。")
                time.sleep(0.3) 
                
            if keyboard.is_pressed('q'):
                print("\n正在退出系统...")
                break
	# ======== 👇 新增：[e] 键 彻底热重启 USB 通信 👇 ========
            if keyboard.is_pressed('e'):
                print("\n🔌 [通信恢复] 正在热重启电机底层通信...")
                current_state = "IDLE"
                target_torque = 0.0
                
                # 1. 停止原控制线程
                motor_running = False
                time.sleep(0.5) # 等待线程自然死亡
                
                # 2. 清理并强杀旧控制器
                try:
                    if control is not None:
                        control.disable_all()
                        control.__exit__(None, None, None) # 强制释放 USB 资源
                except:
                    pass
                
                time.sleep(1.5) # 给 Linux 系统内核一点时间回收 USB 端口
                
                # 3. 重新建立连接
                try:
                    print("   -> 正在重新接管 USB2CANFD 硬件...")
                    # 重新生成初始化数据
                    init_data = [DmActData(motorType=DM_Motor_Type.DM4310, mode=Control_Mode.MIT_MODE, can_id=CAN_ID, mst_id=MST_ID)]
                    control = Motor_Control(1000000, 5000000, "91F3E4730611D4E059B68138C08043B3", init_data)
                    control.switchControlMode(control.getMotor(CAN_ID), Control_Mode_Code.MIT)
                    control.enable_all()
                    
                    # 4. 重启控制线程
                    motor_running = True
                    motor_thread = threading.Thread(target=motor_control_loop, args=(control, CAN_ID), daemon=True)
                    motor_thread.start()
                    
                    print("✅ 热重启成功！通信已恢复，按 [r] 张开夹爪。")
                    time.sleep(0.5) # 防误触缓冲
                except Exception as e:
                    print(f"❌ 热重启失败！虚拟机 USB 彻底锁死，请拔插 USB 线后按 [q] 重启程序: {e}")
                
                continue # 跳过本次循环，防止读取错误数据
            # --- 读取传感器 ---
            try:
                line = sensor.readline().decode('utf-8', errors='ignore').strip()
            except Exception:
                continue
                
            if not line: continue
            parts = line.split()
            if len(parts) < 5: continue

            try:
                fy, fx, fz = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                continue

            ft = np.sqrt(fx**2 + fy**2)
            mu = ft / (abs(fz) + EPS)
            dfz = (fz - last_fz) * SAMPLE_RATE if last_fz is not None else 0
            last_fz = fz

            fx_buf.append(fx); fy_buf.append(fy); fz_buf.append(fz)
            ft_buf.append(ft); mu_buf.append(mu); dfz_buf.append(dfz)

            # ======== 🛡️ 硬件级超载瞬时保护与底层重置 ========
            if abs(fz) > SAFE_FZ_LIMIT and current_state != "IDLE":
                current_state = "IDLE"
                target_torque = 0.0
                print(f"\n🚨 [紧急停止] 受力 {fz:.2f}N 突破极限 ({SAFE_FZ_LIMIT}N)！")
                print("🔄 正在触发电机底层硬件重置...")
                
                try:
                    # 1. 强制失能切断电流
                    if control is not None:
                        control.disable_all() 
                    time.sleep(0.5)
                    
                    # 2. 重新使能并下发张开指令
                    if control is not None:
                        control.enable_all()
                        target_torque = OPEN_TORQUE  # 强制反转张开
                        time.sleep(1.0)              # 【关键】给予1秒钟的物理张开时间
                        target_torque = 0.0          # 卸力待命
                    
                    # 3. 彻底清空传感器陈旧数据 (核心修复)
                    sensor.reset_input_buffer()      # 清空 Linux 底层串口堆积的旧数据
                    fx_buf.clear(); fy_buf.clear(); fz_buf.clear()
                    ft_buf.clear(); mu_buf.clear(); dfz_buf.clear()
                    last_fz = None                   # 重置差分计算基准
                        
                    print("✅ 电机已物理张开，传感器脏数据已清空，恢复安全待命。")
                except Exception as e:
                    print(f"❌ 电机重置失败，请手动重启程序: {e}")
                    motor_running = False 
                    break
                    
                continue
            # ==================================================

            if current_state == "APPROACH":
                print(f"📈 监控 -> 缓冲: {len(fz_buf)}/100, Fz: {fz:.2f} N", end='\r')

            # --- 核心状态机 ---
            if len(fz_buf) >= FEATURE_WIN:
                
                # 状态 1：碰触物体
                if current_state == "APPROACH":
                    if abs(fz) > FZ_CONTACT:
                        current_state = "SQUEEZE"
                        start_record_time = time.time()
                        recording_data.clear()
                        
                        import glob
                        existing_files = glob.glob(os.path.join(Config.RAW_DIR, f"{MATERIAL_LABEL}_*.csv"))
                        file_idx = len(existing_files) + 1
                        csv_filepath = os.path.join(Config.RAW_DIR, f"{MATERIAL_LABEL}_{file_idx}.csv")
                        
                        csv_file = open(csv_filepath, "w", newline="")
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow(["time","Fx","Fy","Fz","Ft","mu","k_eff","mu_mean","mu_std","slip","micro","material"])
                        
                        print(f"\n💥 [接触] 触发点: {fz:.2f}N。启动柔顺加载... (文件: {file_idx})")

                # 状态 2：挤压与录制中
                elif current_state == "SQUEEZE":
                    elapsed_time = time.time() - start_record_time
                    
                    # ======== 🌊 扭矩柔性爬坡算法 ========
                    if elapsed_time < RAMP_DURATION:
                        torque_step = (SQUEEZE_TORQUE - APPROACH_TORQUE) * (elapsed_time / RAMP_DURATION)
                        target_torque = APPROACH_TORQUE + torque_step
                    else:
                        target_torque = SQUEEZE_TORQUE
                    # ======================================

                    fz_win = np.array(list(fz_buf))
                    ft_win = np.array(list(ft_buf))
                    dfz_win = np.array(list(dfz_buf))
                    mu_win = np.array(list(mu_buf))

                    k_eff = np.mean(np.abs(dfz_win))
                    mu_mean = np.mean(mu_win)
                    mu_std = np.std(mu_win)
                    slip = np.max(np.abs(dfz_win))
                    micro = np.sqrt(np.mean((ft_win - np.mean(ft_win))**2))

                    recording_data.append([
                        time.time(), fx, fy, fz, ft, mu, 
                        k_eff, mu_mean, mu_std, slip, micro, f"{MATERIAL_LABEL}_{file_idx}"
                    ])
                    
                    # 录制结束条件
                    if elapsed_time >= RECORD_DURATION:
                        current_state = "HOLD"
                        
                        for row in recording_data:
                            csv_writer.writerow(row)
                        csv_file.close()
                        
                        print(f"✅ [完成] 录制 {len(recording_data)} 帧结束！")
                        print("✋ 挤压已锁定！请用手接住物品，然后按 [r] 键张开夹爪。\n")

    except KeyboardInterrupt:
        print("\n⚠️ 强制终止。")
    finally:
        print("清理资源...")
        motor_running = False
        if csv_file and not csv_file.closed:
            csv_file.close()
        sensor.close()
        if control:
            control.disable_all()
            time.sleep(0.1)
            control.__exit__(None, None, None)
        print("安全退出。")

if __name__ == "__main__":
    main()
