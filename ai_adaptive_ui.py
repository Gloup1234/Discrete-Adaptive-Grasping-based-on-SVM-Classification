#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 认知与自适应夹取 UI 控制系统 (滑脱补偿进阶版)
特性: 动态去皮、特征提取实时推理、高频微滑脱脊髓反射补偿
"""

import sys
import time
import serial
import numpy as np
import pandas as pd
import joblib
import traceback
import threading
import os

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QProgressBar,
                             QGroupBox, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont

from collections import deque
from damiao import *

# 确保能找到 modules 文件夹
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from modules.feature_extraction import FeatureExtractor

# ================== 硬件与控制配置 ==================
SENSOR_PORT = '/dev/ttyACM1'  # 传感器串口，请根据实际情况确认
BAUD = 115200

CAN_ID = 0x01
MST_ID = 0x11
OPEN_TORQUE = -0.25
APPROACH_TORQUE = 0.25
SQUEEZE_TORQUE = 0.50

# 自适应保持扭矩配置 (AI 推理基础值)
HOLD_TORQUE_SOFT = 0.15   
HOLD_TORQUE_HARD = 0.50   

# 安全参数
SAFE_FZ_LIMIT = 15.0
RECORD_DURATION = 1.5
FEATURE_WIN = 100
EPS = 1e-6

# ================== 后台控制与推理线程 ==================
class AIGraspThread(QThread):
    log_signal = pyqtSignal(str)
    force_signal = pyqtSignal(float)
    state_signal = pyqtSignal(str)
    ai_result_signal = pyqtSignal(str, float) 
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.motor_enabled = False
        self.current_state = "IDLE"
        self.target_torque = 0.0
        
        self.control = None
        self.sensor = None
        
        # 实时监控滑动窗口
        self.fx_buf = deque(maxlen=FEATURE_WIN)
        self.fy_buf = deque(maxlen=FEATURE_WIN)
        self.fz_buf = deque(maxlen=FEATURE_WIN)
        self.ft_buf = deque(maxlen=FEATURE_WIN)
        self.mu_buf = deque(maxlen=FEATURE_WIN)
        
        # 录制给 AI 推理用的去皮净数据
        self.ai_record_frames = []      
        
        # 动作指令标志位
        self.cmd_start_grasp = False
        self.cmd_release = False
        self.cmd_emergency = False
        
        # 动态去皮(Tare)基线
        self.baseline_fz = 0.0
        self.baseline_ft = 0.0
        
        # 加载 AI 大脑
        try:
            self.extractor = FeatureExtractor(window_size=FEATURE_WIN)
            model_data = joblib.load('models/classifier_svm.pkl')
            self.model = model_data.get('model')
            self.scaler = model_data.get('scaler')
            self.label_encoder = model_data.get('label_encoder')
            self.feature_names = model_data.get('feature_names')
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.model = None

    def motor_heartbeat_loop(self):
        """绝对不被阻塞的 50Hz 电机力矩下发线程"""
        while self.running:
            if self.motor_enabled and self.control is not None:
                try:
                    self.control.control_mit(self.control.getMotor(CAN_ID), 0.0, 0.0, 0.0, 0.0, self.target_torque)
                except Exception:
                    pass
            time.sleep(0.02)

    def init_hardware(self):
        try:
            self.sensor = serial.Serial(SENSOR_PORT, BAUD, timeout=0.1)
            time.sleep(1)
            self.sensor.write(b"100 Hz,1")
            self.log_signal.emit("✅ 传感器连接成功")
            
            init_data = [DmActData(motorType=DM_Motor_Type.DM4310, mode=Control_Mode.MIT_MODE, can_id=CAN_ID, mst_id=MST_ID)]
            self.control = Motor_Control(1000000, 5000000, "91F3E4730611D4E059B68138C08043B3", init_data)
            self.control.switchControlMode(self.control.getMotor(CAN_ID), Control_Mode_Code.MIT)
            self.control.enable_all()
            self.motor_enabled = True
            
            # 启动电机心跳守护线程
            threading.Thread(target=self.motor_heartbeat_loop, daemon=True).start()
            
            self.log_signal.emit("✅ 电机初始化成功")
            return True
        except Exception as e:
            self.log_signal.emit(f"❌ 硬件初始化失败: {str(e)}")
            return False

    def run(self):
        if not self.init_hardware():
            return
            
        start_record_time = 0
        self.target_torque = OPEN_TORQUE
        time.sleep(0.5)
        self.target_torque = 0.0
        
        while self.running:
            # --- 处理外部指令 ---
            if self.cmd_start_grasp and self.current_state == "IDLE":
                self.cmd_start_grasp = False
                
                # 【动态去皮】：捕获寻找物体前的空气阻力
                self.baseline_fz = np.mean(list(self.fz_buf)[-10:]) if len(self.fz_buf) >= 10 else 0.0
                self.baseline_ft = np.mean(list(self.ft_buf)[-10:]) if len(self.ft_buf) >= 10 else 0.0
                
                self.current_state = "APPROACH"
                self.target_torque = APPROACH_TORQUE
                self.state_signal.emit("APPROACH")
                self.log_signal.emit(f"⚙️ 寻找物体 (已归零, 抵消应力 Fz_base={self.baseline_fz:.2f}N)")
                
            if self.cmd_release:
                self.cmd_release = False
                self.current_state = "IDLE"
                self.target_torque = OPEN_TORQUE
                self.state_signal.emit("RELEASING")
                time.sleep(0.5)
                self.target_torque = 0.0
                self.state_signal.emit("IDLE")
                self.log_signal.emit("💤 夹爪已张开待命")
                self.ai_result_signal.emit("WAITING", 0.0)
                
            if self.cmd_emergency:
                self.cmd_emergency = False
                self.log_signal.emit("🚨 触发紧急硬件重置！")
                self.execute_emergency_reset()
                continue

            # --- 读取传感器数据 ---
            try:
                line = self.sensor.readline().decode('utf-8', errors='ignore').strip()
            except Exception:
                time.sleep(0.01)
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
            
            self.fx_buf.append(fx); self.fy_buf.append(fy); self.fz_buf.append(fz)
            self.ft_buf.append(ft); self.mu_buf.append(mu)
            
            # UI力条依然显示绝对力以便于物理观察
            self.force_signal.emit(abs(fz))

            # --- 硬件过载保护 (基于绝对力) ---
            if abs(fz) > SAFE_FZ_LIMIT and self.current_state != "IDLE":
                self.log_signal.emit(f"🚨 过载保护: {fz:.2f}N！")
                self.execute_emergency_reset()
                continue

            # --- 计算净受力 (用于逻辑触发和AI推理) ---
            net_fz = fz - self.baseline_fz
            net_ft = max(0, ft - self.baseline_ft)
            net_mu = net_ft / (abs(net_fz) + EPS)

            # --- 核心状态机与推理闭环 ---
            if len(self.fz_buf) >= FEATURE_WIN:
                if self.current_state == "APPROACH":
                    # 使用【净受力】判断，彻底消灭空气触发的 Bug
                    if abs(net_fz) > 0.5: 
                        self.current_state = "SQUEEZE"
                        start_record_time = time.time()
                        
                        self.ai_record_frames.clear()
                        self.state_signal.emit("SQUEEZING")
                        self.log_signal.emit(f"💥 接触物体 (净受力 {abs(net_fz):.2f}N)，开始采集...")

                elif self.current_state == "SQUEEZE":
                    elapsed_time = time.time() - start_record_time
                    
                    # 记录去皮后的净波形供 AI 提取特征
                    self.ai_record_frames.append({
                        'Fz': abs(net_fz), 
                        'Ft': net_ft, 
                        'mu': net_mu
                    })
                    
                    # 柔顺爬坡控制
                    if elapsed_time < 0.4:
                        torque_step = (SQUEEZE_TORQUE - APPROACH_TORQUE) * (elapsed_time / 0.4)
                        self.target_torque = APPROACH_TORQUE + torque_step
                    else:
                        self.target_torque = SQUEEZE_TORQUE
                        
                    if elapsed_time >= RECORD_DURATION:
                        self.current_state = "INFERENCE"
                        self.state_signal.emit("INFERENCE")
                        
                        # [注释掉 A/B 测试的 Debug 文件保存，减少磁盘 IO]
                        # df_debug = pd.DataFrame(self.debug_raw_frames)
                        # debug_filename = f"debug_ui_{int(time.time())}.csv"
                        # df_debug.to_csv(debug_filename, index=False)
                        # self.log_signal.emit(f"💾 A/B 原生数据已保存: {debug_filename}")
                        
                        df_realtime = pd.DataFrame(self.ai_record_frames)
                        
                        if self.model is not None and len(df_realtime) >= 10:
                            try:
                                features = self.extractor.extract_features_from_window(df_realtime)
                                features_df = pd.DataFrame([features])
                                
                                if self.feature_names is not None:
                                    features_df = features_df[self.feature_names]
                                
                                if self.scaler is not None:
                                    X_pred = self.scaler.transform(features_df)
                                else:
                                    X_pred = features_df.values
                                    
                                pred_raw = self.model.predict(X_pred)[0]
                                
                                if self.label_encoder is not None:
                                    prediction = self.label_encoder.inverse_transform([pred_raw])[0]
                                else:
                                    prediction = pred_raw
                                
                                self.log_signal.emit(f"🧠 推理完成 | fz_max: {features.get('fz_max',0):.2f}N, E_absorb: {features.get('energy_absorb',0):.2f}")
                                
                                if prediction == "Material_Soft":
                                    self.target_torque = HOLD_TORQUE_SOFT
                                    self.log_signal.emit(f"🟢 AI 判定: 【软体】 -> 降扭矩至 {HOLD_TORQUE_SOFT}N·m")
                                else:
                                    self.target_torque = HOLD_TORQUE_HARD
                                    self.log_signal.emit(f"🔴 AI 判定: 【硬体】 -> 锁定扭矩至 {HOLD_TORQUE_HARD}N·m")
                                    
                                self.ai_result_signal.emit(prediction, self.target_torque)
                                
                            except Exception as e:
                                self.log_signal.emit(f"❌ 推理出错: {str(e)}\n{traceback.format_exc()}")
                        else:
                            self.target_torque = HOLD_TORQUE_HARD
                            
                        # 进入自适应保持与防滑监控状态
                        self.current_state = "ADAPTIVE_HOLD"
                        self.state_signal.emit("HOLDING")
                        # 清空缓冲，为即将到来的防滑监控准备干净的数据
                        self.ft_buf.clear() 

                # ======== 👇 第一阶段：脊髓反射 (动态微滑脱补偿) 👇 ========
                elif self.current_state == "ADAPTIVE_HOLD":
                    # 利用高频传感器，每收集 10 帧 (约 0.1秒) 进行一次滑动检测
                    if len(self.ft_buf) >= 10:
                        recent_ft = list(self.ft_buf)[-10:]
                        # 计算 0.1秒内横向滑动力的突变差值
                        ft_derivative = recent_ft[-1] - recent_ft[0]
                        
                        # 阈值 0.15N 可根据实际传感器的摩擦灵敏度进行微调
                        if ft_derivative > 0.15:
                            # 触发防滑脱补偿：力矩瞬间增加 0.05 N·m，最高不超过 0.70N·m 保护电机
                            self.target_torque = min(self.target_torque + 0.05, 0.70)
                            
                            self.log_signal.emit(f"⚠️ 滑脱预警! (dFt={ft_derivative:.2f}N) -> 捏紧补偿至 {self.target_torque:.2f}N·m")
                            self.ai_result_signal.emit("Slip Compensating...", self.target_torque)
                            
                            # 补偿后清空缓冲区，给予夹爪 0.05 秒的机械稳定冷却期，防止鬼畜抖动
                            self.ft_buf.clear()
                            time.sleep(0.05)
                # ==========================================================

    def execute_emergency_reset(self):
        self.current_state = "IDLE"
        self.motor_enabled = False 
        self.state_signal.emit("ERROR/RESET")
        
        if self.sensor:
            try:
                self.sensor.close()
                time.sleep(0.5)
                self.sensor.open()
                self.sensor.write(b"100 Hz,1")
                self.log_signal.emit("✅ 传感器底层连接已重启")
            except Exception as e:
                self.log_signal.emit(f"❌ 传感器重启失败: {e}")

        if self.control:
            try:
                self.control.disable_all()
                time.sleep(0.5)
                self.control.enable_all()
                self.target_torque = OPEN_TORQUE
                self.motor_enabled = True 
                time.sleep(1.0)
                self.target_torque = 0.0
            except:
                pass
                
        self.fx_buf.clear(); self.fy_buf.clear(); self.fz_buf.clear(); self.ft_buf.clear()
        self.ai_record_frames.clear()
        self.state_signal.emit("IDLE")
        self.log_signal.emit("✅ 系统硬重置完成")

    def stop(self):
        self.running = False
        time.sleep(0.1) 
        if self.control and self.motor_enabled:
            try:
                self.control.disable_all()
                time.sleep(0.1)
                self.control.__exit__(None, None, None)
            except:
                pass
        if self.sensor:
            self.sensor.close()

# ================== PyQt5 主界面 ==================
class AIGripperUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 认知与自适应控制终端")
        self.resize(800, 600)
        
        self.init_ui()
        
        self.backend = AIGraspThread()
        self.backend.log_signal.connect(self.update_log)
        self.backend.force_signal.connect(self.update_force)
        self.backend.state_signal.connect(self.update_state)
        self.backend.ai_result_signal.connect(self.update_ai_result)
        self.backend.start()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        left_panel = QVBoxLayout()
        
        status_group = QGroupBox("实时系统状态")
        status_layout = QVBoxLayout()
        self.lbl_state = QLabel("系统待命 (IDLE)")
        self.lbl_state.setFont(QFont("Arial", 16, QFont.Bold))
        self.lbl_state.setAlignment(Qt.AlignCenter)
        self.lbl_state.setStyleSheet("color: white; background-color: #34495e; padding: 10px; border-radius: 5px;")
        status_layout.addWidget(self.lbl_state)
        
        self.bar_force = QProgressBar()
        self.bar_force.setRange(0, 1500)
        self.bar_force.setTextVisible(True)
        self.bar_force.setFormat("绝对法向受力 Fz: %v / 1500")
        self.bar_force.setStyleSheet("QProgressBar::chunk { background-color: #e74c3c; }")
        status_layout.addWidget(self.bar_force)
        status_group.setLayout(status_layout)
        
        ai_group = QGroupBox("触觉认知与控制追踪")
        ai_layout = QVBoxLayout()
        self.lbl_ai = QLabel("等待抓取...")
        self.lbl_ai.setFont(QFont("Arial", 22, QFont.Bold))
        self.lbl_ai.setAlignment(Qt.AlignCenter)
        self.lbl_ai.setStyleSheet("color: #7f8c8d; background-color: #ecf0f1; padding: 20px; border-radius: 8px;")
        
        self.lbl_torque = QLabel("当前目标力矩: 0.00 N·m")
        self.lbl_torque.setFont(QFont("Arial", 14))
        self.lbl_torque.setAlignment(Qt.AlignCenter)
        
        ai_layout.addWidget(self.lbl_ai)
        ai_layout.addWidget(self.lbl_torque)
        ai_group.setLayout(ai_layout)
        
        left_panel.addWidget(status_group)
        left_panel.addWidget(ai_group)
        
        right_panel = QVBoxLayout()
        
        ctrl_group = QGroupBox("指令控制台")
        ctrl_layout = QVBoxLayout()
        
        btn_start = QPushButton("启动抓取 [S]")
        btn_start.setMinimumHeight(60)
        btn_start.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; font-size: 14px;")
        btn_start.clicked.connect(lambda: setattr(self.backend, 'cmd_start_grasp', True))
        
        btn_release = QPushButton("张开夹爪 [R]")
        btn_release.setMinimumHeight(60)
        btn_release.setStyleSheet("background-color: #f39c12; color: white; font-weight: bold; font-size: 14px;")
        btn_release.clicked.connect(lambda: setattr(self.backend, 'cmd_release', True))
        
        btn_estop = QPushButton("紧急重置 [E]")
        btn_estop.setMinimumHeight(60)
        btn_estop.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold; font-size: 14px;")
        btn_estop.clicked.connect(self.trigger_emergency)
        
        ctrl_layout.addWidget(btn_start)
        ctrl_layout.addWidget(btn_release)
        ctrl_layout.addWidget(btn_estop)
        ctrl_group.setLayout(ctrl_layout)
        
        log_group = QGroupBox("系统日志")
        log_layout = QVBoxLayout()
        self.lbl_log = QLabel("系统启动中...")
        self.lbl_log.setWordWrap(True)
        self.lbl_log.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.lbl_log.setStyleSheet("background-color: black; color: #00ff00; padding: 5px; font-family: Consolas;")
        log_layout.addWidget(self.lbl_log)
        log_group.setLayout(log_layout)
        
        right_panel.addWidget(ctrl_group, 1)
        right_panel.addWidget(log_group, 2)
        
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)

    def trigger_emergency(self):
        self.backend.cmd_emergency = True
        if self.backend.sensor and self.backend.sensor.is_open:
            try:
                self.backend.sensor.close()
            except:
                pass

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S:
            self.backend.cmd_start_grasp = True
        elif event.key() == Qt.Key_R:
            self.backend.cmd_release = True
        elif event.key() == Qt.Key_E:
            self.trigger_emergency()

    def update_log(self, text):
        current_text = self.lbl_log.text()
        lines = current_text.split('\n')
        if len(lines) > 15:
            lines = lines[-15:]
        lines.append(time.strftime("[%H:%M:%S] ") + text)
        self.lbl_log.setText('\n'.join(lines))

    def update_force(self, fz):
        self.bar_force.setValue(int(fz * 100))
        self.bar_force.setFormat(f"绝对法向受力 Fz: {fz:.2f} N")

    def update_state(self, state):
        self.lbl_state.setText(f"系统状态: {state}")
        if state == "SQUEEZING":
            self.lbl_state.setStyleSheet("color: white; background-color: #d35400; padding: 10px; border-radius: 5px;")
        elif state == "HOLDING":
            self.lbl_state.setStyleSheet("color: white; background-color: #27ae60; padding: 10px; border-radius: 5px;")
        else:
            self.lbl_state.setStyleSheet("color: white; background-color: #34495e; padding: 10px; border-radius: 5px;")

    def update_ai_result(self, result, torque):
        if result == "Material_Soft":
            self.lbl_ai.setText("柔软易损物 (Soft)")
            self.lbl_ai.setStyleSheet("color: white; background-color: #2ecc71; padding: 20px; border-radius: 8px;")
        elif result == "Material_Hard":
            self.lbl_ai.setText("坚硬刚体 (Hard)")
            self.lbl_ai.setStyleSheet("color: white; background-color: #e74c3c; padding: 20px; border-radius: 8px;")
        elif "Slip" in result:
            self.lbl_ai.setText("⚠️ 触发防滑脱保护！")
            self.lbl_ai.setStyleSheet("color: white; background-color: #f39c12; padding: 20px; border-radius: 8px;")
        else:
            self.lbl_ai.setText("等待抓取...")
            self.lbl_ai.setStyleSheet("color: #7f8c8d; background-color: #ecf0f1; padding: 20px; border-radius: 8px;")
            
        self.lbl_torque.setText(f"当前目标力矩: {torque:.2f} N·m")

    def closeEvent(self, event):
        reply = QMessageBox.question(self, '退出', "确认关闭系统？", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.backend.stop()
            self.backend.wait()
            event.accept()
        else:
            event.ignore()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AIGripperUI()
    window.show()
    sys.exit(app.exec_())
