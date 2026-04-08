# AI-Driven Adaptive Grasping System (Phase 1)
**基于多维触觉反馈与机器学习的智能自适应抓取系统（第一阶段：离散力控）**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyQt5](https://img.shields.io/badge/PyQt5-UI-green.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-SVM-orange.svg)

## 📌 项目概述 (Project Overview)
本项目旨在解决传统机器人夹爪在面对未知刚度、易碎物体时容易发生“夹碎”或“滑脱”的行业痛点。通过集成高频三轴力觉传感器与基于 MIT 模式的直接力矩控制电机，构建了一套**“感知 -> 认知 -> 决策 -> 执行”**的闭环控制系统。

本仓库保存的是**阶段一（Phase 1）**的代码。系统通过 SVM (支持向量机) 对接触瞬间的力学波形进行分类推理，将物体实时判别为“硬质刚体 (Hard)”或“柔性易损物 (Soft)”，并动态下发离散的安全保持扭矩。

## ✨ 核心特性 (Key Features)
- **动态去皮 (Dynamic Tare):** 在每次抓取指令下发瞬间捕捉环境零点，彻底免疫传感器因温度或机械装配产生的基线漂移。
- **高保真物理特征工程:** 摒弃黑盒深度学习，通过滑动窗口提取净最大受力 (`fz_max`)、瞬态刚度 (`ramp_slope`)、能量吸收比 (`energy_absorb`) 等具有强物理可解释性的时域特征。
- **异步多线程架构:** 基于 PyQt5 构建的独立线程管线。完美分离 UI 渲染、传感器 100Hz 串口无阻塞读取、以及 CAN 总线 50Hz 无间断电机心跳指令下发。
- **Sim-to-Real 零偏差落地:** 离线管线与实时控制管线共享同一套特征提取类，实现了训练环境与现场部署 100% 的物理逻辑对齐。

## ⚙️ 硬件依赖 (Hardware Setup)
- **执行器:** 达妙 DM4310 动力电机 (经 CAN 总线透传，使用 MIT 控制模式)
- **传感器:** 自研/定制高灵敏度三轴触觉传感器 (串口通信，100Hz 采样率)
- **机械结构:** 刚性平行连杆夹爪

## 📂 核心文件目录 (Core Structure)
```text
├── ai_adaptive_ui.py            # 核心：实时闭环控制中枢与 PyQt5 可视化界面
├── auto_squeeze_collect.py      # 采集：基于状态机的自动化高保真数据采集脚本
├── main_pipeline.py             # 训练：标准化的模型预处理、特征提取与评估管线
├── predict_pipeline.py          # 测试：针对离线特征的预测验证流水线
├── modules/
│   └── feature_extraction.py    # 算法：物理特征提取模块（消除零点漂移，提取刚度/阻尼特征）
└── damiao/                      # 底层：达妙电机驱动 SDK
