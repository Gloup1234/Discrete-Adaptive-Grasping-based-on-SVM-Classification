import os

# 基础目录配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据目录
RAW_DIR = os.path.join(BASE_DIR, "data_raw")
PREPROCESS_DIR = os.path.join(BASE_DIR, "data_preprocess")
FEATURES_DIR = os.path.join(BASE_DIR, "data_features")

# 模型目录
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 可视化输出目录
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, "visualizations")

# 用于预测的数据目录
PREDICTION_RAW_DIR = os.path.join(BASE_DIR, "prediction_raw")
PREDICTION_PREPROCESS_DIR = os.path.join(BASE_DIR, "prediction_preprocessed")