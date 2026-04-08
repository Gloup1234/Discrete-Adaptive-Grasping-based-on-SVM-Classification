"""
特征提取模块
基于针对"恒定挤压"动作优化的物理特征：
1. k_max - 瞬态最大刚度 (捕捉接触瞬间的硬度斜率)
2. fz_integral - 受力积分做功 (区分软材料的缓慢上升与硬材料的瞬间达到峰值)
3. fz_std - 稳态阻尼吸收 (评估夹紧后的高频震荡，硬材料方差大)
4. micro - 泊松横向微震动 (辅助形变特征)
5. mu_mean - 平均摩擦系数 (辅助特征)
"""

import pandas as pd
import numpy as np
import os
import sys

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import Config

class FeatureExtractor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        
    def extract_features_from_window(self, df_window):
        features = {}
        if len(df_window) < 20: 
            return {'fz_max': 0.0, 'ramp_slope': 0.0, 'steady_std': 0.0, 'energy_absorb': 0.0, 'mu_mean': 0.0}
            
        fz_raw = df_window['Fz'].values
        mu_values = df_window['mu'].values
        
        # ======== 🛡️ 核心生命线：强制去皮 ========
        # 彻底消除环境零点漂移，只提取水瓶纯粹的形变净受力
        fz_values = np.abs(fz_raw - fz_raw[0])
        # ==========================================
        
        # 1. 净最大受力 (fz_max)
        features['fz_max'] = np.max(fz_values)
        
        # 2. 宏观爬坡斜率 (ramp_slope)
        ramp_end_idx = min(40, len(fz_values) - 1)
        if ramp_end_idx > 0:
            # 由于 fz_values 已经去皮归零，这里直接取值相除即可
            features['ramp_slope'] = fz_values[ramp_end_idx] / (ramp_end_idx * 0.01) 
        else:
            features['ramp_slope'] = 0.0
        
        # 3. 稳态阻尼特性 (steady_std)
        steady_start_idx = max(0, len(fz_values) - 50)
        features['steady_std'] = np.std(fz_values[steady_start_idx:])
        
        # 4. 能量吸收比 (energy_absorb)
        if features['fz_max'] > 0:
            features['energy_absorb'] = np.mean(fz_values) / features['fz_max']
        else:
            features['energy_absorb'] = 0.0
            
        # 5. 稳态摩擦系数 (mu_mean)
        features['mu_mean'] = np.mean(mu_values[steady_start_idx:])
        
        return features
    
    def extract_features_sliding_window(self, df, overlap=0.5):
        step = int(self.window_size * (1 - overlap))
        features_list = []
        
        material_label = None
        if 'material' in df.columns:
            import re
            raw_label = df.iloc[0]['material']
            raw_label = re.sub(r'_\d+$', '', raw_label)
            
            # 【核心映射：3硬 2软】
            if raw_label in ["Material_Wood", "Material_POM", "Material_Silicone"]:
                material_label = "Material_Hard"
            elif raw_label in ["Material_EPEfoam", "Material_Bottle"]:
                material_label = "Material_Soft"
            else:
                material_label = raw_label 
        
        for start in range(0, len(df) - self.window_size + 1, step):
            end = start + self.window_size
            window = df.iloc[start:end]
            features = self.extract_features_from_window(window)
            if material_label:
                features['material'] = material_label
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def extract_features_global(self, df):
        features = self.extract_features_from_window(df)
        if 'material' in df.columns:
            import re
            raw_label = df.iloc[0]['material']
            raw_label = re.sub(r'_\d+$', '', raw_label)
            
            # 【核心映射：3硬 2软】
            if raw_label in ["Material_Wood", "Material_POM", "Material_Silicone"]:
                material_label = "Material_Hard"
            elif raw_label in ["Material_EPEfoam", "Material_Bottle"]:
                material_label = "Material_Soft"
            else:
                material_label = raw_label 
                
            features['material'] = material_label
        return features
    
    def process_file(self, filename, method='global'):
        filepath = os.path.join(Config.PREPROCESS_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")
        df = pd.read_csv(filepath)
        if method == 'global':
            features = self.extract_features_global(df)
            features_df = pd.DataFrame([features])
        elif method == 'sliding':
            features_df = self.extract_features_sliding_window(df)
        else:
            raise ValueError(f"未知方法: {method}")
        return features_df
    
    def process_all(self, method='global', save=True):
        all_features = []
        for filename in os.listdir(Config.PREPROCESS_DIR):
            if not filename.endswith(".csv") or "preprocessed" not in filename:
                continue
            try:
                features_df = self.process_file(filename, method=method)
                all_features.append(features_df)
            except Exception:
                pass
        
        if all_features:
            combined_df = pd.concat(all_features, ignore_index=True)
            if save:
                features_dir = os.path.join(Config.BASE_DIR, "data_features")
                os.makedirs(features_dir, exist_ok=True)
                output_file = os.path.join(features_dir, f"features_{method}.csv")
                combined_df.to_csv(output_file, index=False)
            return combined_df
        return None

def main():
    extractor = FeatureExtractor(window_size=100)
    features_df = extractor.process_all(method='global', save=True)
    if features_df is not None:
        print("\n特征统计:")
        print(features_df.describe())
        print("\n材料分布:")
        if 'material' in features_df.columns:
            print(features_df['material'].value_counts())

if __name__ == "__main__":
    main()