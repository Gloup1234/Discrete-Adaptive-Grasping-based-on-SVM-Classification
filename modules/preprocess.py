"""
数据预处理模块
功能：
1. Fz阈值过滤
2. 异常值检测与处理
3. 数据平滑（可选）
4. 数据标准化准备
"""

import pandas as pd
import numpy as np
import os
from scipy import signal
import sys

# 添加父目录到路径以导入Config
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import Config


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, fz_contact=2.2, remove_outliers=True, smooth_data=False):
        """
        初始化预处理器
        
        Args:
            fz_contact: Fz接触阈值
            remove_outliers: 是否移除异常值
            smooth_data: 是否平滑数据
        """
        self.fz_contact = fz_contact
        self.remove_outliers = remove_outliers
        self.smooth_data = smooth_data
        
    def load_raw_data(self, filename):
        """加载原始数据"""
        filepath = os.path.join(Config.RAW_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")
        return pd.read_csv(filepath)
    
    def filter_contact(self, df):
        """过滤未接触数据"""
        return df[df["Fz"] > self.fz_contact].copy()
    
    def remove_outliers_iqr(self, df, columns=None):
        """
        使用IQR方法移除异常值
        
        Args:
            df: 数据框
            columns: 需要检测异常值的列，默认为所有数值列
        """
        if not self.remove_outliers:
            return df
            
        if columns is None:
            columns = ['Fx', 'Fy', 'Fz', 'Ft', 'mu']
        
        df_clean = df.copy()
        for col in columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR  # 使用3倍IQR
                upper_bound = Q3 + 3 * IQR
                
                # 标记异常值为NaN，然后插值
                mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                if mask.sum() > 0:
                    print(f"  - {col}: 检测到 {mask.sum()} 个异常值")
                    df_clean.loc[mask, col] = np.nan
                    df_clean[col] = df_clean[col].interpolate(method='linear')
        
        return df_clean
    
    def smooth_data_filter(self, df, window_size=5):
        """
        使用移动平均平滑数据
        
        Args:
            df: 数据框
            window_size: 窗口大小
        """
        if not self.smooth_data:
            return df
            
        df_smooth = df.copy()
        smooth_columns = ['Fx', 'Fy', 'Fz', 'Ft', 'mu']
        
        for col in smooth_columns:
            if col in df_smooth.columns:
                df_smooth[col] = df_smooth[col].rolling(
                    window=window_size, 
                    center=True, 
                    min_periods=1
                ).mean()
        
        return df_smooth
    
    def add_derived_features(self, df):
        """添加派生特征"""
        df = df.copy()
        
        # 确保Ft存在
        if 'Ft' not in df.columns:
            df['Ft'] = np.sqrt(df['Fx']**2 + df['Fy']**2)
        
        # 确保mu存在
        if 'mu' not in df.columns:
            df['mu'] = df['Ft'] / (df['Fz'] + 1e-6)
        
        # 添加Fz的时间导数（如果time列存在）
        if 'time' in df.columns and len(df) > 1:
            dt = df['time'].diff().fillna(df['time'].iloc[1] - df['time'].iloc[0])
            df['dFz_dt'] = df['Fz'].diff() / dt
            df['dFz_dt'] = df['dFz_dt'].fillna(0)
        
        return df
    
    def process_file(self, filename, save=True):
        """
        处理单个文件
        
        Args:
            filename: 文件名
            save: 是否保存处理后的数据
            
        Returns:
            处理后的DataFrame
        """
        print(f"正在处理: {filename}")
        
        # 1. 加载数据
        df = self.load_raw_data(filename)
        print(f"  原始数据: {len(df)} 行")
        
        # 2. 过滤未接触数据
        df = self.filter_contact(df)
        print(f"  接触过滤后: {len(df)} 行")
        
        # 3. 添加派生特征
        df = self.add_derived_features(df)
        
        # 4. 移除异常值
        df = self.remove_outliers_iqr(df)
        
        # 5. 平滑数据（可选）
        df = self.smooth_data_filter(df)
        
        # 6. 移除NaN行
        df = df.dropna()
        print(f"  最终数据: {len(df)} 行")
        
        # 7. 保存
        if save:
            # 处理各种命名格式: Material_XXX_raw.csv 或 Material_XXX_raw_N.csv
            output_filename = filename.replace("_raw", "_preprocessed")
            if not output_filename.endswith("_preprocessed.csv"):
                # 处理 Material_XXX_raw_N.csv -> Material_XXX_preprocessed_N.csv
                output_filename = output_filename.replace(".csv", "_preprocessed.csv") if "_preprocessed" not in output_filename else output_filename
            output_path = os.path.join(Config.PREPROCESS_DIR, output_filename)
            df.to_csv(output_path, index=False)
            print(f"  已保存到: {output_path}")
        
        return df
    
    def process_all(self):
        """处理所有原始数据文件"""
        os.makedirs(Config.PREPROCESS_DIR, exist_ok=True)
        
        processed_files = []
        for filename in os.listdir(Config.RAW_DIR):
            if not filename.endswith(".csv"):
                continue
            
            try:
                df = self.process_file(filename, save=True)
                processed_files.append((filename, df))
            except Exception as e:
                print(f"处理 {filename} 时出错: {str(e)}")
        
        print(f"\n完成！共处理 {len(processed_files)} 个文件")
        return processed_files


def main():
    """测试函数"""
    preprocessor = DataPreprocessor(
        fz_contact=2.2,
        remove_outliers=True,
        smooth_data=False
    )
    preprocessor.process_all()


if __name__ == "__main__":
    main()
