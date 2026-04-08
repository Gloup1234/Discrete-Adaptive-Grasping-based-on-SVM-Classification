"""
预测模块
用于实时或离线预测材料类型
"""

import pandas as pd
import numpy as np
import os
import sys

# 添加父目录到路径以导入Config
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import Config

from .train_classifier import MaterialClassifier
from .feature_extraction import FeatureExtractor


class MaterialPredictor:
    """材料预测器"""
    
    def __init__(self, model_path=None, classifier_type='svm'):
        """
        初始化预测器
        
        Args:
            model_path: 模型文件路径
            classifier_type: 分类器类型（如果不提供model_path）
        """
        self.classifier = MaterialClassifier(classifier_type=classifier_type)
        
        if model_path:
            self.classifier.load_model(model_path)
        else:
            # 尝试加载默认模型
            try:
                self.classifier.load_model()
            except FileNotFoundError:
                print("警告: 未找到训练好的模型，请先训练模型或指定模型路径")
        
        self.feature_extractor = FeatureExtractor(window_size=100)
    
    def predict_from_features(self, features_df):
        """
        从特征DataFrame预测
        
        Args:
            features_df: 包含特征的DataFrame
            
        Returns:
            预测结果
        """
        # 移除material列（如果存在）
        X = features_df.drop('material', axis=1, errors='ignore')
        
        # 确保特征顺序正确
        if self.classifier.feature_names:
            X = X[self.classifier.feature_names]
        
        predictions = self.classifier.predict(X)
        
        return predictions
    
    def predict_from_preprocessed_data(self, preprocessed_file):
        """
        从预处理后的数据文件预测
        
        Args:
            preprocessed_file: 预处理后的CSV文件路径
            
        Returns:
            预测结果
        """
        print(f"正在预测: {preprocessed_file}")
        
        # 加载预处理数据
        df = pd.read_csv(preprocessed_file)
        
        # 提取特征
        features = self.feature_extractor.extract_features_global(df)
        features_df = pd.DataFrame([features])
        
        # 移除material列
        if 'material' in features_df.columns:
            true_material = features_df['material'].iloc[0]
            features_df = features_df.drop('material', axis=1)
        else:
            true_material = None
        
        # 预测
        prediction = self.classifier.predict(features_df)[0]
        
        print(f"预测结果: {prediction}")
        if true_material:
            print(f"真实材料: {true_material}")
            print(f"预测{'正确' if prediction == true_material else '错误'}")
        
        return prediction
    
    def predict_from_raw_data(self, raw_file):
        """
        从原始数据文件预测（包含预处理步骤）
        
        Args:
            raw_file: 原始CSV文件路径
            
        Returns:
            预测结果
        """
        from .preprocess import DataPreprocessor
        
        print(f"正在处理并预测: {raw_file}")
        
        # 预处理
        preprocessor = DataPreprocessor(
            fz_contact=0.5,
            remove_outliers=True,
            smooth_data=False
        )
        
        # 加载原始数据
        df = pd.read_csv(raw_file)
        
        # 预处理（不保存）
        df_processed = preprocessor.filter_contact(df)
        df_processed = preprocessor.add_derived_features(df_processed)
        df_processed = preprocessor.remove_outliers_iqr(df_processed)
        df_processed = df_processed.dropna()
        
        # 提取特征
        features = self.feature_extractor.extract_features_global(df_processed)
        features_df = pd.DataFrame([features])
        
        # 移除material列
        if 'material' in features_df.columns:
            true_material = features_df['material'].iloc[0]
            features_df = features_df.drop('material', axis=1)
        else:
            true_material = None
        
        # 预测
        prediction = self.classifier.predict(features_df)[0]
        
        print(f"预测结果: {prediction}")
        if true_material:
            print(f"真实材料: {true_material}")
            print(f"预测{'正确' if prediction == true_material else '错误'}")
        
        return prediction
    
    def predict_batch(self, file_list, file_type='preprocessed'):
        """
        批量预测
        
        Args:
            file_list: 文件路径列表
            file_type: 'preprocessed' 或 'raw'
            
        Returns:
            预测结果列表
        """
        results = []
        
        for filepath in file_list:
            try:
                if file_type == 'preprocessed':
                    pred = self.predict_from_preprocessed_data(filepath)
                elif file_type == 'raw':
                    pred = self.predict_from_raw_data(filepath)
                else:
                    raise ValueError(f"未知文件类型: {file_type}")
                
                results.append({
                    'file': os.path.basename(filepath),
                    'prediction': pred
                })
            except Exception as e:
                print(f"预测 {filepath} 时出错: {str(e)}")
                results.append({
                    'file': os.path.basename(filepath),
                    'prediction': None,
                    'error': str(e)
                })
        
        return results


def main():
    """测试预测功能"""
    
    print("="*60)
    print("材料预测测试")
    print("="*60)
    
    # 创建预测器（使用SVM模型）
    try:
        predictor = MaterialPredictor(classifier_type='svm')
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        print("请先运行train_classifier.py训练模型")
        return
    
    # 测试1: 从预处理数据预测
    print("\n--- 测试1: 从预处理数据预测 ---")
    preprocess_dir = Config.PREPROCESS_DIR
    if os.path.exists(preprocess_dir):
        files = [os.path.join(preprocess_dir, f) 
                for f in os.listdir(preprocess_dir) 
                if f.endswith('_preprocessed.csv')]
        
        if files:
            print(f"找到 {len(files)} 个预处理文件")
            for filepath in files[:3]:  # 只测试前3个
                print(f"\n处理: {os.path.basename(filepath)}")
                try:
                    predictor.predict_from_preprocessed_data(filepath)
                except Exception as e:
                    print(f"错误: {str(e)}")
        else:
            print("未找到预处理文件")
    
    # 测试2: 批量预测
    print("\n--- 测试2: 批量预测所有预处理文件 ---")
    if os.path.exists(preprocess_dir):
        files = [os.path.join(preprocess_dir, f) 
                for f in os.listdir(preprocess_dir) 
                if f.endswith('_preprocessed.csv')]
        
        if files:
            results = predictor.predict_batch(files, file_type='preprocessed')
            
            print("\n批量预测结果:")
            for result in results:
                print(f"  {result['file']}: {result['prediction']}")
        else:
            print("未找到预处理文件")


if __name__ == "__main__":
    main()
