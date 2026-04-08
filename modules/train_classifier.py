"""
分类器训练模块
支持多种分类器：SVM, Random Forest, KNN等
"""

import pandas as pd
import numpy as np
import os
import pickle
import sys

# 添加父目录到路径以导入Config
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import Config

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class MaterialClassifier:
    """材料分类器"""
    
    def __init__(self, classifier_type='svm'):
        """
        初始化分类器
        
        Args:
            classifier_type: 'svm', 'rf' (random forest), 或 'knn'
        """
        self.classifier_type = classifier_type
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_names = None
        self.class_names = None
        
    def _create_classifier(self):
        """创建分类器"""
        if self.classifier_type == 'svm':
            return SVC(kernel='rbf', random_state=42, probability=True)
        elif self.classifier_type == 'rf':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.classifier_type == 'knn':
            return KNeighborsClassifier(n_neighbors=5)
        else:
            raise ValueError(f"未知分类器类型: {self.classifier_type}")
    
    def load_features(self, features_file='features_global.csv'):
        """
        加载特征数据
        
        Args:
            features_file: 特征文件名
            
        Returns:
            X, y, feature_names
        """
        features_dir = os.path.join(Config.BASE_DIR, "data_features")
        filepath = os.path.join(features_dir, features_file)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"特征文件不存在: {filepath}")
        
        df = pd.read_csv(filepath)
        
        # 分离特征和标签
        if 'material' not in df.columns:
            raise ValueError("特征文件中缺少'material'列")
        
        y = df['material']
        X = df.drop('material', axis=1)
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def prepare_data(self, X, y, test_size=0.2):
        """
        准备训练数据
        
        Args:
            X: 特征
            y: 标签
            test_size: 测试集比例
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # 编码标签
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_
        
        # 分割数据 - 自动判断是否使用stratify
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )
        except ValueError:
            # 样本太少，无法分层采样
            print("  警告: 样本数较少，使用随机分割（不保持类别比例）")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42
            )
        
        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train, y_train, use_grid_search=True):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            use_grid_search: 是否使用网格搜索优化参数
        """
        print(f"正在训练 {self.classifier_type.upper()} 分类器...")
        
        # 自动调整交叉验证折数
        n_splits = min(5, len(X_train))
        
        if use_grid_search:
            print("使用网格搜索优化参数...")
            
            if self.classifier_type == 'svm':
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    'kernel': ['rbf', 'poly']
                }
            elif self.classifier_type == 'rf':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
            elif self.classifier_type == 'knn':
                param_grid = {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            
            base_model = self._create_classifier()
            grid_search = GridSearchCV(
                base_model, param_grid, cv=n_splits, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            print(f"最佳参数: {grid_search.best_params_}")
            print(f"交叉验证最佳得分: {grid_search.best_score_:.4f}")
        else:
            self.model = self._create_classifier()
            self.model.fit(X_train, y_train)
        
        # 交叉验证
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=n_splits)
        print(f"{n_splits}折交叉验证得分: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    def evaluate(self, X_test, y_test):
        """
        评估模型
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
            
        Returns:
            准确率
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n" + "="*50)
        print("模型评估结果")
        print("="*50)
        print(f"测试集准确率: {accuracy:.4f}")
        print("\n分类报告:")
        print(classification_report(
            y_test, y_pred, 
            target_names=self.class_names,
            digits=4
        ))
        
        print("\n混淆矩阵:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        return accuracy
    
    def save_model(self, filename=None):
        """
        保存模型
        
        Args:
            filename: 模型文件名
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        models_dir = os.path.join(Config.BASE_DIR, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        if filename is None:
            filename = f"classifier_{self.classifier_type}.pkl"
        
        filepath = os.path.join(models_dir, filename)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'classifier_type': self.classifier_type
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n模型已保存到: {filepath}")
    
    def load_model(self, filename=None):
        """
        加载模型
        
        Args:
            filename: 模型文件名
        """
        models_dir = os.path.join(Config.BASE_DIR, "models")
        
        if filename is None:
            filename = f"classifier_{self.classifier_type}.pkl"
        
        filepath = os.path.join(models_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.class_names = model_data['class_names']
        self.classifier_type = model_data['classifier_type']
        
        print(f"模型已加载: {filepath}")
    
    def predict(self, X):
        """
        预测
        
        Args:
            X: 特征（DataFrame或numpy array）
            
        Returns:
            预测的材料标签
        """
        if self.model is None:
            raise ValueError("模型尚未训练或加载")
        
        X_scaled = self.scaler.transform(X)
        y_pred_encoded = self.model.predict(X_scaled)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred


def main():
    """训练和评估分类器"""
    
    # 测试不同的分类器
    classifier_types = ['svm', 'rf', 'knn']
    results = {}
    
    for clf_type in classifier_types:
        print(f"\n{'='*60}")
        print(f"训练 {clf_type.upper()} 分类器")
        print(f"{'='*60}")
        
        try:
            # 创建分类器
            classifier = MaterialClassifier(classifier_type=clf_type)
            
            # 加载数据
            X, y = classifier.load_features('features_global.csv')
            print(f"加载了 {len(X)} 个样本，{len(X.columns)} 个特征")
            print(f"材料类别: {y.unique()}")
            
            # 准备数据
            X_train, X_test, y_train, y_test = classifier.prepare_data(X, y, test_size=0.2)
            print(f"训练集: {len(X_train)} 样本")
            print(f"测试集: {len(X_test)} 样本")
            
            # 训练
            classifier.train(X_train, y_train, use_grid_search=True)
            
            # 评估
            accuracy = classifier.evaluate(X_test, y_test)
            results[clf_type] = accuracy
            
            # 保存模型
            classifier.save_model()
            
        except Exception as e:
            print(f"训练 {clf_type} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 总结
    print("\n" + "="*60)
    print("所有分类器结果总结")
    print("="*60)
    for clf_type, acc in results.items():
        print(f"{clf_type.upper()}: {acc:.4f}")
    
    if results:
        best_clf = max(results, key=results.get)
        print(f"\n最佳分类器: {best_clf.upper()} (准确率: {results[best_clf]:.4f})")


if __name__ == "__main__":
    main()
