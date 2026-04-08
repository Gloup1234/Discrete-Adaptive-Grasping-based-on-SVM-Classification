"""
可视化模块
生成各种可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# 添加父目录到路径以导入Config
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import Config

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class DataVisualizer:
    """数据可视化器"""
    
    def __init__(self, figsize=(12, 8)):
        """
        初始化可视化器
        
        Args:
            figsize: 图表大小
        """
        self.figsize = figsize
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置风格
        sns.set_style("whitegrid")
        
    def load_features(self, features_file='features_global.csv'):
        """加载特征数据"""
        features_dir = os.path.join(Config.BASE_DIR, "data_features")
        filepath = os.path.join(features_dir, features_file)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"特征文件不存在: {filepath}")
        
        return pd.read_csv(filepath)
    
    def plot_feature_distributions(self, df, save_path=None):
        """
        绘制特征分布图
        
        Args:
            df: 特征DataFrame
            save_path: 保存路径
        """
        feature_cols = [col for col in df.columns if col != 'material']
        n_features = len(feature_cols)
        
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for i, col in enumerate(feature_cols):
            if 'material' in df.columns:
                for material in df['material'].unique():
                    data = df[df['material'] == material][col]
                    axes[i].hist(data, alpha=0.5, label=material, bins=20)
                axes[i].legend()
            else:
                axes[i].hist(df[col], bins=20)
            
            axes[i].set_title(f'{col} Distribution')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        
        # 隐藏多余的子图
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特征分布图已保存: {save_path}")
        
        plt.show()
    
    def plot_scatter_matrix(self, df, save_path=None):
        """
        绘制散点图矩阵（报告中的scatter plot）
        
        Args:
            df: 特征DataFrame
            save_path: 保存路径
        """
        # 选择主要特征
        feature_cols = ['k_eff', 'Fz_peak', 'mu_mean', 'mu_std', 'slip', 'micro']
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        if 'material' in df.columns:
            # 按材料类型着色
            materials = df['material'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(materials)))
            color_map = dict(zip(materials, colors))
            
            fig = plt.figure(figsize=(15, 15))
            
            n = len(feature_cols)
            for i in range(n):
                for j in range(n):
                    ax = plt.subplot(n, n, i*n + j + 1)
                    
                    if i == j:
                        # 对角线：直方图
                        for material in materials:
                            data = df[df['material'] == material][feature_cols[i]]
                            ax.hist(data, alpha=0.5, label=material, 
                                   color=color_map[material], bins=15)
                        if i == 0:
                            ax.legend(loc='upper right', fontsize=8)
                    else:
                        # 非对角线：散点图
                        for material in materials:
                            data = df[df['material'] == material]
                            ax.scatter(data[feature_cols[j]], data[feature_cols[i]], 
                                     alpha=0.6, s=30, color=color_map[material],
                                     label=material)
                    
                    if j == 0:
                        ax.set_ylabel(feature_cols[i], fontsize=10)
                    else:
                        ax.set_yticklabels([])
                    
                    if i == n-1:
                        ax.set_xlabel(feature_cols[j], fontsize=10)
                    else:
                        ax.set_xticklabels([])
            
            plt.suptitle('Feature Scatter Matrix by Material', fontsize=16, y=0.995)
        else:
            # 使用pandas绘制
            pd.plotting.scatter_matrix(df[feature_cols], figsize=self.figsize, 
                                     alpha=0.6, diagonal='hist')
            plt.suptitle('Feature Scatter Matrix', fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"散点图矩阵已保存: {save_path}")
        
        plt.show()
    
    def plot_pca_2d(self, df, save_path=None):
        """
        绘制PCA降维到2D的可视化
        
        Args:
            df: 特征DataFrame
            save_path: 保存路径
        """
        feature_cols = [col for col in df.columns if col != 'material']
        X = df[feature_cols].values
        
        # PCA降维
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        plt.figure(figsize=self.figsize)
        
        if 'material' in df.columns:
            materials = df['material'].unique()
            for material in materials:
                mask = df['material'] == material
                plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                           alpha=0.7, s=100, label=material)
            plt.legend()
        else:
            plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=100)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA Visualization (2D)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PCA可视化已保存: {save_path}")
        
        plt.show()
    
    def plot_tsne_2d(self, df, save_path=None):
        """
        绘制t-SNE降维到2D的可视化
        
        Args:
            df: 特征DataFrame
            save_path: 保存路径
        """
        feature_cols = [col for col in df.columns if col != 'material']
        X = df[feature_cols].values
        
        # t-SNE降维
        print("正在进行t-SNE降维...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
        X_tsne = tsne.fit_transform(X)
        
        plt.figure(figsize=self.figsize)
        
        if 'material' in df.columns:
            materials = df['material'].unique()
            for material in materials:
                mask = df['material'] == material
                plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                           alpha=0.7, s=100, label=material)
            plt.legend()
        else:
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7, s=100)
        
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('t-SNE Visualization (2D)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"t-SNE可视化已保存: {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(self, df, save_path=None):
        """
        绘制特征相关性矩阵
        
        Args:
            df: 特征DataFrame
            save_path: 保存路径
        """
        feature_cols = [col for col in df.columns if col != 'material']
        corr_matrix = df[feature_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"相关性矩阵已保存: {save_path}")
        
        plt.show()
    
    def plot_all(self, features_file='features_global.csv'):
        """
        生成所有可视化图表
        
        Args:
            features_file: 特征文件名
        """
        # 创建输出目录
        vis_dir = os.path.join(Config.BASE_DIR, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # 加载数据
        df = self.load_features(features_file)
        print(f"加载了 {len(df)} 个样本")
        
        # 生成各种图表
        print("\n生成特征分布图...")
        self.plot_feature_distributions(df, 
            save_path=os.path.join(vis_dir, "feature_distributions.png"))
        
        print("\n生成散点图矩阵...")
        self.plot_scatter_matrix(df, 
            save_path=os.path.join(vis_dir, "scatter_matrix.png"))
        
        print("\n生成相关性矩阵...")
        self.plot_correlation_matrix(df, 
            save_path=os.path.join(vis_dir, "correlation_matrix.png"))
        
        print("\n生成PCA可视化...")
        self.plot_pca_2d(df, 
            save_path=os.path.join(vis_dir, "pca_2d.png"))
        
        if len(df) > 1:  # t-SNE需要至少2个样本
            print("\n生成t-SNE可视化...")
            self.plot_tsne_2d(df, 
                save_path=os.path.join(vis_dir, "tsne_2d.png"))
        
        print(f"\n所有可视化图表已保存到: {vis_dir}")


def main():
    """测试函数"""
    visualizer = DataVisualizer()
    visualizer.plot_all('features_global.csv')


if __name__ == "__main__":
    main()
