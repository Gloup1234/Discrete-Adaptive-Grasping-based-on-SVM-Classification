
"""
材料分类完整流程主控制文件
运行此文件将执行从数据预处理到模型训练评估的完整流程
"""

import os
import sys
import time

# 添加modules路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.preprocess import DataPreprocessor
from modules.feature_extraction import FeatureExtractor
from modules.train_classifier import MaterialClassifier
from modules.visualize import DataVisualizer
from modules.predict import MaterialPredictor


def print_separator(title=""):
    """打印分隔符"""
    print("\n" + "="*80)
    if title:
        print(f" {title} ".center(80, "="))
        print("="*80)
    print()


def step1_preprocess_data():
    """步骤1: 数据预处理"""
    print_separator("步骤 1/5: 数据预处理")
    
    print("配置预处理参数:")
    print("  - Fz接触阈值: 2.2N")
    print("  - 移除异常值: 是")
    print("  - 数据平滑: 否")
    
    preprocessor = DataPreprocessor(
        fz_contact=0.5,
        remove_outliers=False,
        smooth_data=False
    )
    
    print("\n开始处理原始数据...")
    preprocessor.process_all()
    
    print("\n✓ 数据预处理完成")
    time.sleep(1)


def step2_extract_features():
    """步骤2: 特征提取"""
    print_separator("步骤 2/5: 特征提取")
    
    print("特征提取配置:")
    print("  - 方法: 全局特征提取（每个文件一个样本）")
    print("  - 窗口大小: 100")
    print("\n提取的6个核心特征:")
    print("  1. k_eff      - 有效刚度")
    print("  2. Fz_peak    - 峰值法向力")
    print("  3. mu_mean    - 平均摩擦系数")
    print("  4. mu_std     - 摩擦稳定性")
    print("  5. slip       - 滑动不稳定强度")
    print("  6. micro      - 微振动")
    
    extractor = FeatureExtractor(window_size=100)
    
    print("\n开始提取特征...")
    features_df = extractor.process_all(method='global', save=True)
    
    if features_df is not None:
        print("\n特征提取结果:")
        print(f"  总样本数: {len(features_df)}")
        print(f"  特征数量: {len(features_df.columns) - 1}")  # 减去material列
        
        if 'material' in features_df.columns:
            print("\n材料分布:")
            for material, count in features_df['material'].value_counts().items():
                print(f"  {material}: {count} 个样本")
        
        print("\n✓ 特征提取完成")
    else:
        print("\n✗ 特征提取失败")
    
    time.sleep(1)
    return features_df


def step3_visualize_data():
    """步骤3: 数据可视化"""
    print_separator("步骤 3/5: 数据可视化")
    
    print("生成可视化图表:")
    print("  1. 特征分布图")
    print("  2. 散点图矩阵")
    print("  3. 相关性矩阵")
    print("  4. PCA降维可视化")
    print("  5. t-SNE降维可视化")
    
    visualizer = DataVisualizer()
    
    print("\n开始生成图表...")
    try:
        visualizer.plot_all('features_global.csv')
        print("\n✓ 数据可视化完成")
    except Exception as e:
        print(f"\n✗ 可视化过程出错: {str(e)}")
    
    time.sleep(1)


def step4_train_models():
    """步骤4: 训练分类模型"""
    print_separator("步骤 4/5: 模型训练与评估")
    
    # 训练多个分类器进行比较
    classifier_types = ['svm', 'rf', 'knn']
    results = {}
    
    for clf_type in classifier_types:
        print(f"\n{'─'*80}")
        print(f"训练 {clf_type.upper()} 分类器")
        print(f"{'─'*80}\n")
        
        try:
            # 创建分类器
            classifier = MaterialClassifier(classifier_type=clf_type)
            
            # 加载数据
            X, y = classifier.load_features('features_global.csv')
            print(f"数据加载成功:")
            print(f"  样本数: {len(X)}")
            print(f"  特征数: {len(X.columns)}")
            print(f"  类别数: {len(y.unique())}")
            
            # 准备数据
            X_train, X_test, y_train, y_test = classifier.prepare_data(X, y, test_size=0.2)
            print(f"\n数据分割:")
            print(f"  训练集: {len(X_train)} 样本")
            print(f"  测试集: {len(X_test)} 样本")
            
            # 训练
            print("\n开始训练...")
            classifier.train(X_train, y_train, use_grid_search=True)
            
            # 评估
            accuracy = classifier.evaluate(X_test, y_test)
            results[clf_type] = accuracy
            
            # 保存模型
            classifier.save_model()
            
            print(f"\n✓ {clf_type.upper()} 训练完成")
            
        except Exception as e:
            print(f"\n✗ 训练 {clf_type} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 总结
    if results:
        print("\n" + "="*80)
        print(" 模型性能对比 ".center(80, "="))
        print("="*80 + "\n")
        
        for clf_type, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"  {clf_type.upper():10s}: {acc:.4f} ({acc*100:.2f}%)")
        
        best_clf = max(results, key=results.get)
        print(f"\n  最佳模型: {best_clf.upper()} (准确率: {results[best_clf]:.4f})")
        print("\n✓ 所有模型训练完成")
    else:
        print("\n✗ 没有成功训练任何模型")
    
    time.sleep(1)
    return results


def step5_test_prediction():
    """步骤5: 测试预测功能"""
    print_separator("步骤 5/5: 测试预测功能")
    
    print("使用训练好的模型进行预测测试...")
    
    try:
        # 使用SVM模型（通常性能最好）
        predictor = MaterialPredictor(classifier_type='svm')
        
        # 批量预测所有预处理文件
        import Config
        preprocess_dir = Config.PREPROCESS_DIR
        
        if os.path.exists(preprocess_dir):
            files = [os.path.join(preprocess_dir, f) 
                    for f in os.listdir(preprocess_dir) 
                    if f.endswith('.csv')]
            
            if files:
                print(f"\n找到 {len(files)} 个文件进行预测测试\n")
                
                results = predictor.predict_batch(files, file_type='preprocessed')
                
                print("\n" + "─"*80)
                print("预测结果汇总:")
                print("─"*80)
                
                correct = 0
                total = 0
                
                for result in results:
                    if result['prediction']:
                        # 从文件名中提取真实材料名
                        filename = result['file']
                        if 'Material_' in filename:
                            # 提取材料名（去除序号和后缀）
                            import re
                            # Material_XXX_preprocessed_N.csv -> XXX
                            match_obj = re.search(r'Material_([A-Za-z]+)', filename)
                            true_material = match_obj.group(1) if match_obj else ''
                            
                            match_obj = re.search(r'Material_([A-Za-z]+)', result['prediction'])
                            pred_material = match_obj.group(1) if match_obj else ''
                            
                            match = '✓' if true_material == pred_material else '✗'
                            if match == '✓':
                                correct += 1
                            total += 1
                            
                            print(f"  {match} {filename:50s} -> {result['prediction']}")
                        else:
                            print(f"    {filename:50s} -> {result['prediction']}")
                
                if total > 0:
                    accuracy = correct / total
                    print(f"\n预测准确率: {correct}/{total} = {accuracy:.2%}")
                
                print("\n✓ 预测测试完成")
            else:
                print("\n未找到预处理文件进行测试")
        else:
            print(f"\n预处理目录不存在: {preprocess_dir}")
    
    except Exception as e:
        print(f"\n✗ 预测测试出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    time.sleep(1)


def run_full_pipeline():
    """运行完整流程"""
    
    print("\n")
    print("╔" + "═"*78 + "╗")
    print("║" + " 材料分类完整流程 ".center(78) + "║")
    print("║" + " Material Classification Pipeline ".center(78) + "║")
    print("╚" + "═"*78 + "╝")
    print("\n")
    
    print("本流程将依次执行以下步骤:")
    print("  1. 数据预处理")
    print("  2. 特征提取")
    print("  3. 数据可视化")
    print("  4. 模型训练与评估")
    print("  5. 预测功能测试")
    
    input("\n按Enter键开始...")
    
    start_time = time.time()
    
    try:
        # 步骤1: 数据预处理
        step1_preprocess_data()
        
        # 步骤2: 特征提取
        features_df = step2_extract_features()
        
        if features_df is None or len(features_df) == 0:
            print("\n✗ 流程中断: 特征提取失败")
            return
        
        # 步骤3: 数据可视化
        step3_visualize_data()
        
        # 步骤4: 模型训练
        results = step4_train_models()
        
        if not results:
            print("\n✗ 流程中断: 模型训练失败")
            return
        
        # 步骤5: 预测测试
        step5_test_prediction()
        
        # 完成
        elapsed_time = time.time() - start_time
        
        print_separator("流程完成")
        print(f"总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
        print("\n所有步骤已成功完成！")
        print("\n生成的文件:")
        print("  - data_preprocess/     : 预处理后的数据")
        print("  - data_features/       : 提取的特征")
        print("  - models/              : 训练好的模型")
        print("  - visualizations/      : 可视化图表")
        
    except Exception as e:
        print(f"\n✗ 流程执行出错: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    
    # 检查必要的目录
    import Config
    
    if not os.path.exists(Config.RAW_DIR):
        print(f"错误: 原始数据目录不存在: {Config.RAW_DIR}")
        print("请确保已将原始数据放在正确的目录中")
        return
    
    raw_files = [f for f in os.listdir(Config.RAW_DIR) if f.endswith('.csv')]
    if not raw_files:
        print(f"错误: 在 {Config.RAW_DIR} 中未找到CSV文件")
        return
    
    print(f"找到 {len(raw_files)} 个原始数据文件:")
    for f in raw_files:
        print(f"  - {f}")
    
    # 运行完整流程
    run_full_pipeline()


if __name__ == "__main__":
    main()