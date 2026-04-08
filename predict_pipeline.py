"""
现场预测流水线
功能：
1. 从 prediction_raw 文件夹读取现场采集的原始数据
2. 使用预处理模块进行数据预处理，保存到 prediction_preprocessed 文件夹
3. 加载训练好的模型，对所有预处理后的数据进行预测
4. 输出预测结果

使用方法：
    python predict_pipeline.py
    python predict_pipeline.py --model svm       # 指定模型类型 (svm/rf/knn)
    python predict_pipeline.py --skip-preprocess  # 跳过预处理，直接预测
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np

# 添加modules路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

import Config
from modules.preprocess import DataPreprocessor
from modules.feature_extraction import FeatureExtractor
from modules.predict import MaterialPredictor


def print_separator(title=""):
    """打印分隔符"""
    print("\n" + "=" * 80)
    if title:
        print(f" {title} ".center(80, "="))
        print("=" * 80)
    print()


def step1_preprocess_prediction_data():
    """
    步骤1: 对 prediction_raw 中的原始数据进行预处理
    保存到 prediction_preprocessed 文件夹
    """
    print_separator("步骤 1/2: 预处理现场采集数据")

    raw_dir = Config.PREDICTION_RAW_DIR
    output_dir = Config.PREDICTION_PREPROCESS_DIR

    # 检查原始数据目录
    if not os.path.exists(raw_dir):
        print(f"错误: 原始数据目录不存在: {raw_dir}")
        print("请确保已将现场采集的数据放在 prediction_raw 文件夹中")
        return False

    # 获取所有CSV文件
    raw_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    if not raw_files:
        print(f"错误: 在 {raw_dir} 中未找到CSV文件")
        return False

    print(f"找到 {len(raw_files)} 个原始数据文件:")
    for f in raw_files:
        print(f"  - {f}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 初始化预处理器（与训练时使用相同的参数）
    preprocessor = DataPreprocessor(
        fz_contact=0.5,
        remove_outliers=False,
        smooth_data=False
    )

    print("\n预处理参数:")
    print("  - Fz接触阈值: 2.2N")
    print("  - 移除异常值: 是")
    print("  - 数据平滑: 否")

    processed_count = 0
    failed_count = 0

    for filename in raw_files:
        try:
            print(f"\n{'─' * 60}")
            print(f"正在处理: {filename}")

            # 1. 加载原始数据
            filepath = os.path.join(raw_dir, filename)
            df = pd.read_csv(filepath)
            print(f"  原始数据: {len(df)} 行, {len(df.columns)} 列")

            # 2. 只保留传感器核心列（去掉采集时可能带的多余特征列）
            core_columns = ['time', 'Fx', 'Fy', 'Fz']
            # 也保留可能存在的其他有用列
            optional_columns = ['Ft', 'mu', 'material']
            keep_columns = [c for c in core_columns if c in df.columns]
            keep_columns += [c for c in optional_columns if c in df.columns]
            df = df[keep_columns]

            # 3. 过滤未接触数据（Fz > 阈值）
            df = preprocessor.filter_contact(df)
            print(f"  接触过滤后: {len(df)} 行")

            if len(df) == 0:
                print(f"  警告: 过滤后无有效数据，跳过此文件")
                failed_count += 1
                continue

            # 4. 添加派生特征（Ft, mu, dFz_dt）
            df = preprocessor.add_derived_features(df)

            # 5. 移除异常值
            df = preprocessor.remove_outliers_iqr(df)

            # 6. 移除NaN行
            df = df.dropna()
            print(f"  最终数据: {len(df)} 行")

            if len(df) == 0:
                print(f"  警告: 处理后无有效数据，跳过此文件")
                failed_count += 1
                continue

            # 7. 保存预处理结果
            # 命名: Material_Predict_1.csv -> Material_Predict_1_preprocessed.csv
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}_preprocessed.csv"
            output_path = os.path.join(output_dir, output_filename)
            df.to_csv(output_path, index=False)
            print(f"  已保存到: {output_path}")

            processed_count += 1

        except Exception as e:
            print(f"  处理 {filename} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_count += 1

    print(f"\n预处理完成: 成功 {processed_count} 个, 失败 {failed_count} 个")
    return processed_count > 0


def step2_predict_all(classifier_type='svm'):
    """
    步骤2: 对 prediction_preprocessed 中的所有数据进行预测

    Args:
        classifier_type: 分类器类型 ('svm', 'rf', 'knn')
    """
    print_separator(f"步骤 2/2: 使用 {classifier_type.upper()} 模型进行预测")

    preprocess_dir = Config.PREDICTION_PREPROCESS_DIR

    # 检查预处理数据目录
    if not os.path.exists(preprocess_dir):
        print(f"错误: 预处理数据目录不存在: {preprocess_dir}")
        print("请先运行预处理步骤")
        return None

    # 获取所有预处理后的CSV文件
    preprocessed_files = [f for f in os.listdir(preprocess_dir) if f.endswith('.csv')]
    if not preprocessed_files:
        print(f"错误: 在 {preprocess_dir} 中未找到预处理文件")
        return None

    print(f"找到 {len(preprocessed_files)} 个预处理文件:")
    for f in preprocessed_files:
        print(f"  - {f}")

    # 加载预测器
    print(f"\n加载 {classifier_type.upper()} 模型...")
    try:
        predictor = MaterialPredictor(classifier_type=classifier_type)
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        print("请确保已运行 main_pipeline.py 训练模型")
        return None

    # 特征提取器
    feature_extractor = FeatureExtractor(window_size=100)

    # 逐个文件预测
    results = []
    print(f"\n{'─' * 80}")
    print(f"{'文件名':<45s} {'预测材料':<25s} {'置信度'}")
    print(f"{'─' * 80}")

    for filename in preprocessed_files:
        filepath = os.path.join(preprocess_dir, filename)

        try:
            # 加载预处理数据
            df = pd.read_csv(filepath)

            # 提取特征
            features = feature_extractor.extract_features_global(df)
            features_df = pd.DataFrame([features])

            # 移除非特征列
            if 'material' in features_df.columns:
                features_df = features_df.drop('material', axis=1)

            # 确保特征顺序与训练时一致
            if predictor.classifier.feature_names:
                features_df = features_df[predictor.classifier.feature_names]

            # 预测
            prediction = predictor.classifier.predict(features_df)[0]

            # 尝试获取置信度（概率）
            confidence = ""
            if hasattr(predictor.classifier.model, 'predict_proba'):
                try:
                    proba = predictor.classifier.model.predict_proba(
                        predictor.classifier.scaler.transform(features_df)
                    )
                    max_proba = np.max(proba)
                    confidence = f"{max_proba:.2%}"
                except Exception:
                    confidence = "N/A"

            results.append({
                'file': filename,
                'prediction': prediction,
                'confidence': confidence,
                'features': features
            })

            print(f"  {filename:<43s} {prediction:<23s} {confidence}")

        except Exception as e:
            print(f"  {filename:<43s} 预测失败: {str(e)}")
            results.append({
                'file': filename,
                'prediction': None,
                'confidence': None,
                'error': str(e)
            })

    # 打印汇总
    print_separator("预测结果汇总")

    successful = [r for r in results if r['prediction'] is not None]
    failed = [r for r in results if r['prediction'] is None]

    print(f"总文件数: {len(results)}")
    print(f"成功预测: {len(successful)}")
    print(f"预测失败: {len(failed)}")

    if successful:
        print(f"\n{'─' * 60}")
        print("预测分布:")
        predictions = [r['prediction'] for r in successful]
        for material in set(predictions):
            count = predictions.count(material)
            print(f"  {material}: {count} 个样本 ({count / len(predictions):.1%})")

        # 保存预测结果到CSV
        results_df = pd.DataFrame([{
            '文件名': r['file'],
            '预测材料': r['prediction'],
            '置信度': r['confidence']
        } for r in successful])

        results_path = os.path.join(Config.BASE_DIR, "prediction_results.csv")
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        print(f"\n预测结果已保存到: {results_path}")

    if failed:
        print(f"\n失败文件:")
        for r in failed:
            print(f"  {r['file']}: {r.get('error', '未知错误')}")

    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='现场采集数据预测流水线')
    parser.add_argument('--model', type=str, default='svm',
                        choices=['svm', 'rf', 'knn'],
                        help='选择分类器模型 (默认: svm)')
    parser.add_argument('--skip-preprocess', action='store_true',
                        help='跳过预处理步骤，直接使用已有的预处理数据进行预测')
    args = parser.parse_args()

    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " 现场数据预测流水线 ".center(70) + "║")
    print("║" + " Field Data Prediction Pipeline ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    print(f"配置:")
    print(f"  - 原始数据目录:   {Config.PREDICTION_RAW_DIR}")
    print(f"  - 预处理输出目录: {Config.PREDICTION_PREPROCESS_DIR}")
    print(f"  - 使用模型:       {args.model.upper()}")
    print(f"  - 跳过预处理:     {'是' if args.skip_preprocess else '否'}")

    start_time = time.time()

    try:
        # 步骤1: 预处理
        if not args.skip_preprocess:
            success = step1_preprocess_prediction_data()
            if not success:
                print("\n预处理失败，流程终止")
                return
        else:
            print("\n跳过预处理步骤，直接使用已有的预处理数据")

        # 步骤2: 预测
        results = step2_predict_all(classifier_type=args.model)

        # 完成
        elapsed_time = time.time() - start_time
        print_separator("流程完成")
        print(f"总耗时: {elapsed_time:.2f} 秒")

        if results:
            successful = [r for r in results if r['prediction'] is not None]
            if successful:
                print(f"\n成功预测 {len(successful)} 个样本")
                print("\n各文件预测结果:")
                for r in successful:
                    conf_str = f" (置信度: {r['confidence']})" if r['confidence'] else ""
                    print(f"  {r['file']}  -->  {r['prediction']}{conf_str}")

    except Exception as e:
        print(f"\n流程执行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()