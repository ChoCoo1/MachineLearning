import pandas as pd
import numpy as np
import itertools
import logging
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, accuracy_score, matthews_corrcoef
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 默认模型配置，不包含调优部分
MODELS = {
    'MLP': MLPClassifier(random_state=42, max_iter=500),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(probability=True, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(verbose=0,random_state=42)
}

METRICS = [
    'AUC-ROC',
    'AUC-PR',
    'Precision',
    'Recall',
    'F1',
    'Accuracy',
    'MCC'
]


def load_data(train_filepath, test_filepath):
    try:
        train_data = pd.read_csv(train_filepath)
        logging.info(f"成功加载训练数据：{train_filepath}")
    except Exception as e:
        logging.error(f"加载训练数据失败：{e}")
        raise e

    try:
        test_data = pd.read_csv(test_filepath)
        logging.info(f"成功加载测试数据：{test_filepath}")
    except Exception as e:
        logging.error(f"加载测试数据失败：{e}")
        raise e

    return train_data, test_data


def generate_feature_combinations(features):
    combinations = []
    for i in range(1, len(features) + 1):
        combinations.extend(itertools.combinations(features, i))
    return combinations


def create_pipeline(model, needs_scaling=True):
    steps = []
    steps.append(('imputer', SimpleImputer(strategy='mean')))
    if needs_scaling:
        steps.append(('scaler', StandardScaler()))
    steps.append(('model', model))
    return Pipeline(steps)


def evaluate_model(y_true, y_pred, y_proba):
    metrics = {}
    try:
        metrics['AUC-ROC'] = roc_auc_score(y_true, y_proba)
    except:
        metrics['AUC-ROC'] = np.nan
    try:
        metrics['AUC-PR'] = average_precision_score(y_true, y_proba)
    except:
        metrics['AUC-PR'] = np.nan
    metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['F1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    return metrics


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    results = []
    for model_name, model in MODELS.items():
        try:
            # needs_scaling = model_name in ['MLP', 'Logistic Regression', 'SVM']
            pipeline = create_pipeline(model)

            # 直接训练模型
            pipeline.fit(X_train, y_train)

            # 预测
            y_pred = pipeline.predict(X_test)
            if hasattr(pipeline.named_steps['model'], "predict_proba"):
                y_proba = pipeline.predict_proba(X_test)[:, 1]
            else:
                y_proba = pipeline.decision_function(X_test)
                y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())

            metrics = evaluate_model(y_test, y_pred, y_proba)
            result = {
                'Model': model_name
            }
            result.update(metrics)
            results.append(result)

            logging.info(
                f"{model_name} 模型评估完成。AUC-ROC: {metrics['AUC-ROC']:.4f}, AUC-PR: {metrics['AUC-PR']:.4f}, Precision: {metrics['Precision']:.4f}, Recall: {metrics['Recall']:.4f}, F1: {metrics['F1']:.4f}, Accuracy: {metrics['Accuracy']:.6f}, MCC: {metrics['MCC']:.4f}")

        except Exception as e:
            logging.error(f"在模型 {model_name} 训练过程中出错：{e}")
            result = {
                'Model': model_name,
                'AUC-ROC': np.nan,
                'AUC-PR': np.nan,
                'Precision': np.nan,
                'Recall': np.nan,
                'F1': np.nan,
                'Accuracy': np.nan,
                'MCC': np.nan
            }
            results.append(result)

    return results


def main():
    train_filepath = "preprocessed_train_data.csv"
    test_filepath = "preprocessed_test_data.csv"
    output_filepath = "model_evaluation_results.xlsx"

    train_data, test_data = load_data(train_filepath, test_filepath)

    feature_columns = ['CosSim', 'ManSim', 'EucSim', 'EditSim', 'diff_change_no']  # 最多5个特征

    logging.info("生成所有特征组合...")
    feature_combinations = generate_feature_combinations(feature_columns)
    logging.info(f"总共有 {len(feature_combinations)} 个特征组合。")

    results = []

    logging.info("开始遍历特征组合并训练模型...")
    for combo in tqdm(feature_combinations, desc="特征组合"):
        combo = list(combo)

        try:
            X_train = train_data[combo].values
            y_train = train_data['ground_truth'].values
            X_test = test_data[combo].values
            y_test = test_data['ground_truth'].values

            combo_results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

            for res in combo_results:
                res['Feature_Combination'] = ', '.join(combo)
                results.append(res)

        except Exception as e:
            logging.error(f"在处理特征组合 {combo} 时出错：{e}")
            for model_name in MODELS.keys():
                result = {
                    'Feature_Combination': ', '.join(combo),
                    'Model': model_name,
                    'AUC-ROC': np.nan,
                    'AUC-PR': np.nan,
                    'Precision': np.nan,
                    'Recall': np.nan,
                    'F1': np.nan,
                    'Accuracy': np.nan,
                    'MCC': np.nan
                }
                results.append(result)

    # 将 Feature_Combination 放到第一列并保存结果
    results_df = pd.DataFrame(results)
    # 检查 'Feature_Combination' 是否确实在列中
    if 'Feature_Combination' in results_df.columns:
        results_df = results_df[['Feature_Combination'] + [col for col in results_df.columns if col != 'Feature_Combination']]
    else:
        print("Error: 'Feature_Combination' not found in columns.")
    results_df.to_excel(output_filepath, index=False)
    logging.info(f"结果已保存到 {output_filepath}")


if __name__ == '__main__':
    main()
