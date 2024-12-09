import pandas as pd
import numpy as np
import logging
from transformers import RobertaTokenizer
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data(data):
    """
    数据预处理函数，提取指定特征并生成新特征
    """
    # 提取已有的特征
    processed_data = data[['CosSim', 'ManSim', 'EucSim', 'EditSim']].copy()

    # 对每一列填充NaN值，使用该列的均值
    processed_data = processed_data.apply(lambda x: x.fillna(x.mean()), axis=0)

    # 计算 diffPair1 和 diffPair2 的 ChangeNo 差值的绝对值
    processed_data['diff_change_no'] = (data['diffPair1ChangeNo'] - data['diffPair2ChangeNo']).abs()

    # 生成 ground_truth 列，依据 FLAG 列设置分类标签
    processed_data['ground_truth'] = data['FLAG'].apply(lambda x: 1 if x in [1, 2] else 0)

    return processed_data

def main():
    """
    主程序：读取数据、预处理并保存结果
    """
    input_file = "../Data/20240917_MLTest_HLWZT_obgear-online-business_latest.xlsx"
    train_output = "preprocessed_train_data.csv"
    test_output = "preprocessed_test_data.csv"

    try:
        # 读取数据文件
        logging.info(f"读取数据文件：{input_file}")
        data = pd.read_excel(input_file)
    except Exception as e:
        logging.error(f"读取 Excel 文件失败：{e}")
        raise e

    # 筛选用于训练的数据 (IsTest 为空)
    logging.info("筛选用于训练的数据 (IsTest 为空)...")
    train_data = data[data['IsTest'].isna()].copy()

    # 筛选用于预测的数据 (IsTest 为 True)
    logging.info("筛选用于预测的数据 (IsTest 为 True)...")
    test_data = data[data['IsTest'] == True].copy()

    # 预处理训练数据
    logging.info("预处理训练数据...")
    train_processed = preprocess_data(train_data)

    # 预处理测试数据
    logging.info("预处理测试数据...")
    test_processed = preprocess_data(test_data)

    # 保存预处理后的数据
    try:
        train_processed.to_csv(train_output, index=False)
        test_processed.to_csv(test_output, index=False)
        logging.info(f"数据预处理完成，已保存为 '{train_output}' 和 '{test_output}'")
    except Exception as e:
        logging.error(f"保存预处理数据失败：{e}")
        raise e

if __name__ == "__main__":
    main()
