import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 加载数据
train_filepath = "preprocessed_train_data.csv"
test_filepath = "preprocessed_test_data.csv"

train_data = pd.read_csv(train_filepath)
test_data = pd.read_csv(test_filepath)

# 拼接训练集和测试集
combined_data = pd.concat([train_data, test_data], axis=0)

# 假设 'ground_truth' 是目标变量，其他列是特征
features = combined_data.drop(columns=['ground_truth'])
ground_truth = combined_data['ground_truth']

# 处理缺失值：填充均值
features.fillna(features.mean(), inplace=True)

# 标准化特征列
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 创建一个新的DataFrame来存储标准化后的数据
scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

# 将 ground_truth 加入到标准化后的数据中
scaled_df['ground_truth'] = ground_truth.reset_index(drop=True)


# 标准化前后特征列的分布对比
def plot_feature_distributions(before, after, feature_name):
    plt.figure(figsize=(12, 6))

    # 获取标准化前后的横坐标范围
    x_min_before = before[feature_name].min()
    x_max_before = before[feature_name].max()
    x_min_after = after[feature_name].min()
    x_max_after = after[feature_name].max()

    # 使用统一的横坐标范围
    x_min = min(x_min_before, x_min_after)
    x_max = max(x_max_before, x_max_after)

    # 画标准化前的数据分布
    plt.subplot(1, 2, 1)
    plt.hist(before[feature_name], bins=30, color='skyblue', edgecolor='black')
    plt.title(f'{feature_name} - Before Standardization')
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.xlim(x_min, x_max)  # 保证横坐标一致

    # 画标准化后的数据分布
    plt.subplot(1, 2, 2)
    plt.hist(after[feature_name], bins=30, color='salmon', edgecolor='black')
    plt.title(f'{feature_name} - After Standardization')
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.xlim(x_min, x_max)  # 保证横坐标一致

    plt.tight_layout()
    plt.show()


# 展示每个特征列标准化前后的分布
for feature in features.columns:
    plot_feature_distributions(features, scaled_df, feature)

# Ground_truth分布对比
plt.figure(figsize=(12, 6))
x_min_before = ground_truth.min()
x_max_before = ground_truth.max()
x_min_after = scaled_df['ground_truth'].min()
x_max_after = scaled_df['ground_truth'].max()

# 使用统一的横坐标范围
x_min = min(x_min_before, x_min_after)
x_max = max(x_max_before, x_max_after)

plt.subplot(1, 2, 1)
plt.hist(ground_truth, bins=30, color='skyblue', edgecolor='black')
plt.title('Ground Truth - Before Standardization')
plt.xlabel('Ground Truth')
plt.ylabel('Frequency')
plt.xlim(x_min, x_max)  # 保证横坐标一致

plt.subplot(1, 2, 2)
plt.hist(scaled_df['ground_truth'], bins=30, color='salmon', edgecolor='black')
plt.title('Ground Truth - After Standardization')
plt.xlabel('Ground Truth')
plt.ylabel('Frequency')
plt.xlim(x_min, x_max)  # 保证横坐标一致

plt.tight_layout()
plt.show()
