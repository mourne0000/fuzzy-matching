import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

df_disease = pd.read_csv("disease.csv")
df_no_disease = pd.read_csv("no_disease.csv")

def preprocess_data(df, features):
    df = df.copy()
    df['gender'] = df['gender'].map({'female': 0, 'male': 1})
    return df[features + ['ID', 'date']]

# standardization
def standardize_features(disease_df, no_disease_df, features):
    combined = pd.concat([disease_df[features], no_disease_df[features]])
    mean = combined.mean()
    std = combined.std()

    disease_features = (disease_df[features] - mean) / std
    no_disease_features = (no_disease_df[features] - mean) / std
    return disease_features, no_disease_features
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from itertools import permutations
from tqdm import tqdm

def global_optimal_match(disease_data, no_disease_data, 
                        disease_features, no_disease_features,
                        VI):
    """全局暴力搜索最优匹配（总距离最小）"""
    # 确保两组数据数量相同
    assert len(disease_data) == len(no_disease_data), "数据集必须等长"
    n = len(disease_data)
    
    # 计算全量距离矩阵
    distance_matrix = cdist(
        disease_features.values, 
        no_disease_features.values, 
        metric='mahalanobis', 
        VI=VI
    )
    
    # 初始化最小总距离和最优排列
    min_total = float('inf')
    best_perm = None
    
    # 遍历所有可能的排列组合
    total_perms = np.math.factorial(n)
    with tqdm(permutations(range(n)), total=total_perms, desc="全局搜索进度") as pbar:
        for perm in pbar:
            # 计算当前排列的总距离
            total = sum(distance_matrix[i, perm[i]] for i in range(n))
            
            # 更新最优解
            if total < min_total:
                min_total = total
                best_perm = perm
            pbar.update(1)
    
    # 构建结果
    matches = []
    for i in range(n):
        matches.append({
            'Disease ID': disease_data.iloc[i]['ID'],
            'Disease Date': disease_data.iloc[i]['date'],
            'no Disease ID': no_disease_data.iloc[best_perm[i]]['ID'],
            'no Disease Date': no_disease_data.iloc[best_perm[i]]['date'],
            'Distance': distance_matrix[i, best_perm[i]]
        })
    return pd.DataFrame(matches)

def compute_global_cov_inv(disease_features, no_disease_features, weights=None):
    """计算全局协方差逆矩阵"""
    combined = pd.concat([disease_features, no_disease_features])
    if weights is not None:
        combined = combined * weights
    cov_matrix = np.cov(combined.T)
    cov_reg = cov_matrix + 1e-6 * np.eye(cov_matrix.shape[0])  # 正则化
    return np.linalg.inv(cov_reg)

def main():
    # 数据预处理
    features = ['age', 'gender', 'blood_pressure', 'height', 'weight', 'BMI']
    weights = np.array([0.2, 0.1, 0.3, 0.1, 0.1, 0.2])
    output_path = "global_brute_force_match.csv"

    # 加载数据并截断至相同长度
    X_disease = preprocess_data(df_disease, features)
    X_no_disease = preprocess_data(df_no_disease, features)
    min_len = min(len(X_disease), len(X_no_disease))
    X_disease = X_disease.iloc[:min_len]
    X_no_disease = X_no_disease.iloc[:min_len]

    # 标准化特征
    disease_features, no_disease_features = standardize_features(
        X_disease, X_no_disease, features
    )

    # 计算全局协方差逆矩阵
    VI = compute_global_cov_inv(disease_features, no_disease_features, weights)

    # 执行全局暴力匹配
    matches_df = global_optimal_match(
        disease_data=X_disease,
        no_disease_data=X_no_disease,
        disease_features=disease_features,
        no_disease_features=no_disease_features,
        VI=VI
    )

    # 保存结果
    matches_df.to_csv(output_path, index=False)
    print("总距离最小值:", matches_df['Distance'].sum())
    print("匹配结果样例:\n", matches_df.head())

if __name__ == "__main__":
    main()