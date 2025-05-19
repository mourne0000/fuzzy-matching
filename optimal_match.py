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

# compute the weighted covariance matrix
def compute_weighted_covariance(disease_features, no_disease_features, weights):

    combined = pd.concat([disease_features, no_disease_features])
    cov_matrix = combined.cov().values

    W_matrix = np.outer(weights, weights)
    cov_weighted = cov_matrix / W_matrix

    try:
        cov_inv = np.linalg.inv(cov_weighted)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_weighted)
    return cov_inv

def optimal_matches(disease_data, no_disease_data, 
                disease_features, no_disease_features, 
                cov_inv):

    n = len(disease_data)
    m = len(no_disease_data)

# initiallize the matrix
    cost_matrix = np.full((n, m), np.inf)

    for i in range(n):
        disease_id = disease_data.iloc[i]['ID']
        mask = (no_disease_data['ID'] != disease_id)
        valid_j = np.where(mask)[0]
        
        if valid_j.size == 0:
            continue

        distances = cdist(
            disease_features.iloc[i].values.reshape(1, -1),
            no_disease_features.iloc[valid_j],
            metric='mahalanobis',
            VI=cov_inv
        )
        cost_matrix[i, valid_j] = distances.ravel()
# applied hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for i, j in zip(row_ind, col_ind):
        cost = cost_matrix[i, j]
        if np.isinf(cost):
            continue
        
        disease_id = disease_data.iloc[i]['ID']
        disease_date = disease_data.iloc[i]['date']
        no_disease_id = no_disease_data.iloc[j]['ID']
        no_disease_date = no_disease_data.iloc[j]['date']
        
        matches.append({
            'Disease ID': disease_id,
            'Disease Date': disease_date,
            'no Disease ID': no_disease_id,
            'no Disease Date': no_disease_date,
            'Distance': cost
        })
    
    return pd.DataFrame(matches)

def main():
    features = ['age', 'gender', 'blood_pressure', 'height', 'weight', 'BMI']
    weights = np.array([0.2, 0.1, 0.3, 0.1, 0.1, 0.2])
    output_path = "optimal_match.csv"

    X_disease = preprocess_data(df_disease, features)
    X_no_disease = preprocess_data(df_no_disease, features)

    disease_features, no_disease_features = standardize_features(
        X_disease, X_no_disease, features
    )

    cov_inv = compute_weighted_covariance(
        disease_features, no_disease_features, weights
    )

    matches_df = optimal_matches(
        disease_data=X_disease,
        no_disease_data=X_no_disease,
        disease_features=disease_features,
        no_disease_features=no_disease_features,
        cov_inv=cov_inv,
    )

    matches_df.to_csv(output_path, index=False)
    print(matches_df.head(20))
    
    total_distance = matches_df['Distance'].sum()
    print(f"Total distance: {total_distance:.4f}")

if __name__ == "__main__":
    main()