import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from itertools import permutations
from tqdm import tqdm
import math
import sys

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
# def compute_weighted_covariance(disease_features, no_disease_features, weights):

#     combined = pd.concat([disease_features, no_disease_features])
#     cov_matrix = combined.cov().values

#     W_matrix = np.outer(weights, weights)
#     cov_weighted = cov_matrix / W_matrix

#     try:
#         cov_inv = np.linalg.inv(cov_weighted)
#     except np.linalg.LinAlgError:
#         cov_inv = np.linalg.pinv(cov_weighted)
#     return cov_inv

def exact_matches_pdist(disease_data, no_disease_data, 
                       disease_features, no_disease_features,
                       enable_deduplication=True):
    matches = []
    used_indices = set() if enable_deduplication else None
    n_features = disease_features.shape[1]

    for i in range(len(disease_data)):
        disease_id = disease_data.iloc[i]['ID']
        disease_date = disease_data.iloc[i]['date']
        current_feature = disease_features.iloc[i].values.reshape(1, -1)

        mask = (no_disease_data['ID'] != disease_id)
        if enable_deduplication:
            mask &= (~no_disease_data.index.isin(used_indices))
        candidates = no_disease_data[mask]
        if candidates.empty:
            continue

        combined = np.vstack([current_feature, no_disease_features[mask].values])

        # calculate the distance
        # calculate the covatiance
        VI = np.linalg.inv(np.cov(combined.T))
        dists = pdist(combined, 'mahalanobis', VI=VI)
        dist_matrix = squareform(dists)[0, 1:]

        # find the nearest one for each index
        min_idx = np.argmin(dist_matrix)
        best_match_idx = candidates.index[min_idx]

        matches.append({
            'Disease ID': disease_id,
            'Disease Date': disease_date,
            'no Disease ID': no_disease_data.loc[best_match_idx, 'ID'],
            'no Disease Date': no_disease_data.loc[best_match_idx, 'date'],
            'Distance': dist_matrix[min_idx]
        })

        if enable_deduplication:
            used_indices.add(best_match_idx)

    return pd.DataFrame(matches)

def main():

    features = ['age', 'gender', 'blood_pressure', 'height', 'weight', 'BMI']
    weights = np.array([0.2, 0.1, 0.3, 0.1, 0.1, 0.2])
    output_path = "pdist_match.csv"

    X_disease = preprocess_data(df_disease, features)
    X_no_disease = preprocess_data(df_no_disease, features)

    disease_features, no_disease_features = standardize_features(
        X_disease, X_no_disease, features
    )

    # cov_inv = compute_weighted_covariance(
    #     disease_features, no_disease_features, weights
    # )

    matches_df = exact_matches_pdist(
        disease_data=X_disease,
        no_disease_data=X_no_disease,
        disease_features=disease_features,
        no_disease_features=no_disease_features,
        enable_deduplication=True
    )

    matches_df.to_csv(output_path, index=False)
    print(matches_df.head(20))

    total_distance = matches_df['Distance'].sum()
    print("Total distance:", total_distance)

if __name__ == "__main__":
    main()