import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import time
from multiprocessing import Manager, Queue, Process, cpu_count
import multiprocessing as mp

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

def parallel_greedy_matches(disease_data, no_disease_data, 
                          disease_features, no_disease_features, 
                          cov_inv, enable_deduplication=True):
    start_time = time.perf_counter()

    manager = Manager()
    used_indices = manager.dict() if enable_deduplication else None

    disease_features_np = disease_features.values
    no_disease_features_np = no_disease_features.values

    batch_size = 100
    batches = [disease_data.iloc[i:i+batch_size] 
              for i in range(0, len(disease_data), batch_size)]

    task_queue = Queue()
    result_queue = Queue()
    processes = []
    num_workers = max(1, cpu_count() - 2)

    for batch in batches:
        task_queue.put({
            'disease_batch': batch,
            'feature_batch': disease_features_np[batch.index]
        })

    for _ in range(num_workers):
        p = Process(
            target=worker,
            args=(task_queue, result_queue, no_disease_data,
                  no_disease_features_np, cov_inv, used_indices)
        )
        p.start()
        processes.append(p)

    matches = []
    for _ in range(len(batches)):
        matches.extend(result_queue.get())

    for _ in range(num_workers):
        task_queue.put(None)
    for p in processes:
        p.join()
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    return pd.DataFrame(matches), elapsed_time

def worker(task_queue, result_queue, no_disease_data, 
          no_disease_features_np, cov_inv, used_indices):
    while True:
        task = task_queue.get()
        if task is None:
            break
        
        batch_matches = []
        disease_batch = task['disease_batch']
        feature_batch = task['feature_batch']
        
        for i in range(len(disease_batch)):
            disease_row = disease_batch.iloc[i]
            current_features = feature_batch[i].reshape(1, -1)

            mask = (no_disease_data['ID'] != disease_row['ID']).values
            if used_indices is not None:
                mask &= ~np.isin(no_disease_data.index, list(used_indices.keys()))
            
            candidates = no_disease_data[mask]
            if candidates.empty:
                continue

            distances = cdist(
                current_features, 
                no_disease_features_np[mask], 
                metric='mahalanobis', 
                VI=cov_inv
            )
            
            min_idx = np.argmin(distances)
            best_match_idx = candidates.index[min_idx]

            record = {
                'Disease ID': disease_row['ID'],
                'Disease Date': disease_row['date'],
                'no Disease ID': no_disease_data.loc[best_match_idx, 'ID'],
                'no Disease Date': no_disease_data.loc[best_match_idx, 'date'],
                'Distance': distances[0, min_idx]
            }
            batch_matches.append(record)

            if used_indices is not None:
                used_indices[int(best_match_idx)] = True
        
        result_queue.put(batch_matches)

def main():
    features = ['age', 'gender', 'blood_pressure', 'height', 'weight', 'BMI']
    weights = np.array([0.2, 0.1, 0.3, 0.1, 0.1, 0.2])
    output_path = "multi_greedy_match.csv"

    X_disease = preprocess_data(df_disease, features)
    X_no_disease = preprocess_data(df_no_disease, features)

    disease_features, no_disease_features = standardize_features(
        X_disease, X_no_disease, features
    )

    cov_inv = compute_weighted_covariance(
        disease_features, no_disease_features, weights
    )

    matches_df, elapsed_time = parallel_greedy_matches(
        disease_data=X_disease,
        no_disease_data=X_no_disease,
        disease_features=disease_features,
        no_disease_features=no_disease_features,
        cov_inv=cov_inv,
        enable_deduplication=True
    )

    matches_df.to_csv(output_path, index=False)
    print(matches_df.head(20))

    total_distance = matches_df['Distance'].sum()
    print(f"Total distance: {total_distance:.4f}")

    print(f"Total time: {elapsed_time} seconds")

if __name__ == "__main__":
    main()