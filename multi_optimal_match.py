import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from multiprocessing import Manager, Queue, Process, cpu_count, Pool
import multiprocessing as mp
import time

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

def parallel_optimal_matches(disease_data, no_disease_data, 
                   disease_features, no_disease_features, 
                   cov_inv):
    start_time = time.perf_counter()
# assign numpy format values to features
    disease_features = disease_features.values
    no_disease_features = no_disease_features.values
    
    task_queue = Queue()
    result_queue = Queue()

    id_groups = disease_data.groupby('ID')
    batch_size = 100
    id_batches = [list(id_groups.groups.keys())[i:i+batch_size] 
                 for i in range(0, len(id_groups.groups), batch_size)]

    for id_batch in id_batches:
        batch_data = disease_data[disease_data['ID'].isin(id_batch)]
        task_queue.put({
            'disease_batch': batch_data,
            'feature_batch': disease_features[batch_data.index],
            'no_disease_data': no_disease_data,
            'no_disease_features': no_disease_features,
            'cov_inv': cov_inv
        })

    num_workers = max(1, cpu_count() - 2)
    processes = []
    for _ in range(num_workers):
        p = Process(target=batch_worker, args=(task_queue, result_queue))
        p.start()
        processes.append(p)

    for _ in range(num_workers):
        task_queue.put(None)

    cost_blocks = []
    while len(cost_blocks) < len(id_batches):
        cost_blocks.append(result_queue.get())

    cost_matrix = np.full((len(disease_data), len(no_disease_data)), np.inf)
    for block in cost_blocks:
        i_start = block['start_idx']
        i_end = block['end_idx']
        cost_matrix[i_start:i_end, :] = block['matrix']

    row_ind, col_ind = linear_sum_assignment(cost_matrix)


    matches = []
    for i, j in zip(row_ind, col_ind):
        if np.isinf(cost_matrix[i,j]):
            continue
        
        matches.append({
            'Disease ID': disease_data.iloc[i]['ID'],
            'Disease Date': disease_data.iloc[i]['date'],
            'no Disease ID': no_disease_data.iloc[j]['ID'],
            'no Disease Date': no_disease_data.iloc[j]['date'],
            'Distance': cost_matrix[i,j]
        })

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    return pd.DataFrame(matches), elapsed_time

def batch_worker(task_queue, result_queue):
    while True:
        task = task_queue.get()
        if task is None:
            break

        disease_batch = task['disease_batch']
        feature_batch = task['feature_batch']
        no_disease_data = task['no_disease_data']
        no_disease_features = task['no_disease_features']
        cov_inv = task['cov_inv']

        block_matrix = np.full((len(disease_batch), len(no_disease_data)), np.inf)

        for i in range(len(disease_batch)):
            disease_id = disease_batch.iloc[i]['ID']
            mask = (no_disease_data['ID'] != disease_id).values
            valid_j = np.where(mask)[0]
            
            if valid_j.size == 0:
                continue

            distances = cdist(
                feature_batch[i].reshape(1, -1),
                no_disease_features[valid_j],
                metric='mahalanobis',
                VI=cov_inv
            )
            block_matrix[i, valid_j] = distances.ravel()

        result_queue.put({
            'start_idx': disease_batch.index[0],
            'end_idx': disease_batch.index[-1] + 1,
            'matrix': block_matrix
        })

def main():
    features = ['age', 'gender', 'blood_pressure', 'height', 'weight', 'BMI']
    weights = np.array([0.2, 0.1, 0.3, 0.1, 0.1, 0.2])
    output_path = "multi_optimal_match.csv"

    X_disease = preprocess_data(df_disease, features)
    X_no_disease = preprocess_data(df_no_disease, features)

    disease_features, no_disease_features = standardize_features(
        X_disease, X_no_disease, features
    )

    cov_inv = compute_weighted_covariance(
        disease_features, no_disease_features, weights
    )

    matches_df, elapsed_time = parallel_optimal_matches(
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

    print(f"Total time: {elapsed_time}")

if __name__ == "__main__":
    main()