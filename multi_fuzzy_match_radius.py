import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import time
import warnings
import math
warnings.filterwarnings('ignore')  # Ignore Mahalanobis distance warnings

# --------------------------
# Data Preprocessing Functions
# --------------------------
def preprocess_data(df, features):
    df = df.copy()
    if 'gender' in features:
        df['gender'] = df['gender'].map({'female': 0, 'male': 1})
    return df[features + ['ID', 'date']]

def standardize_features(*dfs, features):
    combined = pd.concat([df[features] for df in dfs])
    mean, std = combined.mean(), combined.std()
    return [(df[features] - mean) / std for df in dfs]

def compute_global_covariance(disease_feat, no_disease_feat, weights):
    """Compute globally weighted inverse covariance matrix"""
    combined = pd.concat([disease_feat, no_disease_feat])
    cov = combined.cov().values
    W = np.outer(weights, weights)
    weighted_cov = cov / (W + 1e-8)
    return np.linalg.pinv(weighted_cov)

# --------------------------
# Improved Block Distance Calculation
# --------------------------
def compute_block_distance(disease_data, no_disease_data, 
                          disease_features, no_disease_features, 
                          cov_inv):
    """
    Calculate matching distances for entire block (n×m)
    """
    start_time = time.perf_counter()
    n = len(disease_data)
    m = len(no_disease_data)
    
    # Create full cost matrix
    cost_matrix = np.full((n, m), np.inf)
    
    # Build ID mask matrix (exclude same ID)
    id_mask = np.zeros((n, m), dtype=bool)
    for i in range(n):
        disease_id = disease_data.iloc[i]['ID']
        id_mask[i] = (no_disease_data['ID'] == disease_id).values
    
    # Vectorized calculation of all pairwise differences
    diff = disease_features.values[:, None, :] - no_disease_features.values[None, :, :]
    
    # Vectorized Mahalanobis distance calculation
    quad_form = np.einsum('ijk,kl,ijl->ij', diff, cov_inv, diff)
    distances = np.sqrt(np.maximum(quad_form, 0))
    
    # Apply ID mask: set same ID to infinity (prevent matching)
    distances[id_mask] = np.inf
    
    # Set cost matrix
    cost_matrix = distances
    
    # Apply Hungarian algorithm for optimal matching
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except ValueError:
        # Handle all-INF matrix case
        return pd.DataFrame(), time.perf_counter() - start_time
    
    # Collect matching results
    matches = []
    for i, j in zip(row_ind, col_ind):
        cost = cost_matrix[i, j]
        if np.isinf(cost):
            continue
            
        matches.append({
            'disease_id': disease_data.iloc[i]['ID'],
            'disease_date': disease_data.iloc[i]['date'],
            'no_disease_id': no_disease_data.iloc[j]['ID'],
            'no_disease_date': no_disease_data.iloc[j]['date'],
            'distance': cost
        })
    
    elapsed_time = time.perf_counter() - start_time
    return pd.DataFrame(matches), elapsed_time

# --------------------------
# Optimized DataBlockManager (Distance-based)
# --------------------------
class RadiusBlockManager:
    def __init__(self, disease_df, no_disease_df, 
                disease_feat_df, no_disease_feat_df, 
                features, cov_inv):
        # Raw data (with ID and date)
        self.disease_data = disease_df.reset_index(drop=True)
        self.no_disease_data = no_disease_df.reset_index(drop=True)
        
        # Feature data (standardized values)
        self.disease_feat = disease_feat_df.reset_index(drop=True)
        self.no_disease_feat = no_disease_feat_df.reset_index(drop=True)
        
        self.features = features
        self.cov_inv = cov_inv
        
        # Allocation status
        self.disease_allocated = np.zeros(len(disease_df), dtype=bool)
        self.no_disease_allocated = np.zeros(len(no_disease_df), dtype=bool)
        self.seed_used = np.zeros(len(disease_df), dtype=bool)  # Track seed usage
        
        # Sample counts
        self.n_disease = len(disease_df)
        self.n_no_disease = len(no_disease_df)
        
        # Initialize NN models with standardized features
        self.disease_nn = NearestNeighbors(
            metric='mahalanobis',
            metric_params={'VI': cov_inv}
        )
        self.disease_nn.fit(self.disease_feat.values)
        
        self.no_disease_nn = NearestNeighbors(
            metric='mahalanobis',
            metric_params={'VI': cov_inv}
        )
        self.no_disease_nn.fit(self.no_disease_feat.values)
    
    def get_available_seeds(self):
        """Get all unallocated disease samples as seed points"""
        available_indices = np.where(~self.disease_allocated)[0]
        if len(available_indices) == 0:
            return None
        return self.disease_feat.iloc[available_indices].values
    
    def allocate_block_by_radius(self, seed, radius):
        """Allocate block based on distance threshold, ensuring seed point participates"""
        # Step 1: Find index corresponding to seed point
        _, seed_idx = self.disease_nn.kneighbors([seed], n_neighbors=1)
        seed_idx = seed_idx[0][0]
        
        # Skip if seed already allocated or used
        if self.disease_allocated[seed_idx] or self.seed_used[seed_idx]:
            return None
        
        # Mark seed as used
        self.seed_used[seed_idx] = True
        
        # Find disease samples within radius (including distances)
        disease_dists, disease_indices = self.disease_nn.radius_neighbors(
            [seed], radius=radius, return_distance=True
        )
        disease_dists = disease_dists[0]
        disease_indices = disease_indices[0]
        
        # Filter unallocated cases
        valid_disease = []
        for i, idx in enumerate(disease_indices):
            if not self.disease_allocated[idx]:
                valid_disease.append((disease_dists[i], idx))
        
        # Sort by distance
        valid_disease.sort(key=lambda x: x[0])
        disease_selected = [seed_idx]  # Ensure seed is included
        
        # Find control samples within radius (including distances)
        no_disease_dists, no_disease_indices = self.no_disease_nn.radius_neighbors(
            [seed], radius=radius, return_distance=True
        )
        no_disease_dists = no_disease_dists[0]
        no_disease_indices = no_disease_indices[0]
        
        # Filter unallocated controls
        valid_no_disease = []
        for i, idx in enumerate(no_disease_indices):
            if not self.no_disease_allocated[idx]:
                valid_no_disease.append((no_disease_dists[i], idx))
        
        # Sort by distance
        valid_no_disease.sort(key=lambda x: x[0])
        no_disease_selected = []
        
        # Determine block size (ensure disease samples match controls)
        max_size = min(len(valid_disease), len(valid_no_disease))
        if max_size < 1:  # Need at least 1 disease and 1 control
            return None
        
        # Select disease samples (including seed)
        for dist, idx in valid_disease:
            if idx == seed_idx:  # Seed already included
                continue
            if len(disease_selected) < max_size:
                disease_selected.append(idx)
        
        # Select control samples (match disease sample count)
        for i in range(min(len(disease_selected), len(valid_no_disease))):
            no_disease_selected.append(valid_no_disease[i][1])
        
        # Skip if no valid samples found
        if not disease_selected or not no_disease_selected:
            return None
        
        # Mark as allocated
        self.disease_allocated[disease_selected] = True
        self.no_disease_allocated[no_disease_selected] = True
        
        # Return complete data block
        return (
            self.disease_data.iloc[disease_selected],
            self.no_disease_data.iloc[no_disease_selected],
            self.disease_feat.iloc[disease_selected],
            self.no_disease_feat.iloc[no_disease_selected]
        )
    
    def get_unmatched_data(self):
        """Get all unmatched samples"""
        unmatched_disease_idx = np.where(~self.disease_allocated)[0]
        unmatched_no_disease_idx = np.where(~self.no_disease_allocated)[0]
        
        return (
            self.disease_data.iloc[unmatched_disease_idx],
            self.no_disease_data.iloc[unmatched_no_disease_idx],
            self.disease_feat.iloc[unmatched_disease_idx],
            self.no_disease_feat.iloc[unmatched_no_disease_idx]
        )
    
    def has_unmatched_samples(self):
        """Check if unmatched samples exist"""
        return (np.sum(~self.disease_allocated) > 0) and \
               (np.sum(~self.no_disease_allocated) > 0)
    
    def get_unmatched_disease_count(self):
        """Get count of unmatched disease samples"""
        return np.sum(~self.disease_allocated)
    
    def select_new_seeds(self, fraction=0.1):
        """Select new seed points from unmatched disease samples"""
        unmatched_indices = np.where(~self.disease_allocated)[0]
        if len(unmatched_indices) == 0:
            return None
            
        n_seeds = max(1, int(len(unmatched_indices) * fraction))
        
        # Randomly select seeds
        selected_indices = np.random.choice(
            unmatched_indices, size=n_seeds, replace=False
        )
        
        # Reset usage status for these seeds
        self.seed_used[selected_indices] = False
        
        return self.disease_feat.iloc[selected_indices].values

# --------------------------
# Optimized Main Workflow (Distance-based allocation)
# --------------------------
def main():
    start_time = time.time()
    
    # Configuration parameters
    features = ['age', 'gender', 'blood_pressure', 'height', 'weight', 'BMI']
    weights = np.array([0.2, 0.1, 0.3, 0.1, 0.1, 0.2])
    output_path = "optimal_matches.csv"
    
    # Distance threshold configuration
    initial_radius = 0.2  # Initial distance threshold
    expanded_radius = 10  # Expanded distance threshold
    seed_fraction = 0.1    # Fraction for selecting new seeds
    
    # Read data
    print("Reading data...")
    df_disease = pd.read_csv("disease.csv")
    df_no_disease = pd.read_csv("no_disease.csv")
    
    # Check if data exists
    if df_disease.empty or df_no_disease.empty:
        print("Error: Input data is empty!")
        return
    
    # Preprocess data
    print("Preprocessing data...")
    X_disease = preprocess_data(df_disease, features)
    X_no_disease = preprocess_data(df_no_disease, features)
    
    # Check sample counts
    if X_disease.empty or X_no_disease.empty:
        print("Error: Preprocessed data is empty!")
        return
    
    # Standardize features
    print("Standardizing features...")
    disease_feat, no_disease_feat = standardize_features(
        X_disease, X_no_disease, features=features
    )
    
    # Compute global inverse covariance
    print("Computing covariance matrix...")
    cov_inv = compute_global_covariance(
        disease_feat, no_disease_feat, weights
    )
    
    # Initialize block manager
    print("Initializing matching manager...")
    manager = RadiusBlockManager(
        disease_df=X_disease, 
        no_disease_df=X_no_disease,
        disease_feat_df=disease_feat,
        no_disease_feat_df=no_disease_feat,
        features=features,
        cov_inv=cov_inv
    )
    
    # Check if samples need matching
    if not manager.has_unmatched_samples():
        print("Error: No samples to match!")
        return
    
    total_distance = 0.0
    all_matches = []
    matched_pairs_count = 0
    
    # Phase 1: Use initial distance threshold
    print("\n=== Phase 1: Using initial distance threshold (radius=0.08) ===")
    radius = initial_radius
    seeds = manager.get_available_seeds()
    
    if seeds is None:
        print("No available seed points in Phase 1")
    else:
        seed_index = 0
        blocks_processed = 0
        
        while seed_index < len(seeds):
            seed = seeds[seed_index]
            seed_index += 1
            
            block = manager.allocate_block_by_radius(seed, radius)
            if block is None:
                continue
            
            blocks_processed += 1
            
            # Unpack data block
            disease_block, no_disease_block, disease_feat_block, no_disease_feat_block = block
            
            # Ensure seed participates in matching
            seed_id = disease_block.iloc[0]['ID']  # Seed is always first
            
            # Calculate matching for current block
            matches_df, _ = compute_block_distance(
                disease_block, no_disease_block,
                disease_feat_block, no_disease_feat_block,
                cov_inv
            )
            
            # Check if seed was matched
            seed_matched = False
            if not matches_df.empty:
                if 'disease_id' in matches_df.columns:
                    seed_matched = any(matches_df['disease_id'] == seed_id)
                else:
                    print(f"Warning: 'disease_id' column missing in matches, actual columns: {matches_df.columns.tolist()}")
            
            if seed_matched:
                print(f"Seed {seed_id} successfully matched")
            else:
                print(f"Warning: Seed {seed_id} not matched")
            
            # Update statistics
            if not matches_df.empty:
                total_distance += matches_df['distance'].sum()
                matched_pairs_count += len(matches_df)
                all_matches.append(matches_df)
        
        print(f"Phase 1 complete: Blocks processed={blocks_processed}, Matched pairs={matched_pairs_count}")
    
    # Phase 2: Expand distance threshold and select new seeds
    print("\n=== Phase 2: Expanding distance threshold and selecting new seeds (radius=0.32) ===")
    radius = expanded_radius
    
    # Select new seeds (10% of unmatched disease samples)
    new_seeds = manager.select_new_seeds(seed_fraction)
    
    if new_seeds is None or len(new_seeds) == 0:
        print("No new seed points available")
    else:
        print(f"Selected {len(new_seeds)} new seed points")
        seed_index = 0
        blocks_processed = 0
        new_matches = 0
        
        while seed_index < len(new_seeds):
            seed = new_seeds[seed_index]
            seed_index += 1
            
            block = manager.allocate_block_by_radius(seed, radius)
            if block is None:
                continue
            
            blocks_processed += 1
            
            # Unpack data block
            disease_block, no_disease_block, disease_feat_block, no_disease_feat_block = block
            
            # Ensure seed participates in matching
            seed_id = disease_block.iloc[0]['ID']  # Seed is always first
            
            # Calculate matching for current block
            matches_df, _ = compute_block_distance(
                disease_block, no_disease_block,
                disease_feat_block, no_disease_feat_block,
                cov_inv
            )
            
            # Check if seed was matched
            seed_matched = False
            if not matches_df.empty:
                if 'disease_id' in matches_df.columns:
                    seed_matched = any(matches_df['disease_id'] == seed_id)
                else:
                    print(f"Warning: 'disease_id' column missing in matches, actual columns: {matches_df.columns.tolist()}")
            
            if seed_matched:
                print(f"Seed {seed_id} successfully matched")
            else:
                print(f"Warning: Seed {seed_id} not matched")
            
            # Update statistics
            if not matches_df.empty:
                total_distance += matches_df['distance'].sum()
                new_matches += len(matches_df)
                all_matches.append(matches_df)
        
        matched_pairs_count += new_matches
        print(f"Phase 2 complete: Blocks processed={blocks_processed}, New matched pairs={new_matches}")
    
    # Collect all matching results
    if all_matches:
        matches_df = pd.concat(all_matches, ignore_index=True)
    else:
        matches_df = pd.DataFrame()
    
    # Get unmatched samples
    unmatched_disease_data, unmatched_no_disease_data, \
    unmatched_disease_feat, unmatched_no_disease_feat = manager.get_unmatched_data()
    
    print(f"\nUnmatched samples: {len(unmatched_disease_data)} cases, {len(unmatched_no_disease_data)} controls")
    
    # Final matching phase (if unmatched samples exist)
    if len(unmatched_disease_data) > 0 and len(unmatched_no_disease_data) > 0:
        print("\n=== Final Matching Phase ===")
        
        # Perform global matching
        final_matches, _ = compute_block_distance(
            unmatched_disease_data, unmatched_no_disease_data,
            unmatched_disease_feat, unmatched_no_disease_feat,
            cov_inv
        )
        
        if not final_matches.empty:
            # Merge with main results
            if matches_df.empty:
                matches_df = final_matches
            else:
                matches_df = pd.concat([matches_df, final_matches], ignore_index=True)
            
            # Update statistics
            final_pairs = len(final_matches)
            matched_pairs_count += final_pairs
            total_distance += final_matches['distance'].sum()
            print(f"Final matching: {final_pairs} pairs, Total matched: {matched_pairs_count}")
    
    # Output final results
    end_time = time.time()
    total_time = end_time - start_time
    
    # Save matching results
    matches_df.to_csv(output_path, index=False)
    print(f"\nMatching results saved to: {output_path}")
    
    # Print statistics
    print("\nMatching complete!")
    print("=" * 50)
    print(f"Total matched pairs: {matched_pairs_count}")
    print(f"Total distance: {total_distance:.4f}")
    print(f"Total processing time: {total_time:.2f} seconds")
    
    # Print first 20 matches
    if not matches_df.empty:
        print("\nFirst 20 matched records:")
        print(matches_df.head(20))
    else:
        print("No matches found")
    
    # Check unmatched samples
    if manager:
        unmatched_disease = np.sum(~manager.disease_allocated)
        unmatched_no_disease = np.sum(~manager.no_disease_allocated)
        print("\nUnmatched statistics:")
        print(f"Disease samples: {unmatched_disease} / {manager.n_disease}")
        print(f"Control samples: {unmatched_no_disease} / {manager.n_no_disease}")
        
        if manager.n_disease > 0:
            print(f"Match coverage: {matched_pairs_count / manager.n_disease:.2%}")

if __name__ == "__main__":
    main()