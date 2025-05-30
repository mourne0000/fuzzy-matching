import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import time
import warnings
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
            'Disease ID': disease_data.iloc[i]['ID'],
            'Disease Date': disease_data.iloc[i]['date'],
            'No Disease ID': no_disease_data.iloc[j]['ID'],
            'No Disease Date': no_disease_data.iloc[j]['date'],
            'Distance': cost
        })
    
    elapsed_time = time.perf_counter() - start_time
    return pd.DataFrame(matches), elapsed_time

# --------------------------
# Optimized DataBlockManager
# --------------------------
class DataBlockManager:
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
        
        # Sample counts
        self.n_disease = len(disease_df)
        self.n_no_disease = len(no_disease_df)
        
        # Initialize NN models with standardized features
        self.disease_nn = NearestNeighbors(
            n_neighbors=min(1000, self.n_disease), 
            metric='mahalanobis',
            metric_params={'VI': cov_inv}
        )
        self.disease_nn.fit(self.disease_feat.values)
        
        self.no_disease_nn = NearestNeighbors(
            n_neighbors=min(1000, self.n_no_disease), 
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

    def allocate_block(self, seed, k=50):
        """Return data block containing seed and other cases/controls"""
        # Step 1: Find index corresponding to seed point
        _, seed_idx = self.disease_nn.kneighbors([seed], n_neighbors=1)
        seed_idx = seed_idx[0][0]
        
        # Skip if seed already allocated
        if self.disease_allocated[seed_idx]:
            return None
        
        # Step 2: Check remaining samples
        remaining_disease = self.n_disease - np.sum(self.disease_allocated)
        remaining_no_disease = self.n_no_disease - np.sum(self.no_disease_allocated)
        
        # Return if either group is insufficient
        if remaining_disease == 0 or remaining_no_disease == 0:
            return None
        
        # Determine block size (min of remaining samples, capped at k)
        block_size = min(k, min(remaining_disease, remaining_no_disease))
        
        # Handle case with few remaining samples
        if remaining_disease <= 100 or remaining_no_disease <= 100:
            block_size = min(block_size, min(remaining_disease, remaining_no_disease))
        
        # Step 3: Select disease samples
        _, disease_indices = self.disease_nn.kneighbors([seed], n_neighbors=min(1000, self.n_disease))
        disease_indices = disease_indices[0]
        
        # Filter unallocated cases
        disease_selected = []
        for idx in disease_indices:
            if not self.disease_allocated[idx]:
                disease_selected.append(idx)
                if len(disease_selected) >= block_size:
                    break
        
        # Ensure seed is included
        if seed_idx not in disease_selected:
            if len(disease_selected) < block_size:
                disease_selected.append(seed_idx)
            else:
                # Replace farthest sample
                disease_features = self.disease_feat.iloc[disease_selected].values
                distances = cdist([seed], disease_features, metric='mahalanobis', VI=self.cov_inv)[0]
                farthest_idx = np.argmax(distances)
                disease_selected[farthest_idx] = seed_idx
        
        # Step 4: Select control samples
        _, no_disease_indices = self.no_disease_nn.kneighbors([seed], n_neighbors=min(1000, self.n_no_disease))
        no_disease_indices = no_disease_indices[0]
        
        no_disease_selected = []
        for idx in no_disease_indices:
            if not self.no_disease_allocated[idx]:
                no_disease_selected.append(idx)
                if len(no_disease_selected) >= block_size:
                    break
        
        # Expand search if insufficient controls
        if len(no_disease_selected) < block_size:
            _, no_disease_indices = self.no_disease_nn.kneighbors([seed], n_neighbors=min(2000, self.n_no_disease))
            no_disease_indices = no_disease_indices[0]
            
            for idx in no_disease_indices:
                if not self.no_disease_allocated[idx] and idx not in no_disease_selected:
                    no_disease_selected.append(idx)
                    if len(no_disease_selected) >= block_size:
                        break
        
        # Ensure equal group sizes
        final_size = min(len(disease_selected), len(no_disease_selected))
        if final_size == 0:
            return None
        
        # Truncate lists to maintain equal sizes
        if len(disease_selected) > final_size:
            disease_features = self.disease_feat.iloc[disease_selected].values
            distances = cdist([seed], disease_features, metric='mahalanobis', VI=self.cov_inv)[0]
            sorted_indices = np.argsort(distances)
            disease_selected = [disease_selected[i] for i in sorted_indices][:final_size]
        
        if len(no_disease_selected) > final_size:
            no_disease_features = self.no_disease_feat.iloc[no_disease_selected].values
            distances = cdist([seed], no_disease_features, metric='mahalanobis', VI=self.cov_inv)[0]
            sorted_indices = np.argsort(distances)
            no_disease_selected = [no_disease_selected[i] for i in sorted_indices][:final_size]
        
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
        return (np.sum(~self.disease_allocated) > 0 and 
            np.sum(~self.no_disease_allocated) > 0)
    
def additional_matching(unmatched_disease_data, unmatched_no_disease_data,
                       unmatched_disease_feat, unmatched_no_disease_feat,
                       cov_inv, min_block_size=5, max_seeds=1000):
    """
    Additional matching for remaining unmatched samples
    """
    print(f"\nStarting additional matching: {len(unmatched_disease_data)} cases vs {len(unmatched_no_disease_data)} controls")
    
    # Global matching if sample size is small
    if len(unmatched_disease_data) < 10 or len(unmatched_no_disease_data) < 10:
        print("Few samples remaining, performing global matching")
        matches_df, _ = compute_block_distance(
            unmatched_disease_data, unmatched_no_disease_data,
            unmatched_disease_feat, unmatched_no_disease_feat,
            cov_inv
        )
        return matches_df
    
    # Create new block manager
    manager = DataBlockManager(
        disease_df=unmatched_disease_data,
        no_disease_df=unmatched_no_disease_data,
        disease_feat_df=unmatched_disease_feat,
        no_disease_feat_df=unmatched_no_disease_feat,
        features=unmatched_disease_feat.columns.tolist(),
        cov_inv=cov_inv
    )
    
    # Different block size strategies
    block_sizes = [min_block_size, 10, 20]
    
    all_matches = []
    total_distance = 0.0
    matched_pairs = 0
    
    for block_size in block_sizes:
        print(f"Trying block size: {block_size}")
        seeds = manager.get_available_seeds()
        if seeds is None or len(seeds) == 0:
            break
            
        # Limit number of seeds
        if len(seeds) > max_seeds:
            seeds = seeds[np.random.choice(len(seeds), max_seeds, replace=False)]
        
        for seed in seeds:
            block = manager.allocate_block(seed, k=block_size)
            if block is None:
                continue
                
            disease_block, no_disease_block, disease_feat_block, no_disease_feat_block = block
            
            # Calculate matching for current block
            matches_df, _ = compute_block_distance(
                disease_block, no_disease_block,
                disease_feat_block, no_disease_feat_block,
                cov_inv
            )
            
            if not matches_df.empty:
                total_distance += matches_df['Distance'].sum()
                matched_pairs += len(matches_df)
                all_matches.append(matches_df)
    
    # Combine results
    if all_matches:
        return pd.concat(all_matches, ignore_index=True)
    return pd.DataFrame()

# --------------------------
# Optimized Main Workflow
# --------------------------
def main():
    start_time = time.time()
    
    # Configuration parameters
    features = ['age', 'gender', 'blood_pressure', 'height', 'weight', 'BMI']
    weights = np.array([0.2, 0.1, 0.3, 0.1, 0.1, 0.2])
    main_block_size = 50  # Main matching block size
    min_block_size = 5    # Additional matching min block size
    output_path = "optimal_matches.csv"
    
    # Block size list
    block_sizes = [20, 10, min_block_size]
    
    # Read data
    print("Reading data...")
    df_disease = pd.read_csv("disease.csv")
    df_no_disease = pd.read_csv("no_disease.csv")
    
    # Preprocess data
    print("Preprocessing data...")
    X_disease = preprocess_data(df_disease, features)
    X_no_disease = preprocess_data(df_no_disease, features)
    
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
    manager = DataBlockManager(
        disease_df=X_disease, 
        no_disease_df=X_no_disease,
        disease_feat_df=disease_feat,
        no_disease_feat_df=no_disease_feat,
        features=features,
        cov_inv=cov_inv
    )
    
    print("\nStarting main matching phase...")
    total_distance = 0.0
    all_matches = []
    matched_pairs_count = 0
    blocks_processed = 0
    skipped_blocks = 0
    failed_blocks = 0
    
    # Dynamically get seed points
    while True:
        seeds = manager.get_available_seeds()
        if seeds is None or len(seeds) == 0:
            print("All disease samples allocated")
            break
        
        # Randomly select a seed point
        seed = seeds[np.random.choice(len(seeds))]
        print(f"\nProcessing block #{blocks_processed+1}: Seed={seed}")
        
        # Attempt to allocate block
        block = manager.allocate_block(seed, k=main_block_size)
        if block is None:
            skipped_blocks += 1
            print(f"Block allocation failed: Skipped {skipped_blocks} blocks")
            
            if skipped_blocks > 100:
                remaining_disease = np.sum(~manager.disease_allocated)
                remaining_no_disease = np.sum(~manager.no_disease_allocated)
                print(f"Too many consecutive skips. Remaining cases={remaining_disease} Remaining controls={remaining_no_disease}")
                break
            continue
        
        skipped_blocks = 0  # Reset skip counter
        blocks_processed += 1
        
        # Unpack data block
        disease_block, no_disease_block, disease_feat_block, no_disease_feat_block = block
        print(f"Block allocated: {len(disease_block)} cases vs {len(no_disease_block)} controls")
        
        # Calculate matching for current block
        matches_df, block_time = compute_block_distance(
            disease_block, no_disease_block,
            disease_feat_block, no_disease_feat_block,
            cov_inv
        )
        
        # Update statistics
        if not matches_df.empty:
            matched_count = len(matches_df)
            total_distance += matches_df['Distance'].sum()
            matched_pairs_count += matched_count
            all_matches.append(matches_df)
            print(f"Matched {matched_count} pairs, Total matched={matched_pairs_count}")
        else:
            failed_blocks += 1
            print("No matches found in current block")
    
    
    # Main matching phase complete
    main_match_time = time.time()
    print(f"\nMain matching complete: {matched_pairs_count} pairs, Time: {main_match_time - start_time:.2f}s")
    
    # Collect main matching results
    if all_matches:
        matches_df = pd.concat(all_matches, ignore_index=True)
    else:
        matches_df = pd.DataFrame()
    
    # Get unmatched samples
    unmatched_disease_data, unmatched_no_disease_data, \
    unmatched_disease_feat, unmatched_no_disease_feat = manager.get_unmatched_data()
    
    print(f"\nUnmatched samples: {len(unmatched_disease_data)} cases, {len(unmatched_no_disease_data)} controls")
    
    # Additional matching phase
    if len(unmatched_disease_data) > 0 and len(unmatched_no_disease_data) > 0:
        print("\nStarting additional matching phase...")
        additional_matches = []
        additional_distance = 0.0
        additional_pairs = 0
        additional_blocks = 0
        
        # Try different block size strategies
        for block_size in block_sizes:
            print(f"Trying additional block size: {block_size}")
            
            # Create new manager with remaining samples
            sub_manager = DataBlockManager(
                disease_df=unmatched_disease_data,
                no_disease_df=unmatched_no_disease_data,
                disease_feat_df=unmatched_disease_feat,
                no_disease_feat_df=unmatched_no_disease_feat,
                features=features,
                cov_inv=cov_inv
            )
            
            sub_blocks = 0
            sub_skipped = 0
            max_sub_blocks = min(100, len(unmatched_disease_data))  # Limit blocks
            
            while sub_blocks < max_sub_blocks:
                seeds = sub_manager.get_available_seeds()
                if seeds is None or len(seeds) == 0:
                    print("  All remaining disease samples allocated")
                    break
                
                # Randomly select seed point
                seed = seeds[np.random.choice(len(seeds))]
                
                # Attempt to allocate block
                block = sub_manager.allocate_block(seed, k=block_size)
                if block is None:
                    sub_skipped += 1
                    if sub_skipped > 20:
                        print("  Too many consecutive skips, stopping current block size")
                        break
                    continue
                
                sub_skipped = 0
                sub_blocks += 1
                additional_blocks += 1
                
                # Unpack data block
                d_block, nd_block, df_block, ndf_block = block
                
                # Calculate matching for current block
                matches_df, _ = compute_block_distance(
                    d_block, nd_block,
                    df_block, ndf_block,
                    cov_inv
                )
                
                if not matches_df.empty:
                    additional_matches.append(matches_df)
                    additional_distance += matches_df['Distance'].sum()
                    additional_pairs += len(matches_df)
            
            # Update remaining unmatched samples
            unmatched_disease_data, unmatched_no_disease_data, \
            unmatched_disease_feat, unmatched_no_disease_feat = sub_manager.get_unmatched_data()
            
            # Exit early if no samples left
            if len(unmatched_disease_data) == 0 or len(unmatched_no_disease_data) == 0:
                break
        
        # Process additional matches
        if additional_matches:
            additional_matches_df = pd.concat(additional_matches, ignore_index=True)
            
            # Merge with main results
            if matches_df.empty:
                matches_df = additional_matches_df
            else:
                matches_df = pd.concat([matches_df, additional_matches_df], ignore_index=True)
            
            # Update statistics
            matched_pairs_count += additional_pairs
            total_distance += additional_distance
            print(f"\nAdditional matching complete: {additional_pairs} pairs, Total matched: {matched_pairs_count}")
        
    # Final unmatched sample handling
    if len(unmatched_disease_data) > 0 and len(unmatched_no_disease_data) > 0:
        print(f"\nFinal unmatched samples: {len(unmatched_disease_data)} cases vs {len(unmatched_no_disease_data)} controls")
        
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
            total_distance += final_matches['Distance'].sum()
            print(f"  Final matching: {final_pairs} pairs, Total matched: {matched_pairs_count}")
    
    # Output final results
    end_time = time.time()
    total_time = end_time - start_time
    
    # Save results
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
        print("\nFirst 20 matches:")
        print(matches_df.head(20))
    else:
        print("No matches found")
    
    # Check unmatched samples
    unmatched_disease = len(unmatched_disease_data)
    unmatched_no_disease = len(unmatched_no_disease_data)
    total_disease = manager.n_disease
    total_no_disease = manager.n_no_disease
    
    print("\nUnmatched statistics:")
    print(f"Disease samples: {unmatched_disease} / {total_disease}")
    print(f"Control samples: {unmatched_no_disease} / {total_no_disease}")
    print(f"Match coverage: {matched_pairs_count / total_disease:.2%}")

if __name__ == "__main__":
    main()