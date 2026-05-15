"""Data utilities for sequence generation.

This module provides functions to create training/validation sequences from
wellbore logs.
"""

import numpy as jnp # Using jax.numpy
import numpy as np
import polars as pl

def create_sequences(df, feature_cols, target_col, window_size=50):
    """Creates sliding window sequences from a DataFrame.
    
    Args:
        df: Polars DataFrame.
        feature_cols: List of feature column names.
        target_col: Target column name.
        window_size: Length of the sequence.
        
    Returns:
        X (sequences, window_size, features), y (sequences, 1)
    """
    data = df.select(feature_cols + [target_col]).to_numpy()
    
    X = []
    y = []
    
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size, :-1])
        y.append(data[i + window_size, -1])
        
    return np.array(X), np.array(y)

def prepare_all_wells(base_path, well_ids, feature_cols, target_col, window_size=50, is_train=True):
    """Loads and aggregates sequences from multiple wells.
    
    Args:
        base_path: Root data directory.
        well_ids: List of well IDs.
        feature_cols: Features to use.
        target_col: Target variable.
        
    Returns:
        X_all, y_all
    """
    from data_loader import load_well_pair, preprocess_logs
    from feature_engineering import process_well_features
    
    X_list = []
    y_list = []
    
    for well_id in well_ids:
        h_df, _ = load_well_pair(base_path, well_id, is_train=is_train)
        h_df = preprocess_logs(h_df)
        h_df = process_well_features(h_df)
        
        # Filter rows where target is not null (for training)
        if is_train:
            h_df = h_df.filter(pl.col(target_col).is_not_null())
            
        if len(h_df) > window_size:
            X, y = create_sequences(h_df, feature_cols, target_col, window_size)
            X_list.append(X)
            y_list.append(y)
            
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

if __name__ == "__main__":
    from data_loader import get_all_well_ids
    DATA_PATH = "/home/samer/Documents/competitions/ROGII/dataset/"
    well_ids = get_all_well_ids(DATA_PATH)
    
    feature_cols = ["GR", "X", "Y", "Z", "inclination", "GR_mean_50"]
    target_col = "TVT"
    
    X, y = prepare_all_wells(DATA_PATH, well_ids[:2], feature_cols, target_col)
    print(f"Total sequences: {X.shape}")
    print(f"Targets: {y.shape}")
