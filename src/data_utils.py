"""Data utilities for sequence generation.

This module provides functions to create training/validation sequences from
wellbore logs.
"""

import numpy as jnp # Using jax.numpy
import numpy as np
import polars as pl

def engineer_features(df: pl.DataFrame, gr_window_size: int = 50) -> pl.DataFrame:
    """Engineers new features for geosteering prediction.

    This function adds features based on geological and geometric principles:
    1.  Wellbore Inclination: Captures the angle of drilling.
    2.  Rolling GR Mean: Smooths the Gamma Ray signal.
    3.  Rolling GR Std Dev: Captures the texture/variability of the formation.
    4.  Distance to Buda Marker: Provides relative stratigraphic position.

    Args:
        df: The input DataFrame with raw features.
        gr_window_size: The window size for rolling GR statistics.

    Returns:
        The DataFrame with new, engineered feature columns.
    """
    # Ensure columns are float for calculations
    df = df.cast({pl.Float64}, strict=False)

    df = df.with_columns(
        # 1. Wellbore Inclination
        pl.arcsin(
            (pl.col("Z").diff().fill_null(0)) / (pl.col("MD").diff().fill_null(1e-6))
        ).alias("inclination"),

        # 2. Rolling GR Mean
        pl.col("GR").rolling_mean(window_size=gr_window_size).alias(f"GR_mean_{gr_window_size}"),

        # 3. Rolling GR Standard Deviation
        pl.col("GR").rolling_std(window_size=gr_window_size).alias(f"GR_std_{gr_window_size}"),

        # 4. Distance to Buda Marker
        (pl.col("Z") - pl.col("BUDA")).alias("dist_to_buda")
    ).fill_nan(None).fill_null(strategy="forward") # Backfill initial nulls from rolling windows

    return df

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
    from data_loader import load_well_pair, preprocess_logs # Assuming these exist
    
    X_list = []
    y_list = []
    
    for well_id in well_ids:
        h_df, _ = load_well_pair(base_path, well_id, is_train=is_train) # Assuming this function loads horizontal well data
        h_df = preprocess_logs(h_df)
        h_df = engineer_features(h_df) # Apply the new feature engineering
        
        # Filter rows where target is not null to ensure we only train on labeled data
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

    # Define the list of features to use, including the newly engineered ones.
    # We keep the base features as they still provide fundamental information.
    feature_cols = [
        "GR", "X", "Y", "Z",             # Base Features
        "inclination",                   # Engineered Feature 1
        "GR_mean_50",                    # Engineered Feature 2
        "GR_std_50",                     # Engineered Feature 3
        "dist_to_buda"                   # Engineered Feature 4
    ]
    target_col = "TVT"

    # This part of the script would now fail unless load_well_pair and preprocess_logs are properly defined.
    # The following lines are for demonstrating the usage context.
    X, y = prepare_all_wells(DATA_PATH, well_ids[:2], feature_cols, target_col)
    print(f"Total sequences: {X.shape}")
    print(f"Targets: {y.shape}")
