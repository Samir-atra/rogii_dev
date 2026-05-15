"""Data loading utilities for ROGII Wellbore Geology Prediction.

This module provides functions to load and preprocess well data using Polars.
"""

import os
import polars as pl
import glob

def load_well_pair(base_path, well_id, is_train=True):
    """Loads a horizontal well and its corresponding typewell.
    
    Args:
        base_path: Root directory containing 'train' or 'test' folders.
        well_id: The ID of the well to load.
        is_train: Whether to load from the training or testing set.
        
    Returns:
        A tuple of (horizontal_well_df, typewell_df).
    """
    mode = "train" if is_train else "test"
    horiz_path = os.path.join(base_path, mode, f"{well_id}__horizontal_well.csv")
    typewell_path = os.path.join(base_path, mode, f"{well_id}__typewell.csv")
    
    horiz_df = pl.read_csv(horiz_path, infer_schema_length=10000).cast({pl.String: pl.Float64}, strict=False)
    type_df = pl.read_csv(typewell_path, infer_schema_length=10000).cast({pl.String: pl.Float64}, strict=False)
    
    return horiz_df, type_df

def preprocess_logs(df):
    """Interpolates missing GR values in the logs.
    
    Args:
        df: Polars DataFrame containing 'GR' column.
        
    Returns:
        DataFrame with interpolated GR values.
    """
    # Polars linear interpolation
    return df.with_columns(
        pl.col("GR").interpolate()
    ).with_columns(
        pl.col("GR").fill_null(strategy="forward").fill_null(strategy="backward")
    )

def get_all_well_ids(base_path, is_train=True):
    """Retrieves all well IDs from the dataset directory.
    
    Args:
        base_path: Root directory containing 'train' or 'test' folders.
        is_train: Whether to look in the training or testing set.
        
    Returns:
        A list of unique well IDs.
    """
    mode = "train" if is_train else "test"
    files = glob.glob(os.path.join(base_path, mode, "*__horizontal_well.csv"))
    return [os.path.basename(f).split("__")[0] for f in files]

if __name__ == "__main__":
    DATA_PATH = "/home/samer/Documents/competitions/ROGII/dataset/"
    well_ids = get_all_well_ids(DATA_PATH)
    print(f"Found {len(well_ids)} training wells.")
    
    # Load and preprocess first well
    h_df, t_df = load_well_pair(DATA_PATH, well_ids[0])
    h_df = preprocess_logs(h_df)
    print(f"Well {well_ids[0]} loaded. Shape: {h_df.shape}")
