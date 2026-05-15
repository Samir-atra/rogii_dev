"""Feature engineering for wellbore data.

This module derives new features from raw logs and spatial coordinates.
"""

import polars as pl
import jax.numpy as jnp
import os

def add_spatial_features(df):
    """Calculates spatial gradients and distances.
    
    Args:
        df: Polars DataFrame with X, Y, Z, MD.
        
    Returns:
        DataFrame with added spatial features.
    """
    # Calculate differentials
    df = df.with_columns([
        (pl.col("X").diff().fill_null(0)).alias("dX"),
        (pl.col("Y").diff().fill_null(0)).alias("dY"),
        (pl.col("Z").diff().fill_null(0)).alias("dZ"),
        (pl.col("MD").diff().fill_null(0)).alias("dMD")
    ])
    
    # Use JAX for Euclidean distance if needed, but Polars is faster for basic ops
    # Inclination calculation: arctan(sqrt(dX^2 + dY^2) / dZ)
    df = df.with_columns(
        ((pl.col("dX")**2 + pl.col("dY")**2).sqrt() / pl.col("dZ").replace(0, 1e-6)).arctan().alias("inclination")
    )
    
    return df

def add_rolling_features(df, window_sizes=[10, 50, 100]):
    """Adds rolling statistics for Gamma Ray logs.
    
    Args:
        df: Polars DataFrame with 'GR' column.
        window_sizes: List of integers for rolling windows.
        
    Returns:
        DataFrame with added rolling features.
    """
    for w in window_sizes:
        df = df.with_columns([
            pl.col("GR").rolling_mean(window_size=w).fill_null(strategy="forward").fill_null(strategy="backward").alias(f"GR_mean_{w}"),
            pl.col("GR").rolling_std(window_size=w).fill_null(0).alias(f"GR_std_{w}"),
            pl.col("GR").rolling_min(window_size=w).fill_null(strategy="forward").fill_null(strategy="backward").alias(f"GR_min_{w}"),
            pl.col("GR").rolling_max(window_size=w).fill_null(strategy="forward").fill_null(strategy="backward").alias(f"GR_max_{w}")
        ])
    return df

def process_well_features(df):
    """Full feature engineering pipeline for a well.
    
    Args:
        df: Raw well DataFrame.
        
    Returns:
        Feature-enriched DataFrame.
    """
    df = add_spatial_features(df)
    df = add_rolling_features(df)
    return df

if __name__ == "__main__":
    from data_loader import load_well_pair, preprocess_logs, get_all_well_ids
    
    DATA_PATH = "/home/samer/Documents/competitions/ROGII/dataset/"
    well_ids = get_all_well_ids(DATA_PATH)
    
    h_df, _ = load_well_pair(DATA_PATH, well_ids[0])
    h_df = preprocess_logs(h_df)
    h_df = process_well_features(h_df)
    
    print(f"Features engineered for well {well_ids[0]}. Columns: {h_df.columns}")
    print(h_df.head())
    
