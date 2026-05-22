"""Validation script for ROGII pipeline.

This script runs a full end-to-end check of the data loading, feature 
engineering, and modeling components to ensure everything is functional.
"""

import os
import sys
import polars as pl
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

def validate_data_loader():
    print("--- Validating Data Loader ---")
    from data_loader import get_all_well_ids, load_well_pair, preprocess_logs
    DATA_PATH = "/home/samer/Documents/competitions/ROGII/dataset/"
    
    well_ids = get_all_well_ids(DATA_PATH)
    assert len(well_ids) > 0, "No wells found!"
    print(f"Success: Found {len(well_ids)} wells.")
    
    h_df, t_df = load_well_pair(DATA_PATH, well_ids[0])
    assert isinstance(h_df, pl.DataFrame), "Horizontal well is not a DataFrame"
    assert isinstance(t_df, pl.DataFrame), "Typewell is not a DataFrame"
    
    h_df = preprocess_logs(h_df)
    assert h_df["GR"].null_count() == 0, "GR interpolation failed!"
    print("Success: Data loading and interpolation verified.")
    return well_ids[0]

def validate_features(well_id):
    print("\n--- Validating Feature Engineering ---")
    from data_loader import load_well_pair, preprocess_logs
    from feature_engineering import process_well_features
    DATA_PATH = "/home/samer/Documents/competitions/ROGII/dataset/"
    
    h_df, _ = load_well_pair(DATA_PATH, well_id)
    h_df = preprocess_logs(h_df)
    h_df = process_well_features(h_df)
    
    required_cols = ["inclination", "GR_mean_50", "GR_std_50"]
    for col in required_cols:
        assert col in h_df.columns, f"Missing feature: {col}"
    
    print("Success: Feature engineering verified.")

def validate_models():
    print("\n--- Validating Model Factory ---")
    from model_factory import build_lstm_model, build_transformer_model
    input_shape = (50, 10)
    
    lstm = build_lstm_model(input_shape)
    assert lstm is not None, "LSTM building failed"
    
    transformer = build_transformer_model(input_shape)
    assert transformer is not None, "Transformer building failed"
    
    print("Success: Model factory verified.")

if __name__ == "__main__":
    try:
        well_id = validate_data_loader()
        validate_features(well_id)
        validate_models()
        print("\nALL VALIDATIONS PASSED. Pipeline is robust.")
    except Exception as e:
        print(f"\nVALIDATION FAILED: {e}")
        sys.exit(1)
