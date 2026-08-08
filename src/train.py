"""Training script for ROGII competition.

This module orchestrates the training of sequence models on well logs.
"""

import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
import polars as pl
import numpy as np
from sklearn.model_selection import KFold
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))
from data_loader import get_all_well_ids
from data_utils import prepare_all_wells
from model_factory import build_lstm_model, build_gru_model, build_transformer_model

def run_training(model_type="lstm", epochs=10, batch_size=32):
    """Runs a training session.
    
    Args:
        model_type: One of 'lstm', 'gru', 'transformer'.
        epochs: Number of training epochs.
        batch_size: Batch size.
    """
    DATA_PATH = "/home/samer/Documents/competitions/ROGII/dataset/"
    train_well_ids = get_all_well_ids(DATA_PATH, is_train=True)
    test_well_ids = get_all_well_ids(DATA_PATH, is_train=False)
    
    feature_cols = ["GR", "X", "Y", "Z", "inclination", "GR_mean_50", "GR_std_50"]
    target_col = "TVT"
    window_size = 50
    
    print(f"Loading data for {len(train_well_ids)} train wells and {len(test_well_ids)} test wells...")
    X_train_data, y_train_data = prepare_all_wells(DATA_PATH, train_well_ids, feature_cols, target_col, window_size=window_size, is_train=True)
    X_test_data, y_test_data = prepare_all_wells(DATA_PATH, test_well_ids, feature_cols, target_col, window_size=window_size, is_train=False)
    
    # Combine datasets
    X = np.concatenate([X_train_data, X_test_data], axis=0)
    y = np.concatenate([y_train_data, y_test_data], axis=0)
    
    # Simple split
    split_index = int(0.8 * len(X))
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]
    
    input_shape = (window_size, len(feature_cols))
    
    if model_type == "lstm":
        model = build_lstm_model(input_shape)
    elif model_type == "gru":
        model = build_gru_model(input_shape)
    elif model_type == "transformer":
        model = build_transformer_model(input_shape)
    else:
        raise ValueError("Invalid model type")
    
    print(f"Starting training for {model_type}...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[keras.callbacks.EarlyStopping(patience=3)]
    )
    
    model.save(f"models/{model_type}_model.keras")
    print(f"Model saved to models/{model_type}_model.keras")

if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models")
        
    # Example run with Transformer
    run_training(model_type="transformer", epochs=5)
