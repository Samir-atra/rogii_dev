"""Model factory for ROGII competition.

This module provides functions to build various sequence-based architectures
using Keras with JAX backend.
"""

import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras import layers

def build_lstm_model(input_shape, units=64, layers_count=2):
    """Builds a multi-layer LSTM model.
    
    Args:
        input_shape: Tuple of (sequence_length, features).
        units: Number of LSTM units.
        layers_count: Number of stacked LSTM layers.
        
    Returns:
        Compiled Keras model.
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    
    for i in range(layers_count):
        return_seq = (i < layers_count - 1)
        model.add(layers.LSTM(units, return_sequences=return_seq))
        model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(1)) # Predicting single TVT value
    
    model.compile(optimizer="adam", loss="mse", metrics=["rmse"])
    return model

def build_gru_model(input_shape, units=64, layers_count=2):
    """Builds a multi-layer GRU model.
    
    Args:
        input_shape: Tuple of (sequence_length, features).
        units: Number of GRU units.
        layers_count: Number of stacked GRU layers.
        
    Returns:
        Compiled Keras model.
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    
    for i in range(layers_count):
        return_seq = (i < layers_count - 1)
        model.add(layers.GRU(units, return_sequences=return_seq))
        model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(1))
    
    model.compile(optimizer="adam", loss="mse", metrics=["rmse"])
    return model

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """A single Transformer block."""
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_transformer_model(input_shape, head_size=128, num_heads=4, ff_dim=128, num_transformer_blocks=2, mlp_units=[128], dropout=0, mlp_dropout=0):
    """Builds a small Transformer-based model.
    
    Args:
        input_shape: Tuple of (sequence_length, features).
        
    Returns:
        Compiled Keras model.
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    
    model.compile(optimizer="adam", loss="mse", metrics=["rmse"])
    return model

if __name__ == "__main__":
    # Test model building
    input_shape = (50, 10) # 50 depth points, 10 features
    
    lstm = build_lstm_model(input_shape)
    print("LSTM built.")
    
    transformer = build_transformer_model(input_shape)
    print("Transformer built.")
    
    transformer.summary()
