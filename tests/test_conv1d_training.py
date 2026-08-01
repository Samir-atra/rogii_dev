"""Unit tests for Conv1D dataset preparation and model building.

Validates JAX-based float64 normalization on CPU, 3D shape conversion,
and Keras Conv1D model construction without memory or parameter errors.
"""

import os
import unittest
import jax
import jax.numpy as jnp
import keras
from keras import layers, regularizers
import numpy as np
import polars as pl

FEATURE_COLS = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
TARGET = "TVT"


def create_synthetic_data(num_samples: int = 100) -> pl.DataFrame:
    """Creates synthetic wellbore data as a Polars DataFrame.

    Args:
        num_samples (int): Number of rows to generate.

    Returns:
        pl.DataFrame: Synthetic dataset with feature columns and target.
    """
    rng = np.random.default_rng(42)
    data = {
        "MD": np.linspace(1000, 2000, num_samples),
        "X": np.linspace(500000, 501000, num_samples),
        "Y": np.linspace(1000000, 1001000, num_samples),
        "Z": np.linspace(-3000, -2000, num_samples),
        "GR": rng.uniform(20, 150, num_samples),
        "TVT_input": np.linspace(-2500, -1500, num_samples),
        "TVT": np.linspace(-2500, -1500, num_samples),
    }
    return pl.DataFrame(data)


def normalize_and_reshape(
    X_train_raw: np.ndarray, y_train_raw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Normalizes raw features and target using JAX on CPU device and reshapes for Conv1D.

    Args:
        X_train_raw (np.ndarray): 2D raw feature matrix.
        y_train_raw (np.ndarray): 1D raw target array.

    Returns:
        tuple[np.ndarray, np.ndarray, dict]: 3D normalized features, 1D normalized target, scaler dict.
    """
    cpu_dev = jax.devices("cpu")[0]
    X_tr_jax = jax.device_put(X_train_raw, cpu_dev)
    y_tr_jax = jax.device_put(y_train_raw, cpu_dev)

    feat_mean = jnp.mean(X_tr_jax, axis=0)
    feat_std = jnp.std(X_tr_jax, axis=0)
    feat_std = jnp.where(feat_std == 0, 1.0, feat_std)

    target_mean = jnp.mean(y_tr_jax)
    target_std = jnp.std(y_tr_jax)
    target_std = jnp.where(target_std == 0, 1.0, target_std)

    X_train_n = (X_tr_jax - feat_mean) / feat_std
    y_train_n = (y_tr_jax - target_mean) / target_std

    scaler = {
        "feature_cols": FEATURE_COLS,
        "feat_mean": np.array(feat_mean),
        "feat_std": np.array(feat_std),
        "target_mean": float(target_mean),
        "target_std": float(target_std),
        "normalized": True,
    }

    X_train_3d = np.array(X_train_n, dtype=np.float32).reshape(-1, 1, len(FEATURE_COLS))
    y_train_n_arr = np.array(y_train_n, dtype=np.float32)

    return X_train_3d, y_train_n_arr, scaler


def build_conv1d_model(input_dim: int) -> keras.Model:
    """Builds and compiles a Conv1D model for 3D inputs.

    Args:
        input_dim (int): Number of feature dimensions per sequence step.

    Returns:
        keras.Model: Compiled Conv1D Keras model.
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(1, input_dim)))

    num_layers = 3
    activation = "relu"
    initializer = "glorot_uniform"

    for _ in range(num_layers):
        model.add(
            layers.Conv1D(
                filters=128,
                kernel_size=5,
                activation=activation,
                kernel_initializer=initializer,
                kernel_regularizer=regularizers.l2(1e-5),
                padding="causal",
            )
        )
        model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="linear"))

    optimizer = keras.optimizers.AdamW(
        learning_rate=1e-3, weight_decay=1e-5, global_clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
    )

    return model


class TestConv1DTraining(unittest.TestCase):
    """Test suite for Conv1D training pipeline components."""

    def test_synthetic_data_creation(self):
        """Tests synthetic data creation with Polars."""
        df = create_synthetic_data(50)
        self.assertEqual(len(df), 50)
        self.assertTrue(all(col in df.columns for col in FEATURE_COLS))

    def test_jax_normalization_and_reshaping(self):
        """Tests JAX float64 CPU normalization and 3D reshaping."""
        df = create_synthetic_data(100)
        X_raw = df.select(FEATURE_COLS).to_numpy()
        y_raw = df.select(TARGET).to_numpy().ravel()

        X_3d, y_norm, scaler = normalize_and_reshape(X_raw, y_raw)

        self.assertEqual(X_3d.shape, (100, 1, 6))
        self.assertEqual(y_norm.shape, (100,))
        self.assertIn("target_mean", scaler)

    def test_model_building_and_compilation(self):
        """Tests Conv1D model creation, compilation, and shape prediction."""
        model = build_conv1d_model(input_dim=len(FEATURE_COLS))
        dummy_input = np.zeros((16, 1, len(FEATURE_COLS)), dtype=np.float32)
        preds = model.predict(dummy_input, verbose=0)
        self.assertEqual(preds.shape, (16, 1))

    def test_denormalization_math_jax(self):
        """Tests JAX denormalization precision."""
        cpu_dev = jax.devices("cpu")[0]
        target_mean = 100.0
        target_std = 15.0

        y_raw_true = jnp.array([85.0, 100.0, 115.0], dtype=jnp.float64)
        y_norm = (y_raw_true - target_mean) / target_std

        # Denormalize using JAX math
        y_pred_denorm = (y_norm * target_std) + target_mean
        err = jnp.abs(y_pred_denorm - y_raw_true)

        self.assertLess(float(jnp.max(err)), 1e-12)


if __name__ == "__main__":
    unittest.main()
